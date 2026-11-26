"""Render a newly solved atmosphere and compare it through the PR13 data operator.

The input must contain the new temperature, populations, radial flow, and
continuum coefficients. No frozen line coefficient or source multiplier is
used. The time-GP covariance is shared and fixed; only RV and one amplitude are
profiled. Q is a conditional descriptive score, not a significance or evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.special import voigt_profile

from gp_halpha import load_gp_data, make_native_flux_fast
from escape.time_gp import prepare_time_gp, time_gp_statistics, time_gp_posterior

jax.config.update("jax_enable_x64", True)
C_KMS = 299792.458
C_CGS = C_KMS * 1e5
RP_CM = 1.9 * 7.1492e9
PHASE = np.arange(.30, .450001, .025)
GRIDS = {"coarse": dict(nlon=48, nlat=24, nrad_per_interval=4, naz=48, subsegments=4),
         "fine": dict(nlon=96, nlat=48, nrad_per_interval=8, naz=96, subsegments=8)}
WINDOWS = {"blue": (-40, -16), "core": (-8, 8), "red": (16, 40)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_line_coefficients(state):
    """Recompute all three H-alpha branches from the supplied physical state."""
    from exojax.database.core.broadening import doppler_sigma
    from escape.hydrogen_atom import make_hydrogen_atom, atom_natural_width_hz
    from escape.nlte_forward import halpha_coefficients

    atom = make_hydrogen_atom(15, resolved_nmax=3)
    centers = np.asarray(atom.line_frequency_hz[2:5]) / C_CGS
    gamma = np.asarray(atom_natural_width_hz(atom)[2:5]) / C_CGS
    alpha, eta = halpha_coefficients(state["bound_cm3"], centers, sum_branches=False)
    sigma = jax.vmap(lambda temperature: doppler_sigma(centers, temperature, 1.008))(
        state["temperature_k"])
    return dict(centers=centers, gamma=gamma, alpha=np.asarray(alpha), eta=np.asarray(eta),
                sigma=np.asarray(sigma))


def physical_response(baseline, candidate, edges_rp):
    """Compare actual physical fields and recomputed local line diagnostics."""
    edges = np.asarray(edges_rp)
    width = np.diff(edges) * RP_CM
    states = {"baseline": baseline, "candidate": candidate}
    quantities = {}
    for label, state in states.items():
        branch = state_line_coefficients(state)
        profile = voigt_profile(branch["centers"][None, :, None] - branch["centers"][None, None, :],
                                branch["sigma"][:, None, :], branch["gamma"][None, None, :])
        at_centers = np.sum(branch["alpha"][:, None, :] * profile, axis=-1)
        own_center = branch["alpha"] * voigt_profile(0., branch["sigma"], branch["gamma"][None])
        source = np.divide(branch["eta"], branch["alpha"],
                           out=np.full_like(branch["eta"], np.nan), where=branch["alpha"] != 0)
        quantity = {key: state[key] for key in ("temperature_k", "radial_velocity_kms")}
        if "electron_cm3" in state:
            quantity["electron_cm3"] = state["electron_cm3"]
        quantity.update(n2_cm3=np.sum(state["bound_cm3"][:, 1:3], axis=1),
                        n3_cm3=np.sum(state["bound_cm3"][:, 3:6], axis=1),
                        branch_source_per_wavenumber=source,
                        branch_own_center_radial_optical_depth=own_center * width[:, None],
                        total_line_radial_optical_depth_at_branch_centers=at_centers * width[:, None])
        quantities[label] = quantity
    def serializable(array):
        return np.where(np.isfinite(array), array, None).tolist()
    result = dict(radius_rp=((edges[:-1] + edges[1:]) / 2).tolist(),
                  branch_centers_cm=branch["centers"].tolist(),
                  convention="Local branch S=eta/alpha in the wavenumber convention. Optical depths use each shell's radial width and rest-frame line profiles, without continuum or intercell Doppler shifts; actual observer transfer retains all velocities. Ratios are candidate/baseline. Undefined zero-denominator ratios are null.",
                  layers={})
    for key, original in quantities["baseline"].items():
        if key not in quantities["candidate"]:
            continue
        changed = quantities["candidate"][key]
        ratio = np.divide(changed, original, out=np.full_like(changed, np.nan), where=original != 0)
        result["layers"][key] = {"baseline": serializable(original), "candidate": serializable(changed),
                                 "ratio": serializable(ratio)}
    result["primary_cells"] = {"inner_emitter": 3, "upper_receiver": 4}
    return result


def render_state(state, edges_rp, phase, grid, zonal_kms, profile_step_kms=.05):
    """Same dayside curved-disk plus halo observer as the archived PR13 run."""
    from numba import njit
    from escape.longitude_map import (disk_rays, halo_rays, observer_basis,
                                       chord_segments, projected_velocity)

    @njit(cache=True)
    def integrate(starts, cells, lengths, los, areas, day, table, x0, dx, offset,
                  alpha, eta, continuum_chi, continuum_eta, boundary):
        result = np.zeros(len(boundary))
        for ray in range(len(areas)):
            for wave in range(len(boundary)):
                foreground = 1.
                for segment in range(starts[ray], starts[ray + 1]):
                    cell = cells[segment]
                    chi_line, eta_line = 0., 0.
                    for branch in range(3):
                        q = (offset[branch, wave] + los[segment] - x0) / dx
                        index = int(q)
                        fraction = q - index
                        phi = table[cell, branch, index] * (1 - fraction)
                        phi += table[cell, branch, index + 1] * fraction
                        chi_line += alpha[cell, branch] * phi
                        eta_line += eta[cell, branch] * phi
                    chi = chi_line + continuum_chi[cell, wave]
                    length = lengths[segment]
                    factor = -np.expm1(-chi * length) / chi if chi != 0 else length
                    result[wave] += areas[ray] * foreground * factor * (
                        eta_line + continuum_eta[cell, wave])
                    foreground *= np.exp(-chi * length)
                if day[ray]:
                    result[wave] += areas[ray] * boundary[wave] * (foreground - 1.)
        return result

    edges = np.asarray(edges_rp)
    radii = (edges[:-1] + edges[1:]) / 2
    observer = observer_basis(phase)[0]
    rotation = 2 * np.pi * RP_CM / (1.48 * 86400) / 1e5
    rays = list(disk_rays(phase, grid["nlon"], grid["nlat"]))
    rays += list(halo_rays(phase, edges, grid["nrad_per_interval"], grid["naz"]))
    starts, cells, lengths, los, areas, day = [0], [], [], [], [], []
    for impact, area, base in rays:
        areas.append(area)
        day.append(base is not None)
        for cell, left, right in reversed(chord_segments(impact, observer, edges)):
            width = (right - left) / grid["subsegments"]
            for n in reversed(range(grid["subsegments"])):
                position = impact + (left + (n + .5) * width) * observer
                cells.append(cell)
                lengths.append(width * RP_CM)
                los.append(projected_velocity(position, observer, state["radial_velocity_kms"][cell],
                                               rotation, zonal_kms, radii[cell]))
        starts.append(len(cells))
    branches = state_line_coefficients(state)
    wn = state["wavenumber_cm"]
    offset = C_KMS * (wn[None, :] / branches["centers"][:, None] - 1)
    extent = np.ceil(np.max(abs(offset)) + np.max(np.abs(los)) + 2)
    tv = np.arange(-extent, extent + profile_step_kms, profile_step_kms)
    table = np.stack([voigt_profile(tv[None, :] * branches["centers"][:, None] / C_KMS,
                                    sigma[:, None], branches["gamma"][:, None])
                      for sigma in branches["sigma"]])
    planck = lambda temperature: (3.741771790075259e-5 * wn**3
                                  / np.expm1(1.4387773538277202 * wn / temperature) / np.pi)
    raw = integrate(*map(np.asarray, (starts, cells, lengths, los, areas, day)),
                    table, tv[0], profile_step_kms, offset, branches["alpha"], branches["eta"],
                    state["continuum_extinction"], state["continuum_emission_hz"] * C_CGS,
                    planck(4500.))
    sigma_ip = C_KMS / 98000 / np.sqrt(8 * np.log(2))
    half = int(np.ceil(6 * sigma_ip / .2))
    kernel = np.exp(-.5 * (np.arange(-half, half + 1) * .2 / sigma_ip)**2)
    kernel /= kernel.sum()
    return (.08**2 * np.convolve(raw, kernel, mode="same") / (np.pi * planck(10000.)))[::-1]


def setup(phase, velocity, paths, surface, noise, pixel_offset=0):
    groups = load_gp_data(paths, pixel_offset=pixel_offset)
    states, matrices, scales, fluxes = [], [], [], []
    for group in groups:
        k = group["night"] * 2 + group["order"]
        basis = np.array([np.ones_like(group["stellar_velocity"]),
                          (group["stellar_velocity"] - 250) / 140])
        state = prepare_time_gp(group["weight"], basis)
        states.append(state)
        scale = np.exp(noise["log_white"][k])
        scales.append(scale)
        distance = np.sqrt(3) * abs(group["time_min"][:, None] - group["time_min"][None, :])
        distance /= np.exp(noise["log_length"][k])
        factor = la.block_diag(*[amplitude * la.cholesky(
            (1 + distance) * np.exp(-distance) + 1e-10 * np.eye(len(distance)), lower=True)
            for amplitude in np.exp(noise["log_gp"][k])])
        system = np.eye(len(factor)) + factor.T @ np.asarray(state.gram) @ factor / scale**2
        matrices.append(jnp.asarray(factor @ la.solve(system, factor.T, assume_a="pos") / scale**4))
        fluxes.append(make_native_flux_fast(velocity, surface, group))

    def models(spectra, rv):
        output = []
        for group, flux in zip(groups, fluxes):
            sampled = jax.vmap(lambda row: jnp.interp(group["phase"], phase, row),
                               in_axes=1, out_axes=1)(spectra)
            sampled = jnp.where(group["science"][:, None], sampled, 0.)
            output.append(jnp.log(flux(sampled, rv)))
        return tuple(output)

    def group_q(spectra, rv):
        result = []
        for group, state, model, matrix, scale in zip(groups, states, models(spectra, rv), matrices, scales):
            norm, product = time_gp_statistics(group["log_data"] - model, state)
            result.append(norm / scale**2 - product @ matrix @ product)
        return jnp.stack(result)

    return dict(groups=groups, states=states, models=jax.jit(models), noise=noise,
                group_q=jax.jit(group_q), q=jax.jit(lambda spectra, rv: group_q(spectra, rv).sum()))


def fixed_fit(problem, spectra):
    spectra = jnp.asarray(spectra)
    objective = jax.jit(jax.value_and_grad(lambda p: problem["q"](spectra * jnp.exp(p[1]), p[0])))
    fits = [minimize(objective, [rv, 0.], jac=True, bounds=[(-15, 15), (-3, 3)], method="L-BFGS-B",
                     options=dict(maxiter=120, ftol=1e-12, gtol=1e-6)) for rv in [0., 3.]]
    fit = min(fits, key=lambda item: item.fun)
    return dict(parameters=fit.x.tolist(), Q=float(fit.fun), success=bool(fit.success), message=str(fit.message),
                group_Q=np.asarray(problem["group_q"](spectra * np.exp(fit.x[1]), fit.x[0])).tolist())


def projections(problem, spectra, parameters, edges):
    """Common-reference projection with the original fixed display bins.

    Night/order summaries retain the globally profiled RV/amplitude. White
    errors only provide display weights; Q retains the full GP covariance.
    """
    rv, log_amplitude = parameters
    models = problem["models"](spectra * np.exp(log_amplitude), rv)
    sums = []
    for group, state, model in zip(problem["groups"], problem["states"], models):
        k = group["night"] * 2 + group["order"]
        noise = problem["noise"]
        posterior = time_gp_posterior(group["log_data"] - model, state, group["time_min"],
            np.exp(noise["log_gp"][k]), np.exp(noise["log_length"][k]), np.exp(noise["log_white"][k]))
        orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
        velocity = C_KMS * ((1 + group["stellar_velocity"][None] / C_KMS)
                            / (1 + orbital[:, None] / C_KMS) - 1)
        index = np.searchsorted(edges, velocity, side="right") - 1
        weight = group["weight"]
        coefficient = np.array([weight * group["science"][:, None] * (index == b)
                                for b in range(len(edges) - 1)])
        common = weight / np.where(weight.sum(axis=0) > 0, weight.sum(axis=0), 1)
        projected = coefficient - common[None] * coefficient.sum(axis=1)[:, None]
        gp = np.asarray(posterior.mean).T @ np.asarray(state.basis)
        sums.append(dict(weight=coefficient.sum(axis=(1, 2)),
                         observed=np.einsum("bep,ep->b", projected, group["log_data"]),
                         planet=np.einsum("bep,ep->b", projected, model),
                         total=np.einsum("bep,ep->b", projected, np.asarray(model) + gp),
                         variance=np.exp(2 * noise["log_white"][k]) * np.sum(
                             projected**2 / np.where(weight > 0, weight, np.inf), axis=(1, 2))))
    selections = {"all": range(4), "night0": [0, 1], "night1": [2, 3],
                  **{f"night{i // 2}_order{i % 2}": [i] for i in range(4)}}
    output = {}
    for label, indices in selections.items():
        combined = {key: sum(sums[i][key] for i in indices) for key in sums[0]}
        keep = combined["weight"] > 0
        norm = combined["weight"][keep]
        row = {key: combined[key][keep] / norm for key in ("observed", "planet", "total")}
        row.update(velocity=((edges[:-1] + edges[1:]) / 2)[keep],
                   error=np.sqrt(combined["variance"][keep]) / norm)
        row["residual"] = row["observed"] - row["total"]
        output[label] = row
    return output


def band_means(row):
    result = {}
    for name, (lower, upper) in WINDOWS.items():
        keep = (row["velocity"] >= lower) & (row["velocity"] < upper)
        result[name] = {key: (float(np.average(row[key][keep], weights=1 / row["error"][keep]**2)) * 1e6
                              if np.any(keep) else None)
                        for key in ("observed", "planet", "total", "residual")}
    return result


def compare_templates(problem, spectra, edges):
    results, projected = {}, {}
    for name, value in spectra.items():
        fit = fixed_fit(problem, value)
        rows = projections(problem, value, fit["parameters"], edges)
        fit["bands_ppm"] = {label: band_means(row) for label, row in rows.items()}
        fit["night_Q"] = [sum(fit["group_Q"][:2]), sum(fit["group_Q"][2:])]
        results[name] = fit
        projected.update({f"{name}__{label}__{key}": data for label, row in rows.items() for key, data in row.items()})
        print(json.dumps(dict(template=name, Q=fit["Q"], parameters=fit["parameters"], bands_ppm=fit["bands_ppm"]["all"])), flush=True)
    if "candidate" in results:
        results["candidate"]["delta_Q_improvement"] = results["baseline"]["Q"] - results["candidate"]["Q"]
        rv, amplitude = results["baseline"]["parameters"]
        group_q = np.asarray(problem["group_q"](spectra["candidate"] * np.exp(amplitude), rv))
        rows = projections(problem, spectra["candidate"], [rv, amplitude], edges)
        results["candidate"]["fixed_at_baseline_nuisance"] = dict(
            parameters=[rv, amplitude], Q=float(group_q.sum()), group_Q=group_q.tolist(),
            night_Q=[float(group_q[:2].sum()), float(group_q[2:].sum())],
            bands_ppm={label: band_means(row) for label, row in rows.items()})
        projected.update({f"candidate_fixed__{label}__{key}": data
                          for label, row in rows.items() for key, data in row.items()})
    return results, projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-npz", type=Path)
    parser.add_argument("--candidate-spectra", type=Path, help="Reanalyze saved spectra.npz without repeating transfer.")
    parser.add_argument("--candidate-render-review", type=Path,
                        help="Required with --candidate-spectra; its output hash must match.")
    parser.add_argument("--root-review", type=Path)
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz"))
    parser.add_argument("--baseline-state", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz"))
    parser.add_argument("--baseline-physical", type=Path,
                        help="Optional matching physical snapshot supplying baseline electron_cm3.")
    parser.add_argument("--noise-samples", type=Path, default=Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz"))
    parser.add_argument("--stellar-surface", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--display-bins", type=Path, default=Path("/tmp/escape-gp-combined-20261125/combined_arrays.npz"))
    parser.add_argument("--observations", nargs=2, type=Path, default=[Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")])
    parser.add_argument("--grid", choices=GRIDS, default="fine")
    parser.add_argument("--zonal-kms", type=float, default=19.99871490128246)
    parser.add_argument("--profile-step-kms", type=float, default=.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.state_npz and args.candidate_spectra:
        parser.error("Choose a physical snapshot or saved candidate spectra, not both")
    if args.candidate_spectra and not args.candidate_render_review:
        parser.error("--candidate-spectra requires --candidate-render-review")
    if not np.isfinite(args.profile_step_kms) or args.profile_step_kms <= 0:
        parser.error("--profile-step-kms must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repository = Path(__file__).resolve().parents[1]
    operator_sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
                        ("gp_halpha.py", "phase_halpha.py", "cloudy_halpha.py")],
                        *[repository / "src" / "escape" / name for name in
                          ("longitude_map.py", "nlte_forward.py", "hydrogen_atom.py",
                           "hydrogen_rates.py", "time_gp.py", "stellar_radiation.py")]]
    source_hashes = {str(path): sha256(path) for path in operator_sources}
    archive = dict(np.load(args.baseline))
    spectra = {"baseline": archive["fine_intrinsic_phases"][:7]}
    velocity = archive["model_velocity_kms"]
    rendering = dict(grid=GRIDS[args.grid], profile_step_kms=args.profile_step_kms,
                     zonal_kms=args.zonal_kms)
    paths = [args.baseline, args.recipe, args.noise_samples, args.stellar_surface, args.display_bins,
             *args.observations, *operator_sources]
    if args.root_review:
        paths.append(args.root_review)
    response = None
    if args.candidate_spectra:
        render_review = json.loads(args.candidate_render_review.read_text())
        response = render_review.get("physical_response")
        if render_review["output_sha256"]["spectra.npz"] != sha256(args.candidate_spectra):
            raise ValueError("Saved candidate spectra do not match their rendering review")
        saved = dict(np.load(args.candidate_spectra))
        np.testing.assert_array_equal(saved["phase"], PHASE)
        np.testing.assert_array_equal(saved["velocity"], velocity)
        spectra["candidate"] = saved["candidate"]
        rendering = {key: render_review[key] for key in rendering}
        if spectra["candidate"].shape != spectra["baseline"].shape or not np.isfinite(spectra["candidate"]).all():
            raise ValueError("Saved candidate spectra have incorrect shape or nonfinite values")
        paths.extend([args.candidate_spectra, args.candidate_render_review])
    if args.state_npz:
        paths.append(args.state_npz)
        state = dict(np.load(args.state_npz))
        required = ("temperature_k", "bound_cm3", "radial_velocity_kms", "wavenumber_cm",
                    "continuum_extinction", "continuum_emission_hz")
        for key in required:
            if key not in state or not np.isfinite(state[key]).all():
                raise ValueError(f"Missing or nonfinite new-state field: {key}")
        np.testing.assert_allclose((C_KMS * (15233.146 / state["wavenumber_cm"] - 1))[::-1], velocity,
                                   rtol=0, atol=1e-10)
        edges = json.loads(args.recipe.read_text())["edges_rp"]
        cells, frequencies = len(edges) - 1, len(state["wavenumber_cm"])
        for key, shape in (("temperature_k", (cells,)), ("radial_velocity_kms", (cells,)),
                           ("bound_cm3", (cells, 18)), ("continuum_extinction", (cells, frequencies)),
                           ("continuum_emission_hz", (cells, frequencies))):
            if state[key].shape != shape:
                raise ValueError(f"New-state shape mismatch for {key}: expected {shape}")
        if np.any(state["temperature_k"] <= 0) or np.any(state["bound_cm3"] < 0):
            raise ValueError("New-state temperature must be positive and bound populations nonnegative.")
        baseline_state = dict(np.load(args.baseline_state))
        paths.append(args.baseline_state)
        if args.baseline_physical:
            physical = dict(np.load(args.baseline_physical))
            for key in ("temperature_k", "bound_cm3", "radial_velocity_kms"):
                np.testing.assert_allclose(physical[key], baseline_state[key], rtol=1e-12, atol=0)
            baseline_state["electron_cm3"] = physical["electron_cm3"]
            paths.append(args.baseline_physical)
        response = physical_response(baseline_state, state, edges)
        rows = []
        identity = dict(state_sha256=sha256(args.state_npz), source_sha256=sha256(__file__),
                        operator_source_sha256=source_hashes, recipe_sha256=sha256(args.recipe),
                        grid=GRIDS[args.grid], zonal_kms=args.zonal_kms, profile_step_kms=args.profile_step_kms)
        for index, phase in enumerate(PHASE):
            cached = args.output / f"phase_{index:02d}.npz"
            if cached.exists():
                saved = dict(np.load(cached))
                if json.loads(str(saved["identity"])) != identity:
                    raise ValueError(f"Cached phase belongs to a different state or renderer: {cached}")
                row = saved["spectrum"]
            else:
                row = render_state(state, edges, phase, GRIDS[args.grid], args.zonal_kms, args.profile_step_kms)
                if not np.isfinite(row).all():
                    raise FloatingPointError("Nonfinite new-state observer spectrum.")
                np.savez_compressed(cached, phase=phase, spectrum=row, identity=json.dumps(identity))
            rows.append(row)
            print(f"Rendered phase {phase:.3f}; elapsed {time.monotonic() - started:.1f} s", flush=True)
        spectra["candidate"] = np.array(rows)
    surface = dict(np.load(args.stellar_surface))
    samples = dict(np.load(args.noise_samples))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in samples.items()}
    edges = np.load(args.display_bins)["edges"]
    selections, projected, ranks = {}, {}, {}
    for label, offset in (("primary", 0), ("alternate_pixels", 1)):
        problem = setup(PHASE, velocity, args.observations, surface, noise, pixel_offset=offset)
        results, rows = compare_templates(problem, spectra, edges)
        selections[label] = results
        ranks[label] = sum(int(state.rank) for state in problem["states"])
        prefix = "" if offset == 0 else "alternate_pixels__"
        projected.update({prefix + key: value for key, value in rows.items()})
    np.savez_compressed(args.output / "spectra.npz", phase=PHASE, velocity=velocity, **spectra)
    np.savez_compressed(args.output / "projected.npz", **projected)
    report = dict(scope="New-state spherical observer with the unchanged PR13 native samples, fixed GP covariance and shared RV/amplitude profiling. Conditional scores and displayed residuals are descriptive, not significances or evidence.",
                  physical_state_scope="The input is a supplied physical snapshot. Physical-root qualification must come from its separate root review; this observer driver does not establish equilibrium.",
                  observer="Dayside curved disk plus halo; prescribed dark transparent nightside; same 4500 K inner boundary, rotation, zonal flow, R=98000 convolution and stellar normalization.",
                  **rendering,
                  rank=ranks["primary"], rank_by_selection=ranks, windows_kms=WINDOWS,
                  nuisance_bounds=dict(rv_kms=[-15, 15], log_amplitude=[-3, 3]),
                  sensitivity_scope="Alternate pixels use the same archived GP hyperparameters, with the covariance/projector recomputed for that pixel selection; covariance is identical between the two templates within each selection.",
                  band_convention="Inverse-white-variance means in the original display bins; total=planet+conditional GP, residual=data-total. By-night/order summaries retain the globally profiled nuisance parameters.",
                  physical_response=response,
                  results=selections["primary"], selections=selections,
                  inputs={str(path): sha256(path) for path in paths},
                  output_sha256={name: sha256(args.output / name) for name in ("spectra.npz", "projected.npz")},
                  elapsed_s=time.monotonic() - started)
    if any(sha256(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError("An observation or rendering source changed during the run.")
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
