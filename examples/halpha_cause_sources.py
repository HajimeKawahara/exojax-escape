"""Frozen source/opacity responses on the sealed complete-phase H-alpha maps.

These are line-coefficient diagnostics, not new statistical-equilibrium or
thermal solutions. Primary physical-template comparisons retain the original
RV and amplitude; separate sensitivities profile those two global parameters.
The conditional GP mean is recomputed from the full data in both cases.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import voigt_profile

from halpha_geometry_fit import setup, PHASE, sha256, fixed_fit
from halpha_data_consistency import PARAMETERS
from halpha_noise_audit import covariance, coefficient_mean
from escape.time_gp import time_gp_statistics
from escape.hydrogen_atom import make_hydrogen_atom

jax.config.update("jax_enable_x64", True)
ARCHIVE = Path("/home/kawahara/exojax-escape/results/contrast_response_20261126")
INPUT = Path("/tmp/escape-e8-mdot-sphere/fine")
TAU = dict(base=1., thinner=float(np.exp(-.7)), thicker=float(np.exp(.7)))
CASES = dict(baseline=(1., 1., "base"),
    inner_source_0p8=(.8, 1., "base"), inner_source_1p2=(1.2, 1., "base"),
    inner_source_1p5=(1.5, 1., "base"), upper_source_0p8=(1., .8, "base"),
    upper_source_1p2=(1., 1.2, "base"), upper_source_half=(1., .5, "base"),
    upper_opacity_thinner=(1., 1., "thinner"), upper_opacity_thicker=(1., 1., "thicker"),
    increased_source_contrast=(1.5, .5, "base"),
    upper_lower_level_accumulation=(1., float(np.exp(-.7)), "thicker"))


def make_context(velocity, selection, maps_directory):
    offset = 0 if selection == "primary" else 1
    paths = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    stellar = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    noise_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    noise = {key: np.median(value, axis=(0, 1)) for key, value in dict(np.load(noise_path)).items()}
    problem = setup(PHASE, velocity, paths, dict(np.load(stellar)), noise, pixel_offset=offset)
    metadata = json.loads((maps_directory / "rows.json").read_text())
    selected = [i for i, row in enumerate(metadata["rows"]) if row["scope"] == "all" and
        ((row["grid"] == "fine" and row["product"] in ("common_complete", "band_complete"))
         or (row["grid"] == "coarse" and row["product"] == "common_complete"))]
    rows = [metadata["rows"][i] for i in selected]
    supported = np.array([metadata["selections"][selection]["rows"][i]["supported"] for i in selected])
    with np.load(maps_directory / "maps.npz") as saved:
        maps = []
        for i, group in enumerate(problem["groups"]):
            for key in ("phase", "stellar_velocity", "pixel_index", "weight", "science", "eclipse", "log_data"):
                np.testing.assert_array_equal(saved[f"{selection}__g{i}__{key}"], group[key])
            maps.append(saved[f"{selection}__g{i}__maps"][selected])
    target = np.array([i for i, row in enumerate(rows) if row["grid"] == "fine" and
        row["product"] == "common_complete" and row["phase_index"] == 3])
    assert len(target) == 3 and supported[target].all()
    target_support = [metadata["selections"][selection]["rows"][selected[i]] for i in target]
    assert all(x["group_exposure_ids"] == target_support[0]["group_exposure_ids"] for x in target_support)
    covs = [covariance(state, group, noise, i) for i, (state, group) in
            enumerate(zip(problem["states"], problem["groups"]))]
    return dict(problem=problem, covs=covs, maps=maps, rows=rows, supported=supported,
        target=target, target_support=target_support[0], inputs=[*paths, stellar, noise_path])


def evaluate(context, spectra, parameters=PARAMETERS):
    problem, maps = context["problem"], context["maps"]
    models = problem["models"](spectra * np.exp(parameters[1]), parameters[0])
    values = {name: np.zeros(len(context["rows"])) for name in ("observed", "planet", "gp")}
    q = []
    for group, state, cov, model, mapping in zip(problem["groups"], problem["states"], context["covs"], models, maps):
        residual = group["log_data"] - model
        gp = coefficient_mean(group, state, cov, residual).T @ np.asarray(state.basis)
        norm, product = time_gp_statistics(residual, state)
        q.append(float(norm / cov["scale"]**2 - np.asarray(product) @ cov["correction"] @ np.asarray(product)))
        for key, field in (("observed", group["log_data"]), ("planet", model), ("gp", gp)):
            values[key] += np.einsum("rep,ep->r", mapping, field) * 1e6
    values["total"] = values["planet"] + values["gp"]
    values["residual"] = values["observed"] - values["total"]
    return dict(Q=float(sum(q)), group_Q=q, parameters=np.asarray(parameters).tolist(), **values)


def fine_components(phase_index, opacity, destination):
    """Reuse the inspected archived ray integrator with current identical geometry."""
    import escape.longitude_map as geometry
    geometry_path = Path(geometry.__file__)
    old_geometry = Path("/tmp/escape-phase-night-pr/src/escape/longitude_map.py")
    if sha256(geometry_path) != sha256(old_geometry):
        raise ValueError("Archived frozen ray geometry differs from the current observer.")
    identity = {str(path): sha256(path) for path in
        (ARCHIVE / "render_response.py", INPUT / "fixed_inputs.npz", INPUT / "branch_arrays.npz", geometry_path)}
    identity.update(phase=float(PHASE[phase_index]), opacity=opacity)
    cache = destination / f"fine_phase_{phase_index:02d}_tau_{opacity:.8f}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError("Fine response cache identity changed.")
            return saved["components"]
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(ARCHIVE))
        renderer = importlib.import_module("render_response")
    finally:
        sys.path[:] = original_path
    started = time.monotonic()
    dx = .05
    velocity = np.arange(-250., 250. + dx / 2, dx)
    branch = renderer.A
    table = np.stack([voigt_profile(velocity[None] * branch["branch_centers"][:, None] / renderer.C,
        branch["branch_sigma"][i, :, None], branch["branch_gamma"][:, None]) for i in range(8)])
    offset = renderer.C * (renderer.WN[None] / branch["branch_centers"][:, None] - 1)
    packed = renderer.pack_rays(PHASE[phase_index], renderer.PLAN["grid"])
    if np.max(abs(offset)) + np.max(abs(packed[3]) + abs(packed[4])) >= 249.:
        raise ValueError("The line-profile interpolation table is too narrow.")
    raw = renderer.integrate(*packed, table, velocity[0], dx, offset,
        branch["branch_alpha"], branch["branch_eta"], renderer.F["continuum_extinction"],
        renderer.F["continuum_emission_hz"] * renderer.C * 1e5,
        renderer.BOUNDARY, renderer.INNER, opacity, 1.)
    components = np.array([.08**2 * np.convolve(row, renderer.KERNEL, mode="same") / renderer.STAR for row in raw])[:, ::-1]
    np.savez_compressed(cache, components=components, identity=json.dumps(identity))
    print(f"Fine response phase={PHASE[phase_index]:.3f}, tau={opacity:.4f}: {time.monotonic()-started:.1f}s", flush=True)
    return components


def fit_sources(context, baseline, components):
    """Fit two nonnegative source coefficients at the original global RV/A."""
    base, parts = jnp.asarray(baseline), jnp.asarray(components)
    def make_spectrum(p):
        return base + (p[0] - 1) * parts[:, 0] + (p[1] - 1) * parts[:, 1]
    objective = jax.jit(jax.value_and_grad(lambda p:
        context["problem"]["q"](make_spectrum(p) * jnp.exp(PARAMETERS[1]), PARAMETERS[0])))
    fits = [minimize(objective, start, jac=True, bounds=[(0., 3.), (0., 3.)],
        method="L-BFGS-B", options=dict(maxiter=100, ftol=1e-12, gtol=1e-6))
        for start in ((1., 1.), (1.5, .5))]
    best = min(fits, key=lambda value: value.fun)
    return np.asarray(make_spectrum(best.x)), dict(inner_upper_source_multipliers=best.x.tolist(),
        bounds=[0., 3.], Q=float(best.fun), success=bool(best.success), message=str(best.message),
        gradient=np.asarray(best.jac).tolist(),
        scope="Two source coefficients fitted to all native data, with the original RV/amplitude and GP covariance fixed. Bounds keep emissivities nonnegative; this is a frozen shape diagnostic, not an equilibrium fit.")


def local_physics(state, recipe):
    atom = make_hydrogen_atom(15, resolved_nmax=3)
    upper, lower = np.asarray(atom.line_upper[2:5]), np.asarray(atom.line_lower[2:5])
    weight = np.asarray(atom.statistical_weights)
    ratio = state["bound_cm3"][:, upper] / state["bound_cm3"][:, lower] * weight[lower] / weight[upper]
    energy_k = np.asarray(atom.line_frequency_hz[2:5]) * 6.62607015e-27 / 1.380649e-16
    edges = np.asarray(recipe["edges_rp"])
    radius = (edges[1:] + edges[:-1]) / 2
    variants = {}
    for name, (inner, outer, tau_name) in CASES.items():
        source = np.where(radius < 1.3, inner, outer)[:, None]
        opacity = np.where(radius < 1.3, 1., TAU[tau_name])[:, None]
        new_ratio = source * ratio / (1 - ratio + source * ratio)
        variants[name] = dict(implied_n3_multiplier=(source * opacity).ravel().tolist(),
            implied_n2_multiplier_per_branch=(opacity * (1 + (source - 1) * ratio)).tolist(),
            branch_excitation_temperature_k=(energy_k[None] / np.log(1 / new_ratio)).tolist())
    return dict(radius_rp=radius.tolist(), stimulated_fraction=ratio.tolist(),
        baseline_branch_excitation_temperature_k=(energy_k[None] / np.log(1 / ratio)).tolist(), variants=variants,
        interpretation="eta is proportional to n_u and alpha to (g_u/g_l)n_l-n_u. The implied lower-level multipliers are per branch; shared lower levels need a consistent population solve. Excitation temperatures are ratios of level populations, not gas temperatures. Inner source changes also perturb the prescribed LTE cells and are therefore diagnostics, not permitted equilibrium solutions.")


def serializable(result, context, baseline):
    target = context["target"]
    outside = np.array([row["grid"] == "fine" and row["product"] == "common_complete"
                       and row["phase_index"] != 3 for row in context["rows"]]) & context["supported"]
    delta = result["total"] - baseline["total"]
    return dict(Q=result["Q"], delta_Q=result["Q"] - baseline["Q"], group_Q=result["group_Q"],
        parameters=result["parameters"],
        target_ppm={key: value[target].tolist() for key, value in result.items() if isinstance(value, np.ndarray)},
        target_change_ppm={key: (result[key][target] - baseline[key][target]).tolist() for key in ("planet", "gp", "total", "residual")},
        blue_up_core_down=bool(delta[target[0]] > 0 and delta[target[1]] < 0),
        outside_target_maximum_complete_row_change_ppm=float(np.max(abs(delta[outside]))),
        phase_row_values_ppm={key: [float(value) if keep else None
            for value, keep in zip(result[key], context["supported"])]
            for key in ("planet", "gp", "total", "residual")})


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    baseline_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    recipe_path = Path("/tmp/escape-corrected-observer-full-f/recipe.json")
    archived = dict(np.load(baseline_path))
    baseline_spectra, velocity = archived["fine_intrinsic_phases"][:7], archived["model_velocity_kms"]
    paths = [Path(__file__), baseline_path, recipe_path, INPUT / "fixed_inputs.npz", INPUT / "branch_arrays.npz", INPUT / "plan.json",
        args.maps / "maps.npz", args.maps / "rows.json", ARCHIVE / "input_manifest.json", ARCHIVE / "render_response.py",
        ARCHIVE / "control_verification.json", ARCHIVE / "free_verified.npz", ARCHIVE / "fine_control_components.npz",
        ARCHIVE / "fine/phase_03.npz", root / "mceg/evidence/halpha_geometry_fit_20261126/free_model_reference.json",
        Path("/tmp/escape-halpha-geometry-fit/final_cpu_fine/spectra.npz"),
        *[root / "examples" / name for name in ("halpha_geometry_fit.py", "halpha_noise_audit.py", "gp_halpha.py", "phase_halpha.py", "cloudy_halpha.py")],
        *[root / "src/escape" / name for name in ("time_gp.py", "longitude_map.py", "hydrogen_atom.py")]]
    components = {key: [] for key in TAU}
    for i in range(7):
        first, opacity = ARCHIVE / f"coarse/phase_{i:02d}.npz", ARCHIVE / f"coarse/opacity_{i:02d}.npz"
        paths.extend((first, opacity))
        with np.load(first) as saved:
            np.testing.assert_allclose(saved["phase"], PHASE[i], atol=1e-14, rtol=0)
            np.testing.assert_allclose(saved["velocity"], velocity, atol=1e-10, rtol=0)
            components["base"].append(saved["base"])
        with np.load(opacity) as saved:
            np.testing.assert_allclose(saved["components"][3], components["base"][-1], atol=1e-14, rtol=1e-12)
            components["thinner"].append(saved["components"][1])
            components["thicker"].append(saved["components"][5])
            np.testing.assert_allclose(saved["log_scales"][[1, 5]], [-.7, .7], atol=1e-14, rtol=0)
    components = {key: np.array(value) for key, value in components.items()}
    contexts = {label: make_context(velocity, label, args.maps) for label in ("primary", "alternate")}
    paths.extend(contexts["primary"]["inputs"])
    pins = {str(path.resolve()): sha256(path) for path in paths}
    old_manifest = json.loads((ARCHIVE / "input_manifest.json").read_text())
    for path in (baseline_path, INPUT / "fixed_inputs.npz", INPUT / "branch_arrays.npz", *contexts["primary"]["inputs"]):
        if old_manifest[str(path)] != sha256(path):
            raise ValueError("A frozen response input differs from its original manifest.")
    baselines = {label: evaluate(context, baseline_spectra) for label, context in contexts.items()}
    assert abs(baselines["primary"]["Q"] - 38326.64894003179) < 1e-6

    def spectrum(parameters, source_components):
        inner, upper, opacity = parameters
        value = source_components[opacity]
        return baseline_spectra + inner * value[:, 0] + upper * value[:, 1] + value[:, 2] - source_components["base"].sum(axis=1)

    coarse_spectra = {name: spectrum(value, components) for name, value in CASES.items()}
    for name, spectra in coarse_spectra.items():
        result = evaluate(contexts["primary"], spectra)
        print(json.dumps(dict(stage="coarse", case=name,
            target_residual_ppm=result["residual"][contexts["primary"]["target"]].tolist())), flush=True)
    refined = {key: value.copy() for key, value in components.items()}
    refined["base"][3] = np.load(ARCHIVE / "fine/phase_03.npz")["base"]
    refined["thicker"][3] = np.load(ARCHIVE / "fine_control_components.npz")["upper_tau"]
    for i, key in ((3, "thinner"), (4, "base"), (4, "thinner"), (4, "thicker")):
        refined[key][i] = fine_components(i, TAU[key], args.output)
    fine_source = {key: value.copy() for key, value in refined.items()}
    for i in (0, 1, 2, 5, 6):
        fine_source["base"][i] = fine_components(i, 1., args.output)
    baseline_reproduction = float(np.max(abs(fine_source["base"].sum(axis=1) - baseline_spectra))) * 1e6
    spectra = {name: spectrum(value, fine_source if value[2] == "base" else refined) for name, value in CASES.items()}
    references = dict(np.load(ARCHIVE / "free_verified.npz"))
    np.testing.assert_allclose(references["phase"], PHASE, rtol=0, atol=1e-14)
    np.testing.assert_allclose(references["velocity"], velocity, rtol=0, atol=1e-10)
    original = json.loads((root / "mceg/evidence/halpha_geometry_fit_20261126/free_model_reference.json").read_text())["reference_rows"]
    spectra["previous_coupled_Halpha"] = np.load("/tmp/escape-halpha-geometry-fit/final_cpu_fine/spectra.npz")["candidate"]
    spectra["free_model1_original"] = references["original_spectra"]
    spectra["free_model1_refitted_shape"] = references["spectra"]
    selection_results, checks, all_rows = {}, {}, {}
    for label, context in contexts.items():
        base = baselines[label]
        evaluated = {}
        for name, template in spectra.items():
            parameters = PARAMETERS
            if name.startswith("free_model1"):
                parameters = original["free_original" if name.endswith("original") else "free_refitted"]["parameters"]
            result = evaluate(context, template, parameters)
            evaluated[name] = result
            all_rows.update({label + "__" + name + "__" + key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        refinement = {}
        for name, template in coarse_spectra.items():
            coarse = evaluate(context, template)
            refinement[name] = dict(target_total_difference_ppm=(evaluated[name]["total"][context["target"]] - coarse["total"][context["target"]]).tolist(),
                maximum_target_response_spectrum_difference_ppm=float(np.max(abs(spectra[name][[3, 4]] - template[[3, 4]]))) * 1e6)
        derivative = np.column_stack([(evaluated[p]["total"][context["target"][:2]] - evaluated[m]["total"][context["target"][:2]]) / .4
            for m, p in (("inner_source_0p8", "inner_source_1p2"), ("upper_source_0p8", "upper_source_1p2"))])
        required = 1 + np.linalg.solve(derivative, base["residual"][context["target"][:2]])
        source_template, source_fit = fit_sources(context, baseline_spectra, fine_source["base"])
        source_evaluation = evaluate(context, source_template)
        source_fit["response"] = serializable(source_evaluation, context, base)
        profiled_names = ["baseline", "upper_source_half", "increased_source_contrast",
            "upper_lower_level_accumulation", "free_model1_original", "free_model1_refitted_shape"]
        profiled = {}
        profile_baseline = None
        for name in profiled_names:
            fit = fixed_fit(context["problem"], spectra[name])
            value = evaluate(context, spectra[name], fit["parameters"])
            if name == "baseline":
                profile_baseline = value
            profiled[name] = dict(fit=fit, response=serializable(value, context, profile_baseline))
        all_rows[label + "__fitted_source_spectrum"] = source_template
        local_template = spectrum((required[0], required[1], "base"), fine_source)
        local_response = serializable(evaluate(context, local_template), context, base)
        full_source_control = {}
        for name, value in CASES.items():
            if value[2] != "base":
                continue
            bracket_only = evaluate(context, spectrum(value, refined))
            full_source_control[name] = dict(delta_Q=evaluated[name]["Q"] - bracket_only["Q"],
                target_total_change_ppm=(evaluated[name]["total"][context["target"]] - bracket_only["total"][context["target"]]).tolist())
        selection_results[label] = dict(target_support=context["target_support"],
            phase_rows=[dict(row, supported=bool(keep)) for row, keep in zip(context["rows"], context["supported"])],
            cases={name: serializable(value, context, base) for name, value in evaluated.items()},
            all_data_source_fit_fixed_RV_amplitude=source_fit,
            global_RV_amplitude_profiled=profiled,
            local_linear_source_solution=dict(required_inner_upper_source_multipliers=required.tolist(),
                positive_multipliers=bool(np.all(required >= 0)), derivative_BC_ppm_per_source_multiplier=derivative.tolist(),
                finite_response_at_linearized_multipliers=local_response,
                scope="First-order solution for two selected residual means only; not a fitted or self-consistent physical state."),
            full_phase_source_refinement=full_source_control,
            target_refinement=refinement)
        checks[label] = float(max(np.max(np.abs(entry["target_total_difference_ppm"])) for entry in refinement.values()))
    raw = args.output / "spectra_and_rows.npz"
    np.savez_compressed(raw, phase=PHASE, velocity=velocity, **spectra, **all_rows)
    result = dict(scope=__doc__.strip(), source_and_input_sha256=pins,
        unchanged_source_inputs=all(sha256(path) == value for path, value in pins.items()),
        baseline_fine_reproduction_max_difference_ppm=baseline_reproduction,
        target_refinement_maximum_difference_ppm=checks,
        cases={name: dict(inner_source_multiplier=value[0], upper_source_multiplier=value[1], upper_opacity_multiplier=TAU[value[2]]) for name, value in CASES.items()},
        source_partition="Inner means radii<1.3 Rp (cells0:4); upper means cells4:8. Source scaling changes line eta at fixed alpha; opacity scaling changes alpha and eta together at fixed source.",
        selections=selection_results, local_population_interpretation=local_physics(dict(np.load(INPUT / "fixed_inputs.npz")), json.loads(recipe_path.read_text())),
        raw_artifact=dict(path=str(raw), sha256=sha256(raw)),
        limitations=["The first-order coefficient experiments hold continuum coefficients, kinetic temperatures, density, velocity, and boundary intensity fixed. They do not satisfy a new coupled population, thermal or wind balance.",
            "Source-only response differences use fine rays at all seven phase nodes. Opacity response differences use coarse rays except the two fine nodes bracketing every target exposure; their full-data score is not a fully refined new physical fit. The atmosphere itself retains eight physical cells; observer ray refinement does not establish atmospheric radial convergence.",
            "The conditional GP mean is recomputed using all retained exposures with unchanged covariance. Both direct planet and GP-induced changes are reported; masks, support and global physical-template RV/amplitude are unchanged.",
            "Separate global-RV/amplitude sensitivities refit exactly two parameters per fixed template using all data and compare against the likewise refitted baseline. The two-source all-data fit instead keeps the original RV/amplitude fixed. Neither changes GP covariance or refits atmospheric equilibrium.",
            "The two combination cases are controlled source/opacity directions: increased inner/upper source contrast, and increased upper lower-level absorption with approximately fixed upper emissivity. Neither establishes a realizable mechanism.",
            "Shared lower sublevels and the imposed LTE inner cells prevent treating independent branch source/opacity scalings as an unconstrained physical population solution.",
            "The previous coupled H-alpha SE/thermal state is shown as an existing result, not an additional new mechanism. Free-model references reuse their archived fitted nuisance parameters and retain additional shape freedom.",
            "Q changes and target residual changes are conditional diagnostics on the same data, not model evidence or detection significances."])
    args.review.mkdir(parents=True, exist_ok=True)
    (args.review / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.review / "source_responses.png")
    print(json.dumps(dict(stage="complete", refinement=checks,
        target={name: entry["target_change_ppm"]["total"] for name, entry in selection_results["primary"]["cases"].items()})), flush=True)
    if not result["unchanged_source_inputs"] or baseline_reproduction > .03 or max(checks.values()) > 20.:
        raise RuntimeError("A source-response control failed; inspect the receipt.")


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = result["selections"]["primary"]["cases"]
    names = ["upper_source_half", "upper_opacity_thinner", "upper_opacity_thicker", "inner_source_1p5",
             "increased_source_contrast", "upper_lower_level_accumulation", "previous_coupled_Halpha"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    for i, (band, color) in enumerate(zip(("blue", "core", "red"), ("C0", "C1", "C3"))):
        ax.bar(x + (i - 1) * .25, [selected[name]["target_change_ppm"]["total"][i] for name in names], .25, label=band, color=color)
    ax.axhline(0, color="0.3", lw=.7)
    ax.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    ax.set_ylabel("Change in planet + reconditioned GP (ppm)")
    ax.set_title("Same complete exposures: phase 0.37615–0.39322\nRequired direction: blue up, core down; global RV/amplitude fixed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("/tmp/escape-halpha-phase-localization"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/escape-halpha-cause-sources"))
    parser.add_argument("--review", type=Path, default=Path("mceg/evidence/halpha_causes_20261127/sources"))
    run(parser.parse_args())
