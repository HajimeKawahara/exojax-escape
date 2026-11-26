"""Frozen-state H-alpha illumination: plane slabs versus spherical shells.

This diagnostic changes formal-transfer geometry, not populations or the
observed spectrum. Extinction includes stimulated emission, so the reported
line absorption is a net absorption rate, not the gross upward Einstein rate.
Boundary intensities are zero. Emission and intensity are per wavenumber.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from scipy.special import voigt_profile

from escape.radiation import solve_slab_transfer
from escape.spherical_radiation import solve_spherical_transfer, spherical_shell_geometry

jax.config.update("jax_enable_x64", True)

C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
RADIUS_CM = 1.9 * 7.1492e9


def load_state(input_dir=Path("/tmp/escape-e8-mdot-sphere/fine"),
               recipe=Path("/tmp/escape-corrected-observer-full-f/recipe.json")):
    """Read the saved PR12/13 eight-cell coefficients without re-solving SE."""
    input_dir, recipe = Path(input_dir), Path(recipe)
    paths = [input_dir / "fixed_inputs.npz", input_dir / "branch_arrays.npz", recipe]
    state = {}
    for path in paths[:2]:
        with np.load(path) as saved:
            state.update({key: saved[key] for key in saved.files})
    edges_rp = np.asarray(json.loads(recipe.read_text())["edges_rp"])
    state.update(edges_cm=edges_rp * RADIUS_CM,
                 radius_rp=(edges_rp[1:] + edges_rp[:-1]) / 2,
                 input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths})
    return state


def line_coefficients(state, wavenumber_cm=None):
    """Expand the three saved Voigt branches at zero macroscopic velocity."""
    wn = state["wavenumber_cm"] if wavenumber_cm is None else np.asarray(wavenumber_cm)
    profile = voigt_profile(wn[None, None, :] - state["branch_centers"][None, :, None],
                            state["branch_sigma"][:, :, None],
                            state["branch_gamma"][None, :, None])
    continuum = {}
    for source, name, factor in [("continuum_extinction", "continuum_chi", 1.),
                                  ("continuum_emission_hz", "continuum_eta", C_CGS)]:
        continuum[name] = factor * np.array([
            np.interp(wn, state["wavenumber_cm"], row) for row in state[source]])
    return dict(wavenumber_cm=wn, line_chi=state["branch_alpha"][:, :, None] * profile,
                line_eta=state["branch_eta"][:, :, None] * profile, **continuum)


def formal_fields(edges_cm, chi, eta, geometry="sphere", angle_order=32):
    """Return J and H with absorbing inner core and zero incoming radiation."""
    zero = np.zeros(np.shape(chi)[-1])
    if geometry == "sphere":
        result = solve_spherical_transfer(spherical_shell_geometry(
            edges_cm, angle_order=angle_order), chi, eta, zero, zero)
    elif geometry == "slab":
        result = solve_slab_transfer(edges_cm, chi, eta, zero, zero,
                                     angle_order=angle_order)
    else:
        raise ValueError("geometry must be 'sphere' or 'slab'")
    if not bool(result.converged):
        raise RuntimeError(f"Formal {geometry} solve failed: {result.balance_error}")
    return result


def photon_integral(wavenumber_cm, spectral_quantity, spectral_weights=None):
    """Integrate a per-wavenumber quantity against d(wavenumber)/(h nu)."""
    wn = np.asarray(wavenumber_cm)
    photons = spectral_quantity / (H_CGS * C_CGS * wn)
    if spectral_weights is not None:
        return np.sum(photons * spectral_weights, axis=-1)
    return np.trapezoid(photons, wn, axis=-1)


def geometry_measure(edges_cm, geometry):
    edges = np.asarray(edges_cm)
    if geometry == "sphere":
        return np.diff(edges**3) / 3, edges**2
    if geometry == "slab":
        return np.diff(edges), np.ones_like(edges)
    raise ValueError("geometry must be 'sphere' or 'slab'")


def photon_fates(edges_cm, wavenumber_cm, line_chi, line_eta, continuum_chi,
                  geometry="sphere", angle_order=32, spectral_weights=None):
    """Resolve one formal flight by emitting cell/branch and absorbing cell/branch.

    The normalized fates sum to one for positive extinction. With net line
    extinction these are stimulated-corrected extinction fractions; they are
    not repeated-scattering destruction probabilities. Pass true absorption
    coefficients instead to build a photon transition matrix for another model.
    No continuum emission is included in these line-source Green functions.
    Optional spectral_weights replace trapezoidal integration and have units
    of wavenumber, e.g. a frequency quadrature's weights divided by c.
    """
    cells, branches, frequencies = np.shape(line_eta)
    volume, area = geometry_measure(edges_cm, geometry)
    chi = np.sum(line_chi, axis=1) + continuum_chi
    if np.any(line_chi < 0) or np.any(continuum_chi < 0):
        raise ValueError("Photon fates require nonnegative extinction.")
    def integrate(quantity):
        return photon_integral(wavenumber_cm, quantity, spectral_weights)
    emitted = volume[:, None] * integrate(line_eta)
    if np.any(emitted <= 0):
        raise ValueError("Each source channel must emit a positive photon rate.")
    mean = np.empty((cells, branches, cells, frequencies))
    flux = np.empty((cells, branches, cells + 1, frequencies))
    line = np.empty((cells, branches, cells, branches))
    continuum = np.empty((cells, branches, cells))
    core, outer = np.empty((cells, branches)), np.empty((cells, branches))
    solver_error = 0.
    for sender in range(cells):
        for branch in range(branches):
            eta = np.zeros((cells, frequencies))
            eta[sender] = line_eta[sender, branch]
            solved = formal_fields(edges_cm, chi, eta, geometry, angle_order)
            intensity = mean[sender, branch] = np.asarray(solved.mean_intensity)
            moment = flux[sender, branch] = np.asarray(solved.flux_moment_edges)
            supplied = emitted[sender, branch]
            line[sender, branch] = volume[:, None] * integrate(
                line_chi * intensity[:, None, :]) / supplied
            continuum[sender, branch] = volume * integrate(
                continuum_chi * intensity) / supplied
            core[sender, branch] = -area[0] * integrate(moment[0]) / supplied
            outer[sender, branch] = area[-1] * integrate(moment[-1]) / supplied
            solver_error = max(solver_error, float(solved.balance_error))
    accounted = line.sum(axis=(2, 3)) + continuum.sum(axis=2) + core + outer
    return dict(line=line, continuum=continuum, core=core, outer=outer,
                emitted=emitted, mean_intensity=mean, flux_moment_edges=flux,
                accounting_error=float(np.max(np.abs(accounted - 1))),
                solver_balance_error=solver_error)


def field_rates(state, coeff, geometry, angle_order=32):
    chi = coeff["line_chi"].sum(axis=1) + coeff["continuum_chi"]
    line_eta = coeff["line_eta"].sum(axis=1)
    total = formal_fields(state["edges_cm"], chi, line_eta + coeff["continuum_eta"],
                           geometry, angle_order)
    inner = formal_fields(state["edges_cm"], chi,
                           line_eta * (state["radius_rp"] < 1.3)[:, None],
                           geometry, angle_order)
    def rate(mean):
        return 4 * np.pi * photon_integral(coeff["wavenumber_cm"],
            coeff["line_chi"] * np.asarray(mean)[:, None, :])
    return dict(total=rate(total.mean_intensity), inner=rate(inner.mean_intensity),
                mean_intensity=np.asarray(total.mean_intensity))


def combined_line_budget(state, coeff, geometry, angle_order=32):
    """Photon fractions for all fixed line sources, with continuum emission off."""
    volume, area = geometry_measure(state["edges_cm"], geometry)
    wn = coeff["wavenumber_cm"]
    eta, line_chi = coeff["line_eta"].sum(axis=1), coeff["line_chi"].sum(axis=1)
    solved = formal_fields(state["edges_cm"], line_chi + coeff["continuum_chi"],
                            eta, geometry, angle_order)
    mean, flux = np.asarray(solved.mean_intensity), np.asarray(solved.flux_moment_edges)
    supplied = np.sum(volume * photon_integral(wn, eta))
    return dict(line_reabsorption=float(np.sum(volume * photon_integral(wn, line_chi * mean)) / supplied),
                continuum_absorption=float(np.sum(volume * photon_integral(wn, coeff["continuum_chi"] * mean)) / supplied),
                core_loss=float(-area[0] * photon_integral(wn, flux[0]) / supplied),
                outer_escape=float(area[-1] * photon_integral(wn, flux[-1]) / supplied))


def run(input_dir, recipe, output_dir, review_path, angle_order=32):
    state = load_state(input_dir, recipe)
    coeff = line_coefficients(state)
    wn, edges = coeff["wavenumber_cm"], state["edges_cm"]
    chi = coeff["line_chi"].sum(axis=1) + coeff["continuum_chi"]
    emission = 4 * np.pi * photon_integral(wn, coeff["line_eta"]).sum(axis=1)
    arrays = dict(wavenumber_cm=wn, radius_rp=state["radius_rp"], edges_cm=edges)
    summary = {}
    for geometry in ("slab", "sphere"):
        print(f"Resolving {geometry} line-source responses", flush=True)
        fates = photon_fates(edges, wn, coeff["line_chi"], coeff["line_eta"],
                            coeff["continuum_chi"], geometry, angle_order)
        continuum = formal_fields(edges, chi, coeff["continuum_eta"], geometry, angle_order)
        rates = field_rates(state, coeff, geometry, angle_order)
        source_mean = fates["mean_intensity"].sum(axis=1)
        source_rates = 4 * np.pi * photon_integral(
            wn, coeff["line_chi"][None] * source_mean[:, :, None, :]).sum(axis=-1)
        continuum_rate = 4 * np.pi * photon_integral(
            wn, coeff["line_chi"] * np.asarray(continuum.mean_intensity)[:, None]).sum(axis=1)
        absorption = rates["total"].sum(axis=1)
        recovered = source_mean.sum(axis=0) + np.asarray(continuum.mean_intensity)
        superposition_error = float(np.max(np.abs(recovered - rates["mean_intensity"]))
                                    / np.max(rates["mean_intensity"]))
        weights = fates["emitted"] / fates["emitted"].sum()
        budget = {"line_reabsorption": float(np.sum(weights * fates["line"].sum(axis=(2, 3)))),
                  "continuum_absorption": float(np.sum(weights * fates["continuum"].sum(axis=2))),
                  "core_loss": float(np.sum(weights * fates["core"])),
                  "outer_escape": float(np.sum(weights * fates["outer"]))}
        summary[geometry] = dict(
            net_line_absorption_cm3_s=absorption.tolist(),
            spontaneous_emission_cm3_s=emission.tolist(),
            net_radiative_excitation_cm3_s=(absorption - emission).tolist(),
            line_source_absorption_by_sender_cm3_s=source_rates.tolist(),
            continuum_source_absorption_cm3_s=continuum_rate.tolist(),
            own_cell_line_absorption_cm3_s=np.diag(source_rates).tolist(),
            inner_four_cell_line_absorption_cm3_s=source_rates[:4].sum(axis=0).tolist(),
            inner_two_LTE_cell_line_absorption_cm3_s=source_rates[:2].sum(axis=0).tolist(),
            inner_NLTE_cells_2_3_line_absorption_cm3_s=source_rates[2:4].sum(axis=0).tolist(),
            outer_cells_5_7_line_absorption_cm3_s=source_rates[5:].sum(axis=0).tolist(),
            line_source_first_flight_photon_fractions=budget,
            photon_accounting_error=fates["accounting_error"],
            solver_balance_error=fates["solver_balance_error"],
            source_superposition_error=superposition_error)
        arrays.update({f"{geometry}_{key}": value for key, value in fates.items()})
        arrays[f"{geometry}_total_mean_intensity"] = rates["mean_intensity"]
        arrays[f"{geometry}_absorption_by_sender_cm3_s"] = source_rates
        print(f"{geometry}: absorption at r=1.5125 Rp = {absorption[4]:.7g}", flush=True)

    refinements = {}
    midpoint = np.sort(np.r_[wn, (wn[:-1] + wn[1:]) / 2])
    fine_coeff = line_coefficients(state, midpoint)
    center, width = (wn[0] + wn[-1]) / 2, wn[-1] - wn[0]
    wide_coeff = line_coefficients(state, np.linspace(center - width, center + width, 2 * wn.size - 1))
    wing_control = dict(
        omitted_normalized_profile_area=(1 - np.trapezoid(
            coeff["line_chi"] / state["branch_alpha"][:, :, None], wn, axis=-1)).tolist(),
        doubled_interval_cm=wide_coeff["wavenumber_cm"][[0, -1]].tolist(),
        continuum_extension="Constant edge values outside the saved interval; only a bandwidth sensitivity check.")
    for geometry in ("slab", "sphere"):
        coarse = field_rates(state, coeff, geometry, angle_order)
        angle = field_rates(state, coeff, geometry, 2 * angle_order)
        spectral = field_rates(state, fine_coeff, geometry, angle_order)
        joint = field_rates(state, fine_coeff, geometry, 2 * angle_order)
        wide = field_rates(state, wide_coeff, geometry, angle_order)
        wide_budget = combined_line_budget(state, wide_coeff, geometry, angle_order)
        base_budget = summary[geometry]["line_source_first_flight_photon_fractions"]
        wing_control[geometry] = {
            "absorption_max_relative_change": float(np.max(np.abs(wide["total"] / coarse["total"] - 1))),
            "inner_absorption_max_relative_change": float(np.max(np.abs(wide["inner"] / coarse["inner"] - 1))),
            "doubled_band_photon_fractions": wide_budget,
            "photon_fraction_absolute_changes": {key: wide_budget[key] - base_budget[key] for key in wide_budget}}
        refinements[geometry] = {}
        for name, other in (("angle", angle), ("spectral", spectral), ("joint", joint)):
            refinements[geometry][name] = {
                key + "_max_relative_change": float(np.max(np.abs(other[key] / coarse[key] - 1)))
                for key in ("total", "inner")}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / "arrays.npz"
    np.savez_compressed(array_path, **arrays)
    sphere, slab = summary["sphere"], summary["slab"]
    ratio = {key: (np.asarray(sphere[key]) / np.asarray(slab[key])).tolist() for key in (
        "net_line_absorption_cm3_s", "inner_four_cell_line_absorption_cm3_s",
        "inner_two_LTE_cell_line_absorption_cm3_s")}
    review = dict(
        status="Executed frozen-state geometry diagnostic; atmospheric equilibrium and observed fit not re-solved.",
        scope="Static three-branch H-alpha Voigt transfer with fixed PR12/13 populations, zero boundaries and saved continuum. Sphere J is a shell-volume average; slab J is a cell-thickness average.",
        interpretation="Absorption uses net extinction (stimulated emission subtracted). Source fates count one formal flight; reabsorbed photons are not iteratively re-emitted here.",
        input_sha256=state["input_sha256"],
        source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                           Path(__file__).resolve(),
                           Path(__file__).resolve().parents[1] / "src/escape/radiation.py",
                           Path(__file__).resolve().parents[1] / "src/escape/spherical_radiation.py")},
        raw_arrays=dict(path=str(array_path), sha256=hashlib.sha256(array_path.read_bytes()).hexdigest()),
        radius_rp=state["radius_rp"].tolist(), angle_order=angle_order,
        wavenumber_samples=wn.size, wavenumber_interval_cm=wn[[0, -1]].tolist(),
        diagnostic=summary, sphere_over_slab=ratio, refinement=refinements, wing_control=wing_control,
        thresholds=dict(photon_accounting_error=1e-9, source_superposition_error=1e-9,
                        refinement_max_relative_change=.01),
        limitations=["No level-population or thermal re-equilibration and no prediction of the observed blue peak or trough.",
                     "No macroscopic velocity, partial redistribution, or explicit H-alpha/Ly-beta cycling.",
                     "The saved finite spectral interval is retained for source decomposition; a doubled-band sensitivity uses constant-edge continuum extrapolation.",
                     "The existing frozen continuum extinction is treated as absorption in this formal-flight budget."])
    review["checks_passed"] = bool(all(
        item["photon_accounting_error"] < 1e-9 and item["source_superposition_error"] < 1e-9
        for item in summary.values()) and all(value < .01 for geom in refinements.values()
            for setting in geom.values() for value in setting.values()))
    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(dict(checks_passed=review["checks_passed"], sphere_over_slab=ratio,
                          refinement=refinements), indent=2), flush=True)
    return review


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fine"))
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/escape-halpha-geometry-effect"))
    parser.add_argument("--review", type=Path,
                        default=Path("mceg/evidence/halpha_geometry_effect_20261126/review.json"))
    parser.add_argument("--angle-order", type=int, default=32)
    options = parser.parse_args()
    run(options.input_dir, options.recipe, options.output_dir, options.review, options.angle_order)
