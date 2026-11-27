"""Frozen opacity, boundary-photoionization and illumination scale estimates.

No new radiation field, level population, thermal state or observed spectrum
is solved. Positive absorption and signed net extinction are kept distinct.
"""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import voigt_profile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples"))
from halpha_equilibrium import build_model
from escape.hydrogen_atom import atom_cross_sections
from escape.hydrogen_continuum import continuum_coefficients

jax.config.update("jax_enable_x64", True)
H, C, EV, RP = 6.62607015e-27, 2.99792458e10, 1.602176634e-12, 1.9 * 7.1492e9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    recipe_path = Path("/tmp/escape-corrected-observer-full-f/recipe.json")
    stellar_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    physical_path = Path("/tmp/escape-solar-molecular-continuation/current8_e_mdot_refined_trust/complete_physical.pkl")
    nist_path = Path("/tmp/escape-five-cell-root-evidence/nist_row.json")
    fixed_path = Path("/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz")
    free_path = ROOT / "mceg/evidence/halpha_causes_20261127/sources/free_shape.json"
    inputs = [Path(__file__), recipe_path, stellar_path, physical_path, nist_path, fixed_path, free_path,
        ROOT / "examples/halpha_equilibrium.py", *sorted((ROOT / "src/escape").glob("*.py"))]
    pins = {str(path): digest(path) for path in inputs}
    with physical_path.open("rb") as stream:
        state = pickle.load(stream)
    fixed = dict(np.load(fixed_path))
    for key in ("temperature_k", "bound_cm3", "radial_velocity_kms"):
        np.testing.assert_array_equal(state[key], fixed[key])
    model, boundary, recipe = build_model(recipe_path, stellar_path, nist_path, "none")
    print("Constructed current model; evaluating fixed coefficients only.", flush=True)
    edges = np.asarray(recipe["edges_rp"])
    radius = (edges[1:] + edges[:-1]) / 2
    above = {str(cut): np.maximum(0., edges[1:] - np.maximum(edges[:-1], cut)) * RP for cut in (1., 1.3, 1.725, 2.15)}
    energies = np.array([1.52, 1.89, 2.5, 3.41, 4., 5., 8., 10.2, 13.7, 20.])
    frequency = jnp.asarray(energies * EV / H)
    atom = model.hydrogen.atom
    sigma = np.asarray(atom_cross_sections(atom, frequency))
    level = np.asarray(state["bound_cm3"])
    components = {name: level[:, take] @ sigma[take] for name, take in
        (("H1_gross", [0]), ("H2_gross", [1, 2]), ("H3_gross", [3, 4, 5]), ("Hhigh_gross", list(range(6, 18))))}
    hnet, _ = jax.vmap(lambda t, b, e, p: continuum_coefficients(atom, t, b, e, p, frequency,
        include_free_free=False))(state["temperature_k"], level, state["electron_cm3"], state["proton_cm3"])
    hproduction, _, _ = model.hydrogen._continuum(state["temperature_k"], level,
        state["proton_cm3"], state["electron_cm3"], frequency, jnp.asarray(sigma))
    total, _ = model.continuum_coefficients_at(state["temperature_k"], state, frequency)
    metal, _ = model.metal.continuum_coefficients_at(state["temperature_k"], state["metal_ion_cm3"],
        state["electron_cm3"], frequency, level_population_cm3=state["metal_levels_cm3"])
    components.update(H_bound_free_signed=np.asarray(hnet), H_retained_gain_suppressed=np.asarray(hproduction),
        metal_gross=np.asarray(metal), total_retained_absorption=np.asarray(total))
    components["other_retained"] = components["total_retained_absorption"] - components["H_retained_gain_suppressed"] - components["metal_gross"]
    opacities = {cut: {name: (width @ value).tolist() for name, value in components.items()} for cut, width in above.items()}
    columns = {cut: (width @ level).tolist() for cut, width in above.items()}

    # Gross surface-boundary photoionization: no atmospheric attenuation,
    # diffuse intensity, OTS packets, or change of saved rates is implied.
    grid_frequency, weight = np.asarray(model.grid.frequency_hz), np.asarray(model.grid.weight_hz)
    grid_energy = H * grid_frequency / EV
    cross = np.asarray(model.hydrogen._sigma)
    photon = 4 * np.pi * np.asarray(boundary["incident_mean_intensity"]) * weight / (H * grid_frequency)
    bands = [0., 1.51, 1.89, 3.4, 6., 10.2, 13.6, float("inf")]
    photo = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        keep = (grid_energy >= lo) & (grid_energy < hi)
        photo.append(dict(energy_ev=[lo, None if not np.isfinite(hi) else hi],
            rate_s=(cross[:, keep] @ photon[keep]).tolist()))
    gamma = cross @ photon
    rates = state["radiation"]["rates"]
    down = np.asarray(rates.radiative_bound_bound_s).sum(axis=-1)
    operators = state["radiation"]["line_operators"]

    free = json.loads(free_path.read_text())["free"]["original"]
    n2_reference = free["n2_column_above_radius_cm2"]["1.725"]
    n2_actual = np.sum(np.asarray(columns["1.725"])[1:3])
    factor = n2_reference / n2_actual
    fraction2 = np.asarray(columns["1.725"])[1:3] / n2_actual
    effective = fraction2 @ sigma[1:3]
    required = [float(np.log(factor) / value) if value > 0 else None for value in effective]
    target_tau = n2_reference * effective

    # Uniform circular-source dilution, a geometric scale, not full transfer.
    star_ratio = recipe["stellar_boundary"]["incident_flux_ratio"]
    illumination = dict(star_radius_over_distance=float(np.sqrt(star_ratio)),
        star_angular_radius_deg=float(np.degrees(np.arcsin(np.sqrt(star_ratio)))),
        exact_uniform_disk_dilution=float((1 - np.sqrt(1 - star_ratio)) / 2),
        normal_beam_equivalent_dilution=star_ratio / 4,
        inner_bright_radius_rp=1.3,
        inner_bright_uniform_sphere_dilution=((1 - np.sqrt(1 - (1.3 / radius[4:])**2)) / 2).tolist(),
        radius_rp=radius[4:].tolist(),
        note="Solid angles alone omit atmospheric extinction, local line emission, angular asymmetry and scattering. The exact disk correction at fixed normal irradiance is small; finite-disk shadows and frequency averaging are separate.")

    with np.load(stellar_path) as saved:
        star_nu, star_flux = saved["frequency_hz"], saved["surface_flux_nu"]
    center = np.asarray(atom.line_frequency_hz)[2:5]
    velocity = np.arange(-500., 500.001, .25)
    angular, angular_weight = np.polynomial.legendre.leggauss(160)
    angular_weight *= np.sqrt(1 - angular**2)
    angular_weight /= angular_weight.sum()
    # Illustration of unresolved uniform-disk velocity averaging. This is
    # not a fit of the planet-view stellar rotation or gravity darkening.
    broadening = {}
    for vmax in (0., 77.4, 110., 160.):
        values = []
        for nu in center:
            query = nu * (1 - (velocity[:, None] - vmax * angular[None]) / 299792.458)
            flux = np.interp(query, star_nu, star_flux) @ angular_weight
            phi = voigt_profile(velocity, 8.2, .02)
            values.append(float(np.trapezoid(flux * phi, velocity) / np.trapezoid(phi, velocity)))
        broadening[str(vmax)] = values
    relative = {key: (np.asarray(value) / broadening["0.0"]).tolist() for key, value in broadening.items()}
    out = dict(scope=__doc__.strip(), source_and_input_sha256=pins,
        unchanged_inputs=all(digest(path) == value for path, value in pins.items()),
        radius_rp=radius.tolist(), energy_ev=energies.tolist(), wavelength_angstrom=(H*C/(energies*EV)*1e8).tolist(),
        hydrogen_cross_section_cm2=sigma[:6].tolist(), positive_level_columns_cm2=columns, continuum_tau=opacities,
        unattenuated_boundary_photoionization_s=gamma.tolist(), boundary_photoionization_energy_bands=photo,
        saved_base_photoionization_s=np.asarray(rates.photoionization_s)[:, :6].tolist(),
        saved_radiative_out_s=down[:, :6].tolist(),
        direct_Gamma3_over_saved_radiative_out=(gamma[None, 3:6] / down[:, 3:6]).tolist(),
        saved_lya_escape_probability=np.asarray(operators.escape_probability)[:, 0].tolist(),
        saved_lya_destruction_probability=np.asarray(operators.destruction_probability)[:, 0].tolist(),
        saved_lya_pump_up_s=np.asarray(operators.pump_up_s)[:, 0].tolist(),
        saved_lya_net_decay_s=(np.asarray(operators.radiative_down_s)[:, 0] - np.asarray(operators.stimulated_down_s)[:, 0]).tolist(),
        pure_n2_screen=dict(outer_column_ratio_free_to_physical=float(factor),
            outer_n2_free_cm2=n2_reference, outer_n2_physical_cm2=n2_actual,
            free_column_balmer_tau_at_physical_2s2p_mix=target_tau.tolist(),
            necessary_n2_column_for_monochromatic_attenuation_by_factor=required,
            scope="A necessary monochromatic pure-absorption screen estimate only: suppressing a total loss rate by this factor also requires unchanged production and negligible competing losses. No population scaling is predicted."),
        illumination=illumination,
        illustrative_stellar_Halpha_velocity_average=dict(maximum_projected_velocity_kms=[0.,77.4,110.,160.],
            local_hydrogen_profile_sigma_kms=8.2, local_profile_lorentz_hwhm_kms=.02,
            branch_profile_weighted_flux=broadening, ratio_to_unbroadened=relative,
            scope="Uniform-disk velocity kernels at fixed supplied PHX local spectrum. No gravity-darkening, limb-angle or planetary viewing geometry reconstruction."),
        limitations=["All saved rates are the original base radiation rates; later receiving PI/RR packets are not included in those stored fields.",
            "Bound-free screens above 1.725 Rp are radial columns. Oblique paths, changed chemistry, arbitrary screen composition and metal line forests are not solved.",
            "Metal absorption uses the retained reduced network: mainly ground-ion continua plus its explicit selected excited continua, not complete excited metal opacity.",
            "Current stellar illumination uses exact unbroadened PHX line samples and a normal incident beam; replacing it with a deeper invented H-alpha line is not a measured correction.",
            "A thick H-alpha shell reemits and produces halo emission. Larger opacity is not equivalent to a darker unresolved core."])
    destination = Path(__file__).with_name("review.json")
    destination.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(unchanged_inputs=out["unchanged_inputs"], n2_factor=factor,
        tau_outer=opacities["1.725"], gamma=gamma[:6].tolist(), broadening=relative)), flush=True)


if __name__ == "__main__":
    main()
