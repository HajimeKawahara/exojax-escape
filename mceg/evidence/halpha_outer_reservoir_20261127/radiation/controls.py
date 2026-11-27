"""Independent arithmetic and stellar-kernel refinement for quantify.py."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import voigt_profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = json.loads((HERE / "review.json").read_text())
    branch_path = Path("/tmp/escape-e8-mdot-sphere/fine/branch_arrays.npz")
    stellar_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    with np.load(branch_path) as branch:
        centers = branch["branch_centers"] * 2.99792458e10
    with np.load(stellar_path) as star:
        frequency, flux = star["frequency_hz"], star["surface_flux_nu"]
    velocity = np.arange(-500., 500.001, .125)
    nodes, weights = np.polynomial.legendre.leggauss(320)
    weights *= np.sqrt(1 - nodes**2)
    weights /= weights.sum()
    phi = voigt_profile(velocity, 8.2, .02)
    refined = {}
    for vmax in (0., 77.4, 110., 160.):
        values = []
        for center in centers:
            query = center * (1 - (velocity[:, None] - vmax * nodes[None]) / 299792.458)
            broadened = np.interp(query, frequency, flux) @ weights
            values.append(float(np.trapezoid(broadened * phi, velocity) / np.trapezoid(phi, velocity)))
        refined[str(vmax)] = values
    ratios = {key: (np.array(value) / refined["0.0"]).tolist() for key, value in refined.items()}
    maximum = max(float(np.max(abs(np.array(ratios[key]) /
        result["illustrative_stellar_Halpha_velocity_average"]["ratio_to_unbroadened"][key] - 1))) for key in ratios)
    sigma = np.asarray(result["hydrogen_cross_section_cm2"])
    columns = np.asarray(result["positive_level_columns_cm2"]["1.725"])
    tau = np.asarray(result["continuum_tau"]["1.725"]["H2_gross"])
    expected = columns[1:3] @ sigma[1:3]
    positive = expected > 0
    integration_error = float(np.max(abs(expected[positive] / tau[positive] - 1)))
    gamma = np.asarray(result["unattenuated_boundary_photoionization_s"])
    bands = np.sum([row["rate_s"] for row in result["boundary_photoionization_energy_bands"]], axis=0)
    photo_error = float(np.max(abs(bands / gamma - 1)))
    checks = dict(source_inputs_unchanged=all(digest(path) == value for path, value in result["source_and_input_sha256"].items()),
        zero_ground_H_opacity_below_13p6_ev=bool(np.all(sigma[0, :8] == 0)),
        zero_n2_opacity_below_3p4_ev=bool(np.all(sigma[1:3, :3] == 0)),
        gross_columns_reproduce_tau=integration_error < 1e-12,
        energy_bands_sum_to_photoionization=photo_error < 1e-12,
        stellar_kernel_refinement_below_0p2_percent=maximum < .002)
    # A conditional angular bound: a foreground absorber covering a bright
    # sphere is within asin(R_emit/r) of the observer direction. It is not
    # an exact extended-star or sector occultation calculation.
    phase = np.array([.3761495834419293, .3932225607463806])
    phase_angle = np.degrees(np.arccos(-np.cos(2 * np.pi * phase)))
    shadow_bound = dict(target_phase=phase.tolist(), observer_star_angle_deg=phase_angle.tolist(),
        assumed_bright_emitter_radius_rp=1.3,
        absorber_radius_rp=[1.9375, 2.2464240461805627],
        maximum_star_zenith_deg_for_foreground_covering=[float(phase_angle.max() + np.degrees(np.arcsin(1.3 / r))) for r in (1.9375, 2.2464240461805627)],
        scope="Spherical bright-emitter bound only. Those foreground locations are on the star-facing hemisphere, so a planetary anti-stellar shadow is not automatically the required foreground screen. Extended/anisotropic emitters and other shields are not excluded.")
    lifetime = dict(cell_index=5, level_names=["2s", "2p"],
        inverse_saved_base_photoionization_s=(1 / np.asarray(result["saved_base_photoionization_s"])[5, 1:3]).tolist(),
        scope="Individual excited-atom photoionization times at the frozen base field, not the total H ionization time or the relaxation time of the coupled excitation/ionization system. Other excitation losses and replenishment coexist; receiving packets are omitted from these base rates.")
    output = dict(checks=checks, all_passed=all(checks.values()), column_tau_relative_error=integration_error,
        photo_band_relative_error=photo_error, stellar_refinement_max_relative_error=maximum,
        stellar_refined_ratio_to_unbroadened=ratios, shadow_geometry_condition=shadow_bound,
        individual_excited_atom_time=lifetime,
        inputs={str(path): digest(path) for path in (Path(__file__), HERE / "quantify.py", HERE / "review.json", branch_path, stellar_path)})
    (HERE / "controls.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))
    assert all(checks.values())


if __name__ == "__main__":
    main()
