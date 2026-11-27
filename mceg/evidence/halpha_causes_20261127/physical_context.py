"""Read saved-state time scales; do not interpret them as relaxation modes.

Run from the repository with PYTHONPATH=src:examples and the archived ExoJAX
dependency available. This only reads the trusted local diagnostic pickle.
"""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from halpha_geometry_effect import load_state, line_coefficients


def run():
    physical_path = Path("/tmp/escape-solar-molecular-continuation/"
                         "current8_e_mdot_refined_trust/complete_physical.pkl")
    physical = pickle.loads(physical_path.read_bytes())
    state = load_state()
    for key in ("temperature_k", "bound_cm3", "radial_velocity_kms"):
        np.testing.assert_array_equal(state[key], physical[key])
    ne = np.asarray(physical["electron_cm3"])
    rates = physical["radiation"]["rates"]
    alpha = np.asarray(rates.radiative_recombination_cm3_s)
    width = np.diff(state["edges_cm"])
    radius = (state["edges_cm"][:-1] + state["edges_cm"][1:]) / 2
    # Saved base H return to 18 bound states, including ground. The historical
    # rate object omits receiving packets added later by the complete equations.
    recombination = 1 / (ne * alpha.sum(axis=1))
    photoionization = np.asarray(rates.photoionization_s)
    flow = width / (np.asarray(physical["radial_velocity_kms"]) * 1e5)
    # Thermal energy divided by the residual's channel scale, not a linear
    # thermal relaxation time and not a treatment of ionization energy.
    turnover = 1.5 * np.asarray(physical["pressure_dyn_cm2"]) / np.asarray(
        physical["thermal_scale"])
    coeff = line_coefficients(state)
    tau = coeff["line_chi"].sum(axis=1).max(axis=1) * width
    delta_phase = 0.3932225607463806 - 0.3761495834419293
    seconds = delta_phase * 1.48 * 86400
    angle = 2 * np.pi * delta_phase
    # F/Fbar=1+a*cos(4*pi*phase+psi), a deliberately specified proxy.
    # Its endpoint-change maximum is independent of unknown phase psi.
    uv_amplitude = .05
    uv_bound = 2 * uv_amplitude * np.sin(angle)
    psi = np.linspace(0, 2*np.pi, 100001)
    direct = np.max(abs(uv_amplitude * (np.cos(psi + 2*angle) - np.cos(psi))))
    np.testing.assert_allclose(direct, uv_bound, rtol=1e-9)
    result = dict(
        status="Executed saved-state scale estimates; no time-dependent atmosphere or fit.",
        input_sha256={**state["input_sha256"], str(physical_path): hashlib.sha256(
            physical_path.read_bytes()).hexdigest()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interval_seconds=seconds, viewing_angle_degrees=360*delta_phase,
        radius_rp=state["radius_rp"].tolist(),
        temperature_k=np.asarray(physical["temperature_k"]).tolist(),
        electron_cm3=ne.tolist(),
        base_gross_recombination_seconds=recombination.tolist(),
        base_n2_photoionization_seconds=(1/photoionization[:, 1:3]).tolist(),
        peak_rest_frame_radial_line_optical_depth=tau.tolist(),
        branch_source_per_wavenumber=(state["branch_eta"]/state["branch_alpha"]).tolist(),
        radial_cell_crossing_seconds=flow.tolist(),
        thermal_turnover_seconds=turnover.tolist(),
        horizontal_radius_crossing_at_20kms_seconds=(radius/2e6).tolist(),
        orbital_illumination_proxy=dict(
            source="https://arxiv.org/html/2004.14812v1#S4.SS2",
            assumption="Illustrative sinusoid with a 10-percent peak-to-peak UV change, twice per orbit; not a reconstructed irradiation history.",
            fractional_endpoint_change_max=float(uv_bound),
            dense_phase_check_absolute_error=float(abs(direct-uv_bound))),
        checks=dict(saved_state_identity=True, positive_finite_scales=bool(all(
            np.isfinite(x).all() and np.all(x > 0) for x in
            (recombination, flow, turnover, photoionization[:, 1:3])))),
        limitations=[
            "The historical radiation.rates object contains base H coefficients, not the receiving PI/RR packets added by the complete solar equations. Reported base times are scale estimates, not complete rates. This known readout limitation is documented in ../halpha_equilibrium_20261126/prior_conversion_receiving_audit.json.",
            "Recombination is a gross one-way time; photon recycling and reionization can substantially lengthen effective return to the ground state.",
            "Photoionization is already present in this model. Its short n=2 one-way time does not determine n=2 or n=3 coupled response times.",
            "Thermal turnover is 3P/(2*thermal_scale); this scale is H heating+cooling plus absolute channel ledger terms. It is not a total cooling rate or perturbation eigenvalue and excludes latent chemical energy.",
            "A 36-minute data window does not establish a 36-minute atmospheric event or intrinsic phase evolution.",
            "Neither the sinusoidal illumination proxy nor the 20 km/s horizontal speed is fitted here. Nonlinear response and directional stellar line profiles remain uncalculated."
        ])
    assert all(result["checks"].values())
    Path(__file__).with_name("physical_context.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("interval_seconds", "viewing_angle_degrees",
          "base_gross_recombination_seconds", "thermal_turnover_seconds", "checks")}, indent=2))


if __name__ == "__main__":
    run()
