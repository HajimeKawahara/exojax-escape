"""Frozen-state H-alpha transfer with radial Doppler shifts on finite chords.

The same local profile controls emission, extinction, and the receiving atom's
absorption integral. This is not a shift of an observer spectrum, an SE solve,
or a scattering/redistribution closure. Continuum coefficients are fixed across
this narrow frequency interval. Both diffuse boundaries are dark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import voigt_profile

C_CGS = 2.99792458e10
C_KMS = C_CGS / 1e5
H_CGS = 6.62607015e-27
RP_CM = 1.9 * 7.1492e9


def load_state(input_dir, recipe):
    """Load the saved physical coefficients, without recomputing populations."""
    with np.load(Path(input_dir) / "fixed_inputs.npz") as data:
        state = {key: data[key] for key in data.files}
    with np.load(Path(input_dir) / "branch_arrays.npz") as data:
        state.update({key: data[key] for key in data.files})
    edges = np.asarray(json.loads(Path(recipe).read_text())["edges_rp"])
    state["edges_cm"] = edges * RP_CM
    state["radius_rp"] = (edges[:-1] + edges[1:]) / 2
    return state


def _weights(tau):
    small = np.abs(tau) < 1e-3
    safe = np.where(small, 1., tau)
    first = -np.expm1(-safe) / safe
    second = (1 - first) / safe
    first = np.where(small, 1 - tau / 2 + tau**2 / 6 - tau**3 / 24
                     + tau**4 / 120 - tau**5 / 720, first)
    second = np.where(small, .5 - tau / 6 + tau**2 / 24 - tau**3 / 120
                      + tau**4 / 720 - tau**5 / 5040, second)
    return first, second


def _chords(edges, angle_order):
    panels = np.unique(np.r_[0., edges])
    t, weights = np.polynomial.legendre.leggauss(angle_order)
    t, weights = (t + 1) / 2, weights / 2
    area = np.diff(panels) * (panels[1:] + panels[:-1])
    p2 = (panels[1:, None]**2 - area[:, None] * t**2).ravel()
    pdp = (area[:, None] * t * weights).ravel()
    z = np.sqrt(np.maximum(edges[None, :]**2 - p2[:, None], 0.))
    volume = np.diff(edges) * (edges[1:]**2 + edges[1:] * edges[:-1]
                               + edges[:-1]**2) / 3
    return p2, pdp, z, volume


def moving_transfer(state, velocity_kms, *, angle_order=8, segments=4,
                    spectral_refinement=1, uniform_los_boost_kms=0.):
    """Integrate radial-flow line absorption, split by emitting shell.

    Each half-shell chord is subdivided into ``segments`` equal distances.
    The radial velocity's projection varies along the chord; the homogeneous
    subsegment formal solution and its path-mean intensity are analytic.
    Impact-parameter panels are identical to the production spherical solver.

    ``pump_by_source[cell, branch, source]`` is net stimulated-corrected line
    absorption in photons cm^-3 s^-1. Sources 0..N-1 are line photons emitted
    by individual cells, source N is the continuum. Local emission is kept
    separate so trapped photons are not confused with remote pumping.

    The optional common LOS boost translates the frequency grid and all line
    centers together in the narrow-band Galilean approximation. It is a frame
    invariance control, not an additional spherical radial velocity. The
    photon-energy denominator is evaluated in the original frequency frame.
    Relative radial shifts use each branch's own rest frequency. Relativistic
    intensity/angle transformations, and frequency redistribution, are absent.
    """
    if angle_order < 1 or segments < 1 or spectral_refinement < 1:
        raise ValueError("Quadrature and refinement orders must be positive.")
    edges = np.asarray(state["edges_cm"], dtype=float)
    velocity = np.asarray(velocity_kms, dtype=float)
    cells = edges.size - 1
    if velocity.shape != (cells,) or not np.isfinite(velocity).all():
        raise ValueError("Supply one finite radial velocity per cell.")
    original_wn = state["wavenumber_cm"]
    wn = np.interp(np.linspace(0, len(original_wn) - 1,
                              (len(original_wn) - 1) * spectral_refinement + 1),
                   np.arange(len(original_wn)), original_wn)
    continuum_chi = np.array([np.interp(wn, original_wn, row)
                              for row in state["continuum_extinction"]])
    continuum_eta = C_CGS * np.array([np.interp(wn, original_wn, row)
                                    for row in state["continuum_emission_hz"]])
    frequency_weights = np.r_[np.diff(wn)[0] / 2,
                              (wn[2:] - wn[:-2]) / 2,
                              np.diff(wn)[-1] / 2]
    photon_weights = 4 * np.pi * frequency_weights / (H_CGS * C_CGS * wn)
    frame_shift = np.mean(state["branch_centers"]) * uniform_los_boost_kms / C_KMS
    wn = wn + frame_shift
    p2, pdp, zedges, volume = _chords(edges, angle_order)
    rays, frequencies = p2.size, wn.size
    intensity = np.zeros((rays, cells + 1, frequencies))
    mean_intensity = np.zeros((cells, frequencies))
    pump = np.zeros((cells, len(state["branch_centers"]), cells + 1))
    max_depth = 0.
    for sign in (-1, 1):
        if sign == 1:
            intensity[p2 < edges[0]**2] = 0.
        indices = range(cells - 1, -1, -1) if sign == -1 else range(cells)
        for cell in indices:
            length = (zedges[:, cell + 1] - zedges[:, cell]) / segments
            geometric_weight = pdp * length / (2 * volume[cell])
            for segment in range(segments):
                fraction = ((segments - segment - .5) if sign == -1
                            else (segment + .5))
                z = sign * (zedges[:, cell] + fraction * length)
                radius = np.sqrt(p2 + z**2)
                vlos = velocity[cell] * z / np.where(radius > 0, radius, 1.)
                centers = (state["branch_centers"][None, :]
                           * (1 + vlos[:, None] / C_KMS) + frame_shift)
                profile = voigt_profile(
                    wn[None, None, :] - centers[:, :, None],
                    state["branch_sigma"][cell][None, :, None],
                    state["branch_gamma"][None, :, None])
                line_chi = state["branch_alpha"][cell][None, :, None] * profile
                line_eta = np.sum(state["branch_eta"][cell][None, :, None]
                                  * profile, axis=1)
                chi = np.sum(line_chi, axis=1) + continuum_chi[cell]
                depth = chi * length[:, None]
                max_depth = max(max_depth, float(np.max(depth)))
                first, second = _weights(depth)
                mean = intensity * first[:, None, :]
                mean[:, cell, :] += line_eta * length[:, None] * second
                mean[:, -1, :] += continuum_eta[cell] * length[:, None] * second
                mean_intensity[cell] += np.einsum(
                    "q,qf->f", geometric_weight, np.sum(mean, axis=1))
                pump[cell] += np.einsum(
                    "qbf,qsf,q,f->bs", line_chi, mean, geometric_weight,
                    photon_weights, optimize=True)
                intensity *= np.exp(-depth)[:, None, :]
                intensity[:, cell, :] += line_eta * length[:, None] * first
                intensity[:, -1, :] += continuum_eta[cell] * length[:, None] * first
    geometry_error = np.max(np.abs(np.sum(pdp[:, None] * np.diff(zedges), axis=0)
                                   / volume - 1))
    if not np.isfinite(pump).all() or not np.isfinite(mean_intensity).all():
        raise FloatingPointError("Non-finite moving transfer result.")
    return {"wavenumber_cm": wn, "mean_intensity": mean_intensity,
            "pump_by_source": pump, "geometry_relative_error": float(geometry_error),
            "maximum_subsegment_depth": max_depth,
            "angle_order": angle_order, "segments_per_half_shell": segments,
            "spectral_refinement": spectral_refinement}


def metrics(result, state):
    pump = result["pump_by_source"].sum(axis=1)
    local = np.diag(pump[:, :-1])
    total = pump.sum(axis=1)
    remote = pump[:, :-1].sum(axis=1) - local
    inner = state["radius_rp"] < 1.3
    inner_pump = pump[:, :-1][:, inner].sum(axis=1)
    wn = state["wavenumber_cm"]
    profile = voigt_profile(wn[None, None, :] - state["branch_centers"][None, :, None],
                            state["branch_sigma"][:, :, None],
                            state["branch_gamma"][None, :, None])
    spontaneous = 4 * np.pi * np.trapezoid(
        (state["branch_eta"][:, :, None] * profile).sum(axis=1)
        / (H_CGS * C_CGS * wn), wn, axis=-1)
    return {"radial_velocity_kms": np.asarray(result["velocity_kms"]).tolist(),
            "total_line_absorption_cm3_s": total.tolist(),
            "rest_frame_spontaneous_emission_cm3_s": spontaneous.tolist(),
            "net_absorption_minus_rest_emission_cm3_s": (total - spontaneous).tolist(),
            "local_line_photon_absorption_cm3_s": local.tolist(),
            "remote_line_photon_absorption_cm3_s": remote.tolist(),
            "inner_line_photon_absorption_cm3_s": inner_pump.tolist(),
            "continuum_photon_absorption_cm3_s": pump[:, -1].tolist(),
            "geometry_relative_error": result["geometry_relative_error"],
            "elapsed_s": result["elapsed_s"]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angle-order", type=int, default=8)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--spectral-refinement", type=int, default=1)
    parser.add_argument("--cases", nargs="+", default=["static", "native", "upper_plus_5p5", "upper_plus_13"])
    parser.add_argument("--refine-case", choices=["static", "native", "upper_plus_5p5", "upper_plus_13"],
                        help="Independently double angle, chord, spectral and wing resolution for this case.")
    args = parser.parse_args()
    state = load_state(args.input_dir, args.recipe)
    upper = state["radius_rp"] >= 1.3
    velocities = {"static": np.zeros_like(state["radial_velocity_kms"]),
                  "native": state["radial_velocity_kms"],
                  "upper_plus_5p5": state["radial_velocity_kms"] + 5.5 * upper,
                  "upper_plus_13": state["radial_velocity_kms"] + 13 * upper}
    output, arrays = {}, {"radius_rp": state["radius_rp"]}
    for name in args.cases:
        started = time.monotonic()
        result = moving_transfer(state, velocities[name], angle_order=args.angle_order,
                                 segments=args.segments,
                                 spectral_refinement=args.spectral_refinement)
        result.update(velocity_kms=velocities[name], elapsed_s=time.monotonic() - started)
        output[name] = metrics(result, state)
        for key in ("pump_by_source", "mean_intensity", "wavenumber_cm"):
            arrays[f"{name}_{key}"] = result[key]
        print(f"{name}: {result['elapsed_s']:.2f} s", flush=True)
    refinements = {}
    if args.refine_case:
        if args.refine_case not in output:
            parser.error("--refine-case must also appear in --cases")
        for kind in ("angle", "chord", "frequency", "wings"):
            refined_state = dict(state)
            if kind == "wings":
                wn = state["wavenumber_cm"]
                midpoint, halfwidth = (wn[0] + wn[-1]) / 2, (wn[-1] - wn[0]) / 2
                wide = np.linspace(midpoint - 2 * halfwidth, midpoint + 2 * halfwidth,
                                   2 * (wn.size - 1) + 1)
                refined_state["wavenumber_cm"] = wide
                for key in ("continuum_extinction", "continuum_emission_hz"):
                    refined_state[key] = np.array([np.interp(wide, wn, row) for row in state[key]])
            started = time.monotonic()
            result = moving_transfer(refined_state, velocities[args.refine_case],
                                     angle_order=args.angle_order * (2 if kind == "angle" else 1),
                                     segments=args.segments * (2 if kind == "chord" else 1),
                                     spectral_refinement=args.spectral_refinement * (2 if kind == "frequency" else 1))
            result.update(velocity_kms=velocities[args.refine_case], elapsed_s=time.monotonic() - started)
            refinements[kind] = metrics(result, refined_state)
            for key in ("pump_by_source", "mean_intensity", "wavenumber_cm"):
                arrays[f"refinement_{kind}_{key}"] = result[key]
            print(f"refinement {kind}: {result['elapsed_s']:.2f} s", flush=True)
    output = {"scope": "Frozen populations, temperature and continuum; finite spherical chords with local radial Doppler profiles. No SE, thermal, scattering redistribution or observer fit solve.",
              "boundary": "Dark inner and outer diffuse boundaries; line and continuum volume emission retained.",
              "source_convention": "Net stimulated-corrected line absorption; source cells 0..7 and continuum last. Inner means cell center below 1.3 Rp, not only dense LTE cells.",
              "net_rate_convention": "Subtract fixed integrated spontaneous emission evaluated on the rest-frame grid. Narrow-band nonrelativistic diagnostic; no population change is inferred from this residual.",
              "velocity_convention": "Positive radial flow is outward. Upper probes add outward velocity to every cell centered at >= 1.3 Rp; they are not uniform observer LOS shifts.",
              "inputs": {str(path): sha256(path) for path in
                         [args.input_dir / "fixed_inputs.npz", args.input_dir / "branch_arrays.npz", args.recipe]},
              "source_sha256": sha256(__file__), "radius_rp": state["radius_rp"].tolist(),
              "numerics": {"angle_order": args.angle_order, "segments_per_half_shell": args.segments,
                           "spectral_refinement": args.spectral_refinement}, "cases": output,
              "refinement_case": args.refine_case, "refinements": refinements,
              "wing_refinement_convention": "Double frequency half-width at approximately fixed spacing; hold continuum endpoint coefficients constant outside the saved interval."}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    output["raw_arrays_sha256"] = sha256(args.output / "arrays.npz")
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
