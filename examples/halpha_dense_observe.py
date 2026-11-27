"""Compare a supplied dense atmosphere through the unchanged native observer.

The saved atmosphere supplies every line/continuum coefficient. This diagnostic
does not establish atomic, thermal or dynamical equilibrium. Baseline spectra,
GP hyperparameters, native samples and complete-band maps are reused unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from halpha_geometry_fit import (C_KMS, GRIDS, PHASE, physical_response,
                                 render_state, sha256)
from halpha_cause_kinematics import prepare_evaluator, summarize
from halpha_data_consistency import PARAMETERS


def validated_state(path, edges, velocity):
    state = dict(np.load(path))
    cells, waves = len(edges) - 1, len(velocity)
    shapes = dict(temperature_k=(cells,), bound_cm3=(cells, 18),
                  radial_velocity_kms=(cells,), wavenumber_cm=(waves,),
                  continuum_extinction=(cells, waves), continuum_emission_hz=(cells, waves))
    for key, shape in shapes.items():
        if key not in state or state[key].shape != shape or not np.isfinite(state[key]).all():
            raise ValueError(f"Invalid supplied physical field {key}; expected {shape}")
    if np.any(state["temperature_k"] <= 0) or np.any(state["bound_cm3"] < 0):
        raise ValueError("Temperature must be positive and bound populations nonnegative")
    np.testing.assert_allclose((C_KMS * (15233.146 / state["wavenumber_cm"] - 1))[::-1],
                               velocity, rtol=0, atol=1e-10)
    return state


def render(args, state, edges, source_hashes):
    identity = dict(state_sha256=sha256(args.state_npz), recipe_sha256=sha256(args.recipe),
                    source_sha256=source_hashes, grid=GRIDS[args.grid],
                    zonal_kms=args.zonal_kms, profile_step_kms=.05)
    rows = []
    for index, phase in enumerate(PHASE):
        path = args.raw_output / f"phase_{index:02d}.npz"
        if path.exists():
            saved = dict(np.load(path))
            if json.loads(str(saved["identity"])) != identity or float(saved["phase"]) != phase:
                raise ValueError(f"Stale observer cache: {path}")
            spectrum = saved["spectrum"]
        else:
            spectrum = render_state(state, edges, phase, GRIDS[args.grid], args.zonal_kms)
            np.savez_compressed(path, spectrum=spectrum, phase=phase, identity=json.dumps(identity))
        if not np.isfinite(spectrum).all():
            raise ValueError("Nonfinite observer spectrum")
        rows.append(spectrum)
        print(f"Observer phase {phase:.3f} ready", flush=True)
    return np.asarray(rows)


def reused_spectrum(args, state, baseline, velocity, repository):
    """Accept an old render only after checking state, code and array identity."""
    review = json.loads(args.render_review.read_text())
    if review["output_sha256"]["spectra.npz"] != sha256(args.candidate_spectra):
        raise ValueError("Saved spectra hash does not match its render receipt")
    if review["inputs"][str(args.rendered_state)] != sha256(args.rendered_state):
        raise ValueError("Archived render state hash mismatch")
    old_state = dict(np.load(args.rendered_state))
    for key in ("temperature_k", "bound_cm3", "radial_velocity_kms", "wavenumber_cm",
                "continuum_extinction", "continuum_emission_hz"):
        np.testing.assert_array_equal(state[key], old_state[key])
    for old_path, digest in review["inputs"].items():
        for marker in ("/examples/", "/src/escape/"):
            if marker in old_path:
                current = repository / (marker[1:] + old_path.split(marker, 1)[1])
                if sha256(current) != digest:
                    raise ValueError(f"Archived renderer source differs: {current}")
    if (review["grid"] != GRIDS[args.grid] or review["zonal_kms"] != args.zonal_kms
            or review["profile_step_kms"] != .05 or sha256(args.recipe) not in review["inputs"].values()):
        raise ValueError("Archived render geometry differs")
    saved = dict(np.load(args.candidate_spectra))
    np.testing.assert_array_equal(saved["phase"], PHASE)
    np.testing.assert_array_equal(saved["velocity"], velocity)
    np.testing.assert_array_equal(saved["baseline"], baseline)
    return saved["candidate"]


def compact(record, reference, evaluator, labels):
    result = summarize(record, reference, evaluator["row_ids"], labels)
    result["optimizer_start_Q"] = record.get("start_Q")
    for index, row_id in enumerate(evaluator["row_ids"]):
        row = labels[row_id]
        if row["product"] == "common_complete" and row["phase_index"] == 3:
            target = result["target"][row["scope"]]
            if target is not None:
                for field in ("observed", "planet", "gp"):
                    target[row["band"]][field + "_ppm"] = float(record[field][index]) if record["support"][index] else None
    return result


def plot(results, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    bands = ["blue", "core", "red"]
    for ax, (selection, values) in zip(axes, results.items()):
        for offset, name, mode, label, color in [
                (-.16, "baseline", "profiled", "Baseline, RV/A refit", "0.5"),
                (.16, "candidate", "profiled", "Candidate, RV/A refit", "#367dad")]:
            residuals = [values[name][mode]["target"]["all"][b]["residual_ppm"] for b in bands]
            ax.bar(np.arange(3) + offset, residuals, .31, label=label, color=color)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(np.arange(3), ["Blue", "Core", "Red"])
        gain = values["candidate"]["profiled"]["Q_improvement"]
        ax.set_title(f"{selection.capitalize()} pixels: Q change = {-gain:+.2f}\nPositive Q change means worse", fontsize=10)
        ax.set_xlabel("Same 12 complete exposures, phase 0.37615–0.39322")
    axes[0].set_ylabel("Data − planet − conditional GP [ppm]")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Supplied atmosphere {destination.name}: same native observer and fixed GP covariance", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"complete_band_comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-npz", type=Path)
    parser.add_argument("--state-review", type=Path, help="Separate physical-state qualification; pinned, not reinterpreted")
    parser.add_argument("--candidate-spectra", type=Path, help="Optional previously rendered spectra.npz")
    parser.add_argument("--render-review", type=Path)
    parser.add_argument("--rendered-state", type=Path, help="Exact original state pinned by --render-review")
    parser.add_argument("--baseline-only", action="store_true", help="Fast operator parity without any new transfer")
    parser.add_argument("--grid", choices=GRIDS, default="fine")
    parser.add_argument("--zonal-kms", type=float, default=19.99871490128246)
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--maps", type=Path, default=Path("/tmp/escape-halpha-phase-localization"))
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline_only == (args.state_npz is not None):
        parser.error("Choose --baseline-only or --state-npz")
    if args.state_npz and not args.state_review:
        parser.error("--state-npz requires --state-review for physical qualification provenance")
    if any((args.candidate_spectra, args.render_review, args.rendered_state)) and not all((
            args.candidate_spectra, args.render_review, args.rendered_state, args.state_npz)):
        parser.error("Reuse requires --state-npz, --candidate-spectra, --render-review and --rendered-state")
    for path in (args.output, args.raw_output):
        path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repository = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        "halpha_geometry_fit.py", "halpha_cause_kinematics.py", "halpha_data_consistency.py",
        "halpha_noise_audit.py", "gp_halpha.py", "phase_halpha.py", "cloudy_halpha.py")],
        *[repository / "src/escape" / name for name in ("longitude_map.py", "nlte_forward.py",
        "hydrogen_atom.py", "hydrogen_rates.py", "time_gp.py", "stellar_radiation.py")]]
    source_hashes = {str(path): sha256(path) for path in sources}
    baseline_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    baseline_state_path = Path("/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz")
    surface_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    noise_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    observations = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    paths = [baseline_path, baseline_state_path, surface_path, noise_path, args.recipe,
             args.maps / "maps.npz", args.maps / "rows.json", *observations, *sources]
    if args.state_npz:
        paths.extend([args.state_npz, args.state_review])
    if args.candidate_spectra:
        paths.extend([args.candidate_spectra, args.render_review, args.rendered_state])
    input_hashes = {str(path): sha256(path) for path in paths}
    archive = dict(np.load(baseline_path))
    spectra = dict(baseline=archive["fine_intrinsic_phases"][:len(PHASE)])
    velocity = archive["model_velocity_kms"]
    edges = np.asarray(json.loads(args.recipe.read_text())["edges_rp"])
    response = None
    if args.baseline_only:
        spectra["candidate"] = spectra["baseline"].copy()
    else:
        state = validated_state(args.state_npz, edges, velocity)
        baseline_state = validated_state(baseline_state_path, edges, velocity)
        response = physical_response(baseline_state, state, edges)
        spectra["candidate"] = (reused_spectrum(args, state, spectra["baseline"], velocity, repository)
                                if args.candidate_spectra else render(args, state, edges, source_hashes))
    np.testing.assert_equal(spectra["candidate"].shape, spectra["baseline"].shape)
    if not np.isfinite(spectra["candidate"]).all():
        raise ValueError("Nonfinite candidate spectra")
    maps = np.load(args.maps / "maps.npz")
    metadata = json.loads((args.maps / "rows.json").read_text())
    labels = metadata["rows"]
    surface = dict(np.load(surface_path))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in np.load(noise_path).items()}
    results, selected_rows, raw = {}, {}, dict(phase=PHASE, velocity=velocity, **spectra)
    for selection, offset in (("primary", 0), ("alternate", 1)):
        evaluator = prepare_evaluator(selection, offset, PHASE, velocity, observations, surface, noise, maps, labels)
        selected_rows[selection] = evaluator["row_ids"]
        reference = evaluator["evaluate"](spectra["baseline"])
        profiled_reference = evaluator["profile"](spectra["baseline"])
        if selection == "primary":
            np.testing.assert_allclose(reference["Q"], 38326.64894003179, rtol=0, atol=1e-6)
        results[selection] = {}
        for name, spectrum in spectra.items():
            fixed = reference if name == "baseline" else evaluator["evaluate"](spectrum)
            profiled = profiled_reference if name == "baseline" or args.baseline_only else evaluator["profile"](spectrum)
            if not profiled["success"] or not np.isfinite(profiled["Q"]):
                raise RuntimeError(f"Unsuccessful nuisance fit: {selection}/{name}: {profiled['message']}")
            results[selection][name] = {}
            for mode, record, previous in (("fixed", fixed, reference), ("profiled", profiled, profiled_reference)):
                results[selection][name][mode] = compact(record, previous, evaluator, labels)
                for field in ("observed", "planet", "gp", "residual", "support"):
                    raw[f"{selection}__{name}__{mode}__{field}"] = record[field]
        print(f"{selection}: profiled candidate Q={profiled['Q']:.9f}", flush=True)
    raw_path = args.raw_output / "comparison_arrays.npz"
    np.savez_compressed(raw_path, **raw)
    plot(results, args.output)
    target_id = next(r["row_id"] for r in labels if r["grid"] == "fine" and r["product"] == "common_complete"
                     and r["phase_index"] == 3 and r["scope"] == "all" and r["band"] == "blue")
    target_support = {s: metadata["selections"][s]["rows"][target_id] for s in results}
    report = dict(scope=__doc__, baseline_only=args.baseline_only, reused_render=bool(args.candidate_spectra), grid=GRIDS[args.grid],
        parameters=PARAMETERS.tolist(), nuisance_bounds=dict(rv_kms=[-15, 15], log_amplitude=[-3, 3]),
        observer="Actual supplied T/populations/continuum/flow; spherical dayside disk and halo, same transparent dark nightside, inner boundary, rotation, zonal flow, instrumental profile and stellar normalization as baseline.",
        target_definition="Sealed common_complete phase row 3: uniform 4 km/s subbins over blue [-40,-16), core [-8,8), red [16,40); equal exposures and orders. All native phases and nights enter Q and GP conditioning.",
        target_support=target_support, row_ids=selected_rows, selections=results, physical_response=response,
        limitations=["The input state has separate physical qualification; observer success does not establish full SE, radiative, energy, force or mass-continuity closure.",
            "GP hyperparameters are fixed; conditional GP means are recomputed for each template. Each interleave has its own projector and independently profiled shared RV/amplitude.",
            "Two nuisance starts provide a local optimization check, not a global optimum guarantee. Q is descriptive, not evidence or significance.",
            "Unsupported bands are null. No late-blue extrapolation is made; the same complete target exposures are used for all templates.",
            "Seven phase nodes and the eight atmospheric cells are retained. Fine observer rays alone do not establish phase or atmospheric radial convergence."],
        inputs=input_hashes, raw_arrays=dict(path=str(raw_path), sha256=sha256(raw_path)),
        figure_sha256={p.name: sha256(p) for p in args.output.glob("complete_band_comparison.*")},
        elapsed_s=time.monotonic() - started)
    changed = [path for path, digest in input_hashes.items() if sha256(path) != digest]
    if changed:
        raise RuntimeError(f"Inputs changed during observer run: {changed}")
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
