"""Finite ray-grid sensitivity at the two phase nodes bracketing the target.

The other five nodes remain unchanged. This is not a global accuracy bound,
phase-grid refinement or atmospheric radial-grid refinement.
"""
import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY / "examples"))
from halpha_dense_observe import compact, validated_state
from halpha_geometry_fit import PHASE, render_state, sha256
from halpha_cause_kinematics import prepare_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="Existing independently CPU-qualified observer case directory")
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    original_path, assessment_path = args.case / "review.json", args.case / "stage_assessment.json"
    original, assessment = [json.loads(path.read_text()) for path in (original_path, assessment_path)]
    if assessment["status"] != "independent_CPU_verified":
        raise ValueError("Refine only an independently CPU-qualified case")
    if assessment["observer_review_sha256"] != sha256(original_path):
        raise ValueError("Observer receipt differs from the CPU-qualified assessment")
    if sha256(assessment["state_review"]["path"]) != assessment["state_review"]["sha256"]:
        raise ValueError("Qualified state review changed")
    baseline_grid = dict(nlon=96, nlat=48, nrad_per_interval=8, naz=96, subsegments=8)
    if original["grid"] != baseline_grid:
        raise ValueError("This control expects the recorded fine-grid observer")
    for path, digest in original["inputs"].items():
        if sha256(path) != digest:
            raise ValueError(f"Original observer input changed: {path}")
    raw_path = Path(original["raw_arrays"]["path"])
    if sha256(raw_path) != original["raw_arrays"]["sha256"]:
        raise ValueError("Original observer arrays changed")
    saved = dict(np.load(raw_path))
    np.testing.assert_array_equal(saved["phase"], PHASE)
    source = Path(assessment["state_review"]["path"]).parent / "fixed_inputs.npz"
    if sha256(source) != original["inputs"][str(source)]:
        raise ValueError("Original physical state changed")
    recipe = Path("/tmp/escape-corrected-observer-full-f/recipe.json")
    edges = np.asarray(json.loads(recipe.read_text())["edges_rp"])
    state = validated_state(source, edges, saved["velocity"])
    grid = dict(nlon=144, nlat=72, nrad_per_interval=12, naz=144, subsegments=12)
    inputs = dict(original["inputs"])
    inputs.update({str(path): sha256(path) for path in (Path(__file__), original_path, assessment_path,
        source, recipe, raw_path, REPOSITORY / "examples/halpha_dense_observe.py",
        REPOSITORY / "examples/halpha_geometry_fit.py", REPOSITORY / "examples/halpha_cause_kinematics.py")})
    args.raw_output.mkdir(parents=True, exist_ok=True)
    spectrum = saved["candidate"].copy()
    identity = dict(inputs=inputs, grid=grid)
    for index in (3, 4):
        path = args.raw_output / f"phase_{index:02d}.npz"
        if path.exists():
            cached = dict(np.load(path))
            if json.loads(str(cached["identity"])) != identity:
                raise ValueError("Refinement cache belongs to different inputs")
            spectrum[index] = cached["spectrum"]
        else:
            spectrum[index] = render_state(state, edges, PHASE[index], grid, 19.99871490128246)
            np.savez_compressed(path, spectrum=spectrum[index], identity=json.dumps(identity))
        print(f"Refined target bracket phase {PHASE[index]:.3f}", flush=True)
    np.testing.assert_array_equal(spectrum[[0, 1, 2, 5, 6]], saved["candidate"][[0, 1, 2, 5, 6]])
    if not np.isfinite(spectrum).all():
        raise ValueError("Nonfinite refined spectrum")
    surface_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    noise_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    map_directory = Path("/tmp/escape-halpha-phase-localization")
    observations = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    for path in (surface_path, noise_path, map_directory / "maps.npz", map_directory / "rows.json", *observations):
        if sha256(path) != original["inputs"][str(path)]:
            raise ValueError(f"Original observer input changed: {path}")
        inputs[str(path)] = sha256(path)
    surface = dict(np.load(surface_path))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in np.load(noise_path).items()}
    maps = np.load(map_directory / "maps.npz")
    labels = json.loads((map_directory / "rows.json").read_text())["rows"]
    selections, nuisance_controls = {}, {}
    for selection, offset in (("primary", 0), ("alternate", 1)):
        evaluator = prepare_evaluator(selection, offset, PHASE, saved["velocity"], observations, surface, noise, maps, labels)
        objective = jax.jit(jax.value_and_grad(lambda p, template: evaluator["problem"]["q"](
            template * jnp.exp(p[1]), p[0]), argnums=0))
        values = {}
        for mode in ("fixed", "profiled"):
            evaluate = evaluator["evaluate"] if mode == "fixed" else evaluator["profile"]
            before, after = evaluate(saved["candidate"]), evaluate(spectrum)
            if mode == "profiled" and not after["success"]:
                raise RuntimeError("Refined nuisance fit did not converge")
            np.testing.assert_allclose(before["Q"], original["selections"][selection]["candidate"][mode]["Q"], rtol=0, atol=1e-7)
            values[mode] = compact(after, before, evaluator, labels)
            target = values[mode]["target"]["all"]
            values[mode]["target_blue_minus_core_residual_change_ppm"] = (
                target["core"]["change_in_planet_plus_GP_ppm"] - target["blue"]["change_in_planet_plus_GP_ppm"])
            if mode == "profiled":
                controls = {}
                for name, template, reference in (("original", saved["candidate"], before), ("refined", spectrum, after)):
                    fits = [minimize(lambda p: objective(p, jnp.asarray(template)),
                        [rv, reference["parameters"][1]], jac=True, method="L-BFGS-B",
                        bounds=[(-15, 15), (-3, 3)], options=dict(maxiter=120, ftol=1e-12, gtol=1e-6))
                        for rv in (-15., -7.5, 7.5, 15.)]
                    records = [dict(Q=float(fit.fun), parameters=fit.x.tolist(), success=bool(fit.success),
                                    message=str(fit.message)) for fit in fits]
                    best = min([reference["Q"]] + [row["Q"] for row in records if row["success"]])
                    controls[name] = dict(additional_starts=records, best_successful_Q=best,
                        improvement_over_original_two_starts=reference["Q"]-best,
                        two_start_Q_agrees_with_best_tried=abs(reference["Q"]-best) < 1e-6)
                nuisance_controls[selection] = controls
        selections[selection] = values
    arrays = args.raw_output / "refined_spectra.npz"
    np.savez_compressed(arrays, phase=PHASE, velocity=saved["velocity"], original=saved["candidate"], refined=spectrum)
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("A refinement input changed during the run")
    report = dict(scope=__doc__, original_grid=baseline_grid, grid=grid,
        grid_dimension_ratio=1.5, unchanged_zonal_speed_kms=19.99871490128246,
        changed_phase_indices=[3, 4], other_phase_nodes_identical=True,
        selections=selections, nuisance_multistart=nuisance_controls,
        inputs=inputs, raw_arrays=dict(path=str(arrays), sha256=sha256(arrays)),
        interpretation="A finite change when refining only the target-bracketing ray grids. Q is evaluated on the mixed-node template; it is not a globally refined fit or an error bound.")
    (args.case / "target_ray_refinement.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
