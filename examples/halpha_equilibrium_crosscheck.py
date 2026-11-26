"""Re-evaluate a saved complete atmosphere on CPU without changing its state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import exojax

from halpha_equilibrium import build_model, digest, physical_checks, write_json

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)


def run(args):
    if jax.default_backend() != "cpu":
        raise ValueError("Run this independent crosscheck with JAX_PLATFORMS=cpu.")
    root = Path(__file__).resolve().parents[1]
    summary_path, configuration_path = (args.root_dir/name for name in
                                         ("summary.json", "configuration.json"))
    summary = json.loads(summary_path.read_text())
    configuration = json.loads(configuration_path.read_text())
    if not (summary.get("converged") and summary.get("inputs_unchanged")
            and summary["checks"]["passed"] and summary["stages"][-1]["blend"] == 1.
            and summary["residual_infinity"] < 1e-9
            and summary["repeated_residual_difference"] < 1e-10
            and summary["baseline_residual_difference"] < 1e-10):
        raise ValueError("The source run must satisfy every complete-root acceptance gate.")
    pins = dict(configuration["source_and_input_sha256"])
    for path in (args.recipe, args.stellar, args.nist):
        if str(path.resolve()) not in pins:
            raise ValueError(f"The requested model input is absent from the source run: {path}")
    dependency_path = root/"mceg/evidence/halpha_equilibrium_20261126/dependency_snapshot.json"
    dependency = json.loads(dependency_path.read_text())
    external = dependency["runtime_exojax"]
    if not Path(exojax.__file__).resolve().is_relative_to(Path(external["path"]).resolve()):
        raise ValueError("Use the archived ExoJAX source tree recorded by the source run.")
    pins.update({str(Path(external["path"])/name): value
                 for name, value in external["python_source_sha256"].items()})
    pins.update({str(root/name): value for name, value in dependency["atomic_tables"].items()})
    if any(digest(path) != value for path, value in pins.items()):
        raise ValueError("Source-run input or dependency hashes no longer match.")
    state_path = args.root_dir/"fresh_final.npz"
    for path in (summary_path, configuration_path, state_path, dependency_path, Path(__file__).resolve()):
        pins[str(path)] = digest(path)
    backend_path = args.root_dir/"runtime_backend.json"
    reference_backend = None
    if backend_path.exists():
        pins[str(backend_path)] = digest(backend_path)
        reference_backend = json.loads(backend_path.read_text())
    with np.load(state_path) as saved:
        theta, reference_f = saved["theta"].copy(), saved["residual"].copy()
        fields = ("temperature_k", "electron_cm3", "bound_cm3", "radial_velocity_kms")
        reference = {key: saved[key].copy() for key in fields}
    if not np.isfinite(reference_f).all() or np.max(np.abs(reference_f)) >= 1e-9:
        raise ValueError("The saved source residual is not a complete root.")
    mass_loss = float(np.exp(theta[-1]))
    if not np.isclose(mass_loss, configuration["mass_loss_g_s"], rtol=1e-13, atol=0):
        raise ValueError("Saved state and source configuration have different mass loss.")
    started = time.monotonic()
    band = configuration["halpha_band"]
    model, boundary, recipe = build_model(args.recipe, args.stellar, args.nist,
        configuration["geometry"], band["wing_extent"], band["angle_order"], band["quadrature_order"])
    reference_t = jnp.asarray(recipe["temperature_reference_k"])

    @jax.jit
    def full(value):
        return model.evaluate_atmosphere(value, reference_t, mass_loss,
            boundary["incident_mean_intensity"], hydrogen_line_intensity=boundary["hydrogen_line_intensity"],
            metal_line_intensity=boundary["metal_line_intensity"], overlap_intensity=boundary["overlap_intensity"],
            infrared_intensity=boundary["infrared_intensity"],
            halpha_intensity=boundary.get("halpha_intensity"), halpha_blend=jnp.asarray(1.))

    print("Evaluating the unchanged final coordinates once on CPU.", flush=True)
    residual, physical = jax.tree.map(np.asarray, full(jnp.asarray(theta[:-1])))
    checks = physical_checks(physical, residual)
    finite = bool(np.isfinite(residual).all())
    maximum = float(np.max(np.abs(residual))) if finite else None
    difference = float(np.max(np.abs(residual-reference_f))) if finite else None
    unchanged = all(digest(path) == value for path, value in pins.items())
    comparisons = {}
    def finite_maximum(value):
        return float(np.max(value)) if np.isfinite(value).all() else None
    for key, original in reference.items():
        delta = np.abs(physical[key]-original)
        positive = original != 0
        comparisons[key] = dict(maximum_absolute_difference=finite_maximum(delta),
            maximum_relative_difference_on_nonzero_reference=finite_maximum(
                np.divide(delta, np.abs(original), out=np.zeros_like(delta), where=positive)),
            maximum_absolute_difference_on_zero_reference=finite_maximum(np.where(positive, 0., delta)))
    passed = bool(finite and maximum < 1e-9 and difference < 1e-10
                  and checks["passed"] and unchanged)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output/"cpu_final.npz"
    np.savez_compressed(result_path, theta=theta, residual=residual,
                        **{key: physical[key] for key in fields})
    review = dict(scope="One CPU evaluation of the exact saved final coordinates, with full residual and diagnostics. No coordinate update or CPU polishing is performed.",
        passed=passed, backend=jax.default_backend(), x64=jax.config.x64_enabled,
        persistent_cache=jax.config.jax_enable_compilation_cache,
        jax_version=jax.__version__, exojax_path=exojax.__file__, affinity=sorted(os.sched_getaffinity(0)),
        reference_backend=reference_backend,
        source_and_input_sha256=pins, source_summary=str(summary_path), mass_loss_g_s=mass_loss,
        halpha_band=band, residual_infinity=maximum, residual_difference_infinity=difference,
        criteria=dict(cpu_residual_infinity_max=1e-9, residual_difference_infinity_max=1e-10,
                      conservation_relative_error_max=1e-12),
        checks=checks, physical_comparisons=comparisons, inputs_unchanged=unchanged,
        elapsed_seconds=time.monotonic()-started,
        output=dict(path=str(result_path), sha256=digest(result_path)))
    args.review.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.review, review)
    print(json.dumps(dict(passed=passed, residual_infinity=maximum,
                          residual_difference_infinity=difference)), flush=True)
    if not passed:
        raise RuntimeError("The unchanged-state CPU crosscheck failed; inspect the receipt before any polishing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--nist", type=Path, default=Path("/tmp/escape-five-cell-root-evidence/nist_row.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/escape-halpha-equilibrium-cpu-crosscheck"))
    parser.add_argument("--review", type=Path,
                        default=Path("mceg/evidence/halpha_equilibrium_20261126/cpu_crosscheck.json"))
    run(parser.parse_args())
