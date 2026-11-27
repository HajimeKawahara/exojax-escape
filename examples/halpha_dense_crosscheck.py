"""Check exact saved dense-atmosphere coordinates on CPU, without polishing.

One model and full-output JIT serve the D=1 control and every supplied stage.
The force is prescribed; its algebraic consistency is not an MHD solution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import time

import exojax
from exojax.utils.constants import G, MJ, kB
import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_forward import PLANET_RADIUS_CM
from escape.parker import PROTON_MASS_CGS
from halpha_dense_outer import checks, compact_physical, evaluate_dense
from halpha_equilibrium import build_model, digest, write_json

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)


def compare(actual, expected, absolute_floor=0.):
    """Report differences with a small field-scale absolute floor near zeros."""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(passed=False, reason="shape_or_nonfinite")
    if b.dtype.kind == "b":
        return dict(passed=bool(np.array_equal(a, b)))
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    scale = float(np.max(np.abs(b))) if b.size else 0.
    tolerance = 1e-8*np.abs(b)+np.maximum(1e-12*scale, absolute_floor)
    return dict(passed=bool(np.all(np.abs(a-b) <= tolerance)),
                maximum_absolute_difference=difference,
                difference_over_maximum_reference=difference/scale if scale else difference)


def flat_summary(p):
    selected = compact_physical(p)
    selected["mass_per_h"] = p["mass_per_h"]
    return {**{k: v for k, v in selected.items() if k not in ("wind_info", "metal_levels_cm3")},
            **{f"metal_levels__{k}": v for k, v in enumerate(selected["metal_levels_cm3"])},
            **{f"wind__{k}": v for k, v in selected["wind_info"].items()}}


def force_and_flow_checks(p, radius, factor, planet_mass, tide):
    """Independently expand momentum and monatomic flow-energy identities."""
    r = np.asarray(radius)
    u = np.clip((r/PLANET_RADIUS_CM-1.3)/.425, 0., 1.)
    q = np.exp(np.log(factor)*u*u*(3-2*u))
    dlogq = np.log(factor)*6*u*(1-u)/(.425*PLANET_RADIUS_CM)
    info = p["wind_info"]
    v, vref = np.asarray(p["radial_velocity_kms"])*1e5, info["reference_velocity_kms"]*1e5
    grad, soundgrad = info["log_velocity_gradient_cm1"], info["sound_squared_gradient_cm_s2"]
    rho = p["n_h_cm3"]*p["mass_per_h"]
    sound2 = kB*p["temperature_k"]/(PROTON_MASS_CGS*p["mean_molecular_weight"])
    gravity_tide = G*MJ*planet_mass/r**2-tide*r
    direct_force = v*v*grad+sound2*(-2/r-grad)+soundgrad+gravity_tide
    force_scale = np.maximum(np.abs(gravity_tide)+np.abs(soundgrad)
                             +np.abs(sound2*(2/r+grad))+np.abs(v*v*grad), 1.)
    reference_grad = grad+dlogq
    reference_force_scale = np.maximum(np.abs(vref*vref*reference_grad)
        +np.abs(sound2*(2/r+reference_grad))+np.abs(soundgrad)
        +np.abs(G*MJ*planet_mass/r**2)+np.abs(tide*r), 1.)
    force_error = float(np.max(np.abs(direct_force-info["required_support_acceleration_cm_s2"])/force_scale))
    advection = -1.5*rho*v*soundgrad
    expansion = -rho*sound2*v*(2/r+grad)
    reference_expansion = -(rho/q)*sound2*vref*(2/r+grad+dlogq)
    thermal_identity = expansion-reference_expansion-rho*v*sound2*dlogq
    identity_scale = np.maximum(np.abs(expansion)+np.abs(reference_expansion)
                                +np.abs(rho*v*sound2*dlogq), np.finfo(float).tiny)
    result = dict(q=compare(info["density_enhancement"], q),
        velocity=compare(v*q, vref), log_density_gradient=compare(info["log_enhancement_gradient_cm1"], dlogq),
        advection=compare(p["thermal_advection"], advection),
        expansion=compare(p["thermal_expansion"], expansion),
        force_identity_scaled_error=force_error,
        reference_force_term_scale_cm_s2=reference_force_scale.tolist(),
        expansion_identity_relative_error=float(np.max(np.abs(thermal_identity)/identity_scale)))
    result["passed"] = bool(all(result[k]["passed"] for k in
        ("q", "velocity", "log_density_gradient", "advection", "expansion"))
        and force_error < 1e-12 and result["expansion_identity_relative_error"] < 1e-12)
    return result


def run(args):
    if jax.default_backend() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu for this independent check.")
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve previous receipts.")
    root = Path(__file__).resolve().parents[1]
    dependency_path = root/"mceg/evidence/halpha_equilibrium_20261126/dependency_snapshot.json"
    dependency = json.loads(dependency_path.read_text())
    external = dependency["runtime_exojax"]
    if not Path(exojax.__file__).resolve().is_relative_to(Path(external["path"]).resolve()):
        raise ValueError("Use the archived ExoJAX import tree.")
    pins = {str(Path(external["path"])/k): v for k, v in external["python_source_sha256"].items()}
    pins.update({str(root/k): v for k, v in dependency["atomic_tables"].items()})
    configurations, historical_drivers = {}, []
    for directory in args.candidate:
        configuration_path = directory.parent/"configuration.json"
        while args.wait_for_stages and not configuration_path.exists():
            print(f"Waiting for source configuration: {configuration_path}", flush=True)
            time.sleep(30)
        configuration = json.loads(configuration_path.read_text())
        if configuration["radius_ramp_rp"] != [1.3, 1.725]:
            raise ValueError("Density-ramp configuration changed.")
        for path, value in configuration["source_and_input_sha256"].items():
            current_driver = root/"examples/halpha_dense_outer.py"
            if Path(path).resolve() == current_driver and digest(path) != value:
                archive = directory.parent/"driver_source.py"
                if (not archive.exists() or digest(archive) != value
                        or archive.read_text().split("def run(args):", 1)[0]
                        != current_driver.read_text().split("def run(args):", 1)[0]):
                    raise ValueError("Historical driver requires an exact archive and unchanged evaluator prefix.")
                pins[str(current_driver)] = digest(current_driver)
                historical_drivers.append(dict(candidate=str(directory),
                    archived_path=str(archive), archived_sha256=value,
                    current_path=str(current_driver), current_sha256=digest(current_driver),
                    evaluator_prefix_byte_identical=True,
                    scope="Revalidate historical coordinates under current source; original receipt stays unchanged."))
                path = str(archive)
            if path in pins and pins[path] != value:
                raise ValueError(f"Candidate source/input disagreement: {path}")
            pins[path] = value
        path = directory.parent/"configuration.json"
        pins[str(path)] = digest(path)
        configurations[directory] = configuration
    for path in (args.recipe, args.stellar, args.nist, args.baseline):
        if str(path) not in pins:
            raise ValueError(f"Requested model input is absent from candidate pins: {path}")
    for path in (args.baseline_state, args.baseline_inputs, dependency_path, Path(__file__).resolve()):
        pins[str(path)] = digest(path)
    if any(digest(path) != value for path, value in pins.items()):
        raise ValueError("Source, input or external dependency changed before the check.")
    with np.load(args.baseline) as saved:
        theta, residual = saved["theta"].copy(), saved["residual"].copy()
    with np.load(args.baseline_state) as saved:
        if not np.array_equal(theta, saved["theta"]):
            raise ValueError("Baseline coordinates differ from the accepted CPU state.")
    mass_loss = float(np.exp(theta[-1]))
    if any(not np.isclose(c["mass_loss_g_s"], mass_loss, rtol=1e-13, atol=0)
           for c in configurations.values()):
        raise ValueError("All stages must share the baseline mass loss.")
    model, boundary, recipe = build_model(args.recipe, args.stellar, args.nist, "spherical")
    tref = jnp.asarray(recipe["temperature_reference_k"])
    with np.load(args.baseline_inputs) as saved:
        wn = saved["wavenumber_cm"].copy()

    @jax.jit
    def full(value, log_factor):
        return evaluate_dense(model, value, tref, mass_loss, boundary, log_factor)

    coefficients = jax.jit(lambda t, p: model.continuum_coefficients_at(t, p, jnp.asarray(wn*2.99792458e10)))
    args.output.mkdir(parents=True)
    write_json(args.output/"configuration.json", dict(source_and_input_sha256=pins,
        backend=jax.default_backend(), x64=jax.config.x64_enabled,
        candidates=[str(path) for path in args.candidate],
        historical_driver_revalidation=historical_drivers,
        scope="CPU evaluation of exact saved coordinates; no polishing."))
    records, started, unavailable = [], time.monotonic(), []

    def cases():
        yield None, theta, residual, 0.
        for directory in args.candidate:
            while args.wait_for_stages and not (directory/"summary.json").exists():
                if (directory.parent/"summary.json").exists():
                    break
                print(f"Waiting for verified stage: {directory}", flush=True)
                time.sleep(30)
            if not (directory/"summary.json").exists():
                unavailable.append(str(directory))
                break
            summary = json.loads((directory/"summary.json").read_text())
            if not (summary.get("converged") and summary.get("accepted_for_observation")
                    and summary["checks"]["passed"] and summary["residual_infinity"] < 1e-9
                    and summary["repeated_residual_difference"] < 1e-10):
                unavailable.append(str(directory))
                break
            for key in ("checkpoint", "fixed_inputs", "physical_summary"):
                path = directory/f"{key}.npz"
                if digest(path) != summary[f"{key}_sha256"]:
                    raise ValueError(f"Stage artifact hash mismatch: {path}")
                pins[str(path)] = summary[f"{key}_sha256"]
            path = directory/"summary.json"
            pins[str(path)] = digest(path)
            with np.load(directory/"checkpoint.npz") as saved:
                current_theta, current_f = saved["theta"].copy(), saved["residual"].copy()
                current_log_factor = float(saved["log_density_factor"])
            if not np.isclose(np.exp(current_log_factor), summary["factor"], rtol=1e-13, atol=0):
                raise ValueError("Checkpoint and summary factors disagree.")
            if abs(current_theta[-1]-theta[-1]) >= 1e-13:
                raise ValueError("Candidate and baseline mass loss disagree.")
            yield directory, current_theta, current_f, current_log_factor

    for number, (directory, coordinates, expected_f, log_factor) in enumerate(cases()):
        label = "baseline_D1" if directory is None else f"case_{number}_D{np.exp(log_factor):g}"
        print(f"Full CPU check: {label}", flush=True)
        begin = time.monotonic()
        f, p = jax.tree.map(np.asarray, full(jnp.asarray(coordinates[:-1]), jnp.asarray(log_factor)))
        np.savez_compressed(args.output/f"{label}_raw_residual.npz", theta=coordinates,
                            residual=f, log_density_factor=log_factor)
        p["mass_per_h"] = np.asarray(model.mass_per_h)
        guard = checks(p, f, model.radius_cm, mass_loss)
        algebra = force_and_flow_checks(p, model.radius_cm, float(np.exp(log_factor)), model.planet_mass_mj,
                                       model.tidal_coefficient_s2)
        selected = {k: p[k] for k in ("bound_cm3", "proton_cm3", "electron_cm3", "molecule_cm3",
                    "helium_cm3", "metal_ion_cm3", "metal_levels_cm3", "atomic_lte_mask")}
        chi, eta = jax.tree.map(np.asarray, coefficients(jnp.asarray(p["temperature_k"]),
                                                       jax.tree.map(jnp.asarray, selected)))
        fields = {}
        with np.load(args.baseline_state if directory is None else directory/"physical_summary.npz") as saved:
            actual = flat_summary(p)
            for key in saved.files:
                if key not in ("theta", "residual"):
                    if key == "wind__reference_momentum_defect_cm_s2":
                        # The defect subtracts large force terms; near-zero
                        # outputs require an absolute term-scale tolerance.
                        scale = np.asarray(algebra["reference_force_term_scale_cm_s2"])
                        fields[key] = compare(actual[key], saved[key], 1e-12*scale)
                        fields[key]["maximum_difference_over_force_terms"] = float(
                            np.max(np.abs(actual[key]-saved[key])/scale))
                        fields[key]["force_term_scale_absolute_tolerance"] = 1e-12
                    else:
                        fields[key] = compare(actual[key], saved[key])
        with np.load(args.baseline_inputs if directory is None else directory/"fixed_inputs.npz") as saved:
            fields["wavenumber_cm"] = compare(wn, saved["wavenumber_cm"])
            fields["continuum_extinction"] = compare(chi, saved["continuum_extinction"])
            fields["continuum_emission_hz"] = compare(eta, saved["continuum_emission_hz"])
        finite = bool(np.isfinite(f).all() and np.isfinite(expected_f).all())
        maximum = float(np.max(np.abs(f))) if finite else None
        difference = float(np.max(np.abs(f-expected_f))) if finite else None
        passed = bool(finite and maximum < 1e-9 and difference < 1e-10 and guard["passed"]
            and guard["required_thermal_scaled_max"] < 1e-9 and algebra["passed"]
            and all(v["passed"] for v in fields.values()) and np.all(chi >= 0) and np.all(eta >= 0))
        path = args.output/f"{label}.npz"
        np.savez_compressed(path, theta=coordinates, residual=f, log_density_factor=log_factor,
            continuum_extinction=chi, continuum_emission_hz=eta, **flat_summary(p))
        record = dict(label=label, source_directory=str(directory) if directory else None,
            factor=float(np.exp(log_factor)), passed=passed, residual_infinity=maximum,
            residual_difference_infinity=difference, checks=guard, force_and_flow=algebra,
            physical_and_coefficient_comparisons=fields, elapsed_seconds=time.monotonic()-begin,
            output=dict(path=str(path), sha256=digest(path)))
        records.append(record)
        write_json(args.output/f"{label}.json", record)
        print(json.dumps(dict(label=label, passed=passed, residual_infinity=maximum,
                              residual_difference_infinity=difference)), flush=True)
        if not passed:
            break
    if args.export_full and records[-1]["passed"]:
        physical_path = args.output/"complete_physical.pkl"
        with physical_path.open("wb") as stream:
            pickle.dump(p, stream, protocol=5)
        records[-1]["full_physical"] = dict(path=str(physical_path), sha256=digest(physical_path))
        write_json(args.output/f"{records[-1]['label']}.json", records[-1])
    unchanged = all(digest(path) == value for path, value in pins.items())
    review = dict(scope="Exact-coordinate CPU checks only; no state update. Full native population/radiation/thermal equations with imposed support, not a momentum or MHD solution.",
        passed=bool(unchanged and all(r["passed"] for r in records)),
        backend=jax.default_backend(), x64=jax.config.x64_enabled,
        jax_version=jax.__version__, exojax_path=exojax.__file__, affinity=sorted(os.sched_getaffinity(0)),
        persistent_cache=jax.config.jax_enable_compilation_cache,
        source_and_input_sha256=pins, inputs_unchanged=unchanged, cases=records,
        historical_driver_revalidation=historical_drivers,
        unavailable_stages=unavailable, all_requested_stages_checked=not unavailable and len(records)==len(args.candidate)+1,
        elapsed_seconds=time.monotonic()-started,
        criteria=dict(residual_infinity=1e-9, residual_difference_infinity=1e-10,
                      conservation_relative_error=1e-12, field_rtol=1e-8, field_scale_atol=1e-12,
                      reference_momentum_defect_term_scale_atol=1e-12))
    write_json(args.output/"review.json", review)
    if not review["passed"]:
        raise RuntimeError("Independent CPU acceptance failed; inspect the preserved receipt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True,
                        help="Successful D*/ stage directories, evaluated in this order.")
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--nist", type=Path, default=Path("/tmp/escape-five-cell-root-evidence/nist_row.json"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/escape-halpha-equilibrium-gpu-cublas/checkpoint.npz"))
    parser.add_argument("--baseline-state", type=Path, default=Path("/tmp/escape-halpha-equilibrium-final-cpu/fresh_final.npz"))
    parser.add_argument("--baseline-inputs", type=Path, default=Path("/tmp/escape-halpha-equilibrium-final-cpu/fixed_inputs.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-full", action="store_true", help="Preserve the last case's full physical dictionary.")
    parser.add_argument("--wait-for-stages", action="store_true", help="Check D=1 now, then await stage receipts from a running continuation.")
    run(parser.parse_args())
