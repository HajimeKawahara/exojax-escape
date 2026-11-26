"""Re-solve the saved atmosphere with resolved H-alpha transfer.

The archived inverse is a numerical accelerator only. Every accepted step
decreases the actual complete residual, and only a fresh converged, guarded
state is exported for the observed-spectrum comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from escape.solar import SolarAtmosphereModel
from escape.stellar_radiation import matched_photospheric_ir_flux

jax.config.update("jax_enable_x64", True)

MODEL_OPTIONS = (
    "nmax", "resolved_nmax", "hydrogen_topoff", "iron_levels", "sulfur_levels",
    "iron_lyalpha", "dense_lower_cells", "lower_temperature_k", "wind_ngrid",
    "wind_outer_radius_rp", "extend_wind_to_edge", "quadrature_order", "angle_order",
    "line_sampling", "hydrogen_line_quadrature_order", "line_quadrature",
    "metal_line_quadrature", "line_quadrature_order", "shielding_model", "core_escape_model",
    "minimum_continuum_energy_ev", "infrared_continuum", "infrared_continuum_angle_order",
    "infrared_band_min_wavelength_micron", "resolved_infrared_lines",
    "resolved_infrared_metal_lines", "infrared_options",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_model(recipe_path, stellar_path, nist_path, geometry, wing_extent=30.,
                angle_order=32, frequency_order=8):
    recipe = json.loads(Path(recipe_path).read_text())
    with np.load(stellar_path) as saved:
        nu, flux = saved["frequency_hz"], saved["surface_flux_nu"]
    low = np.geomspace(1e9, nu[0], 4097)[:-1]
    tail = np.asarray(matched_photospheric_ir_flux(low, nu[0], flux[0], 10000.))
    nist = json.loads(Path(nist_path).read_text())["row"]
    with patch("exojax.database.nist.api.fetch_nist_lines",
               lambda *args, **kwargs: (pd.DataFrame(nist), [])):
        model = SolarAtmosphereModel(recipe["edges_rp"],
            **{name: recipe[name] for name in MODEL_OPTIONS},
            newton_merit="correction", tidal_coefficient_s2=3*(2*np.pi/(1.48*86400))**2,
            extra_frequency_edges=nu[[0, -1]], observer_options=dict(nlon=2, nlat=2, nnu=101),
            halpha_transfer=None if geometry == "none" else geometry,
            halpha_options=dict(wing_extent=wing_extent, angle_order=angle_order,
                                quadrature_order=frequency_order))
    boundary = model.stellar_boundary(np.r_[low, nu], np.r_[tail, flux],
        incident_flux_ratio=recipe["stellar_boundary"]["incident_flux_ratio"], outside_zero=True)
    return model, boundary, recipe


def physical_checks(physical, residual):
    p = physical
    velocity = np.asarray(p["radial_velocity_kms"])
    flags = dict(valid=bool(p["valid"]), reference_domain=bool(p["reference_domain_valid"]),
        wind_valid=bool(p["wind_info"]["valid"]), wind_converged=bool(p["wind_info"]["converged"]),
        positive_velocity=bool(np.isfinite(velocity).all() and np.all(velocity > 0)),
        finite_residual=bool(np.isfinite(residual).all()),
        prescribed_masks=bool(np.array_equal(p["atomic_lte_mask"], np.arange(len(velocity)) < 2)
            and np.array_equal(p["thermal_balance_required"], np.arange(len(velocity)) >= 2)))
    bound, molecules, helium, metals = (np.asarray(p[key]) for key in
        ("bound_cm3", "molecule_cm3", "helium_cm3", "metal_ion_cm3"))
    nuclei = bound.sum(1)+p["proton_cm3"]+(molecules*np.array([1., 2., 2., 2.])).sum(1)
    charge = (p["proton_cm3"]+helium[:, 1]+2*helium[:, 2]
        +(metals*np.arange(3)).sum((1, 2))+molecules[:, 1]-molecules[:, 0])
    particles = (bound.sum(1)+p["proton_cm3"]+molecules.sum(1)+helium.sum(1)
        +p["element_density_cm3"].sum(1)+p["electron_cm3"])
    pressure = particles*1.380649e-16*p["temperature_k"]
    errors = {name: float(np.max(np.abs(value/np.asarray(p[key])-1))) for name, value, key in
        [("hydrogen_nuclei", nuclei, "n_h_cm3"), ("charge", charge, "electron_cm3"),
         ("pressure", pressure, "pressure_dyn_cm2")]}
    positive = all(np.isfinite(p[key]).all() and np.all(np.asarray(p[key]) >= 0)
        for key in ("bound_cm3", "molecule_cm3", "helium_cm3", "metal_ion_cm3",
                    "electron_cm3", "temperature_k", "n_h_cm3"))
    flags["nonnegative_species"] = bool(positive)
    flags["conservation"] = all(value < 1e-12 for value in errors.values())
    return dict(passed=all(flags.values()), flags=flags,
        conservation_relative_error={key: value if np.isfinite(value) else None
                                     for key, value in errors.items()})


def solve_broyden(evaluate, x, residual, left, right, *, domain, save, event,
                  max_steps=64, tolerance=1e-9, maximum_coordinate_step=.5):
    """Inverse Broyden secants with actual-residual decrease, no derivative claim."""
    x, f, left, right = (np.array(a, copy=True) for a in (x, residual, left, right))
    accepted = 0
    for iteration in range(max_steps):
        if np.max(np.abs(f)) < tolerance:
            return x, f, left, right, dict(converged=True, accepted=accepted, reason="root_candidate")
        step = -f-left@(right@f)
        largest = np.max(np.abs(step))
        if not np.isfinite(largest) or largest == 0:
            break
        step *= min(1., maximum_coordinate_step/largest)
        norm = np.linalg.norm(f)
        found = False
        for backtrack in range(9):
            fraction = 2.**(-backtrack)
            trial = x+fraction*step
            if not domain(trial):
                continue
            trial_f, valid = evaluate(trial, "trial")
            trial_norm = np.linalg.norm(trial_f)
            found = bool(valid and np.isfinite(trial_norm)
                         and trial_norm < (1-1e-4*fraction)*norm)
            event("line_search", dict(iteration=iteration, fraction=fraction,
                  accepted=found, norm=float(trial_norm) if np.isfinite(trial_norm) else None))
            if found:
                break
        if not found:
            return x, f, left, right, dict(converged=False, accepted=accepted, reason="no_residual_decrease")
        displacement, change = trial-x, trial_f-f
        denominator = change@change
        if denominator > np.finfo(float).tiny:
            correction = displacement-change-left@(right@change)
            left = np.column_stack((left, correction))
            right = np.vstack((right, change/denominator))
        x, f = trial, trial_f
        accepted += 1
        save(x, f, left, right)
    return x, f, left, right, dict(converged=False, accepted=accepted, reason="iteration_limit")


def solve_subspace(evaluate, x, residual, left, right, *, domain, save, event,
                   max_steps=64, tolerance=1e-9, maximum_coordinate_step=.5,
                   max_basis=24, forcing=.2):
    """Measure current finite-difference directions before proposing a step.

    The retained inverse only chooses search directions. The least-squares
    model uses new complete residuals at one anchor, and acceptance still
    requires an actual residual decrease and every physical guard.
    """
    x, f, left, right = (np.array(a, copy=True) for a in (x, residual, left, right))
    accepted = 0
    for iteration in range(max_steps):
        if np.max(np.abs(f)) < tolerance:
            return x, f, left, right, dict(converged=True, accepted=accepted, reason="root_candidate")
        norm = np.linalg.norm(f)
        directions, products = [], []
        remainder = f.copy()
        for column in range(min(max_basis, len(x))):
            direction = -remainder-left@(right@remainder)
            for _ in range(2):
                for previous in directions:
                    direction -= previous@direction*previous
            length = np.linalg.norm(direction)
            if not np.isfinite(length) or length < 1e-14*norm:
                break
            direction /= length
            delta = min(1e-5, max(3e-8, 1e-3*np.sqrt(norm)))
            h = delta/np.max(np.abs(direction))
            probe = x+h*direction
            if not domain(probe):
                return x, f, left, right, dict(converged=False, accepted=accepted, reason="probe_domain")
            probe_f, valid = evaluate(probe, "basis")
            if not valid:
                return x, f, left, right, dict(converged=False, accepted=accepted, reason="probe_physical")
            directions.append(direction)
            products.append((probe_f-f)/h)
            s, y = np.column_stack(directions), np.column_stack(products)
            coefficients, _, rank, _ = np.linalg.lstsq(y, -f, rcond=1e-10)
            remainder = f+y@coefficients
            eta = np.linalg.norm(remainder)/norm
            event("current_basis", dict(iteration=iteration, columns=column+1,
                  rank=int(rank), predicted_relative_residual=float(eta), difference_step=float(h)))
            if eta <= forcing:
                break
        if not directions or eta >= .999:
            return x, f, left, right, dict(converged=False, accepted=accepted, reason="no_descent_model")
        # All columns belong to this same anchor. Discard singular directions
        # rather than interpreting an old secant as a current derivative.
        correction = s-y-left@(right@y)
        left = np.column_stack((left, correction))
        right = np.vstack((right, np.linalg.pinv(y, rcond=1e-10)))
        save(x, f, left, right)
        step = s@coefficients
        step *= min(1., maximum_coordinate_step/np.max(np.abs(step)))
        found = False
        for backtrack in range(9):
            fraction = 2.**(-backtrack)
            trial = x+fraction*step
            if not domain(trial):
                continue
            trial_f, valid = evaluate(trial, "trial")
            trial_norm = np.linalg.norm(trial_f)
            found = bool(valid and np.isfinite(trial_norm)
                         and trial_norm < (1-1e-4*fraction)*norm)
            event("line_search", dict(iteration=iteration, fraction=fraction,
                  accepted=found, norm=float(trial_norm) if np.isfinite(trial_norm) else None))
            if found:
                break
        if not found:
            return x, f, left, right, dict(converged=False, accepted=accepted, reason="no_residual_decrease")
        x, f = trial, trial_f
        accepted += 1
        save(x, f, left, right)
    return x, f, left, right, dict(converged=False, accepted=accepted, reason="iteration_limit")


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists() and not args.resume:
        raise ValueError("Use a new output directory or --resume.")
    start = time.monotonic()
    calls = 0
    def event(kind, values):
        row = dict(values, event=kind, elapsed_seconds=time.monotonic()-start)
        line = json.dumps(row, allow_nan=False)
        with (output/"events.jsonl").open("a") as stream:
            stream.write(line+"\n")
        print(line, flush=True)
    with np.load(args.seed_dir/"checkpoint.npz") as saved:
        theta, reference_residual = saved["theta"], saved["residual"]
        left, right = saved["inverse_L"], saved["inverse_R"]
    x, log_mass = theta[:-1], float(theta[-1])
    root = Path(__file__).resolve().parents[1]
    pins = {str(path): digest(path) for path in [args.recipe, args.stellar, args.nist,
        args.seed_dir/"checkpoint.npz", args.observer_inputs, Path(__file__).resolve(),
        *sorted((root/"src/escape").glob("*.py"))]}
    if args.initial_checkpoint:
        pins[str(args.initial_checkpoint)] = digest(args.initial_checkpoint)
    configuration = dict(source_and_input_sha256=pins, geometry=args.geometry,
        halpha_band=dict(wing_extent=args.wing_extent, angle_order=args.angle_order,
                        quadrature_order=args.frequency_order), mass_loss_g_s=float(np.exp(log_mass)),
        blends=args.blends, tolerance=args.tolerance, final_thermal_blend=1.,
        solver=args.solver, reset_inverse=args.reset_inverse,
        scope="Full population, thermal and wind equilibrium at fixed mass loss; no posterior inference.")
    write_json(output/"configuration.json", configuration)
    event("construct", dict(geometry=args.geometry, dimension=x.size))
    model, boundary, recipe = build_model(args.recipe, args.stellar, args.nist, args.geometry,
        args.wing_extent, args.angle_order, args.frequency_order)
    reference_t = jnp.asarray(recipe["temperature_reference_k"])
    @jax.jit
    def native(value, blend):
        return model.evaluate_atmosphere(value, reference_t, np.exp(log_mass),
            boundary["incident_mean_intensity"], hydrogen_line_intensity=boundary["hydrogen_line_intensity"],
            metal_line_intensity=boundary["metal_line_intensity"], overlap_intensity=boundary["overlap_intensity"],
            infrared_intensity=boundary["infrared_intensity"],
            halpha_intensity=boundary.get("halpha_intensity"), halpha_blend=blend)
    last_physical = None
    blend = 0.
    def evaluate(value, kind):
        nonlocal calls, last_physical
        calls += 1
        event("before_call", dict(call=calls, kind=kind, blend=blend))
        residual, last_physical = jax.tree.map(np.asarray, native(jnp.asarray(value), jnp.asarray(blend)))
        check = physical_checks(last_physical, residual)
        finite = np.isfinite(residual).all()
        selected = {key: last_physical[key] for key in ("temperature_k", "n_h_cm3", "bound_cm3",
                    "electron_cm3", "radial_velocity_kms", "thermal_net", "thermal_scale")}
        np.savez_compressed(output/f"call_{calls:03d}_{kind}.npz", theta=np.r_[value, log_mass],
                            residual=residual, blend=blend, **selected)
        event("after_call", dict(call=calls, kind=kind, blend=blend, checks=check,
            infinity=float(np.max(np.abs(residual))) if finite else None,
            l2=float(np.linalg.norm(residual)) if finite else None))
        return residual, check["passed"]
    def save(value, residual, l, r):
        temporary = output/"checkpoint.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, theta=np.r_[value, log_mass], residual=residual,
                inverse_L=l, inverse_R=r, blend=blend)
        temporary.replace(output/"checkpoint.npz")
    def domain(value):
        with np.errstate(over="ignore", invalid="ignore"):
            t = np.asarray(reference_t)*np.exp(value.reshape(model.cells, -1)[:, 0])
        return bool(np.isfinite(value).all() and np.all((t >= 4000.) & (t <= 14000.)))
    baseline, valid = evaluate(x, "baseline")
    parity = float(np.max(np.abs(baseline-reference_residual)))
    event("baseline_parity", dict(maximum_difference=parity))
    if not valid or np.max(np.abs(baseline)) >= 1e-9 or parity >= 1e-10:
        raise RuntimeError("The original complete root was not reproduced at zero H-alpha blend.")
    start_blend = -1.
    initial_checkpoint = output/"checkpoint.npz" if args.resume else args.initial_checkpoint
    if initial_checkpoint:
        with np.load(initial_checkpoint) as saved:
            x = saved["theta"][:-1]
            left, right, start_blend = saved["inverse_L"], saved["inverse_R"], float(saved["blend"])
            if not np.isclose(saved["theta"][-1], log_mass, rtol=0., atol=1e-13):
                raise ValueError("The initial checkpoint must use the same fixed mass loss.")
    if args.reset_inverse:
        left, right = np.empty((len(x), 0)), np.empty((0, len(x)))
    stages = []
    for blend in args.blends:
        if blend < start_blend:
            continue
        f, valid = evaluate(x, "initial")
        if not valid:
            raise RuntimeError("The new radiation operator failed physical guards at its initial state.")
        save(x, f, left, right)
        if args.probe_only:
            stages.append(dict(blend=blend, converged=bool(np.max(np.abs(f)) < args.tolerance)))
            continue
        solver = solve_broyden if args.solver == "broyden" else solve_subspace
        x, f, left, right, result = solver(evaluate, x, f, left, right,
            domain=domain, save=save, event=event, max_steps=args.max_steps, tolerance=args.tolerance)
        stages.append(dict(blend=blend, **result))
        if not result["converged"]:
            write_json(output/"summary.json", dict(converged=False, stages=stages,
                residual_infinity=float(np.max(np.abs(f))), calls=calls))
            raise RuntimeError("Continuation stopped at a saved, unconverged checkpoint.")
    if not args.probe_only:
        fresh, valid = evaluate(x, "fresh_final")
        checks = physical_checks(last_physical, fresh)
        repeated = float(np.max(np.abs(fresh-f)))
        converged = bool(valid and blend == 1. and np.max(np.abs(fresh)) < args.tolerance and repeated < 1e-10)
        if not converged:
            raise RuntimeError("Final complete-root validation failed.")
        if not all(digest(path) == value for path, value in pins.items()):
            raise RuntimeError("Inputs changed during the solve; revalidate in a new output directory.")
        with (output/"complete_physical.pkl").open("wb") as stream:
            pickle.dump(last_physical, stream, protocol=5)
        np.savez_compressed(output/"fresh_final.npz", theta=np.r_[x, log_mass], residual=fresh,
            **{key: last_physical[key] for key in ("temperature_k", "bound_cm3", "electron_cm3", "radial_velocity_kms")})
        with np.load(args.observer_inputs) as saved:
            wn = saved["wavenumber_cm"]
        continuum = jax.jit(lambda: model.continuum_coefficients_at(
            jnp.asarray(last_physical["temperature_k"]), jax.tree.map(jnp.asarray, last_physical),
            jnp.asarray(wn*2.99792458e10)))()
        np.savez_compressed(output/"fixed_inputs.npz", wavenumber_cm=wn,
            continuum_extinction=np.asarray(continuum[0]), continuum_emission_hz=np.asarray(continuum[1]),
            **{key: last_physical[key] for key in ("temperature_k", "bound_cm3", "electron_cm3", "radial_velocity_kms")})
        summary = dict(converged=True, stages=stages, calls=calls, checks=checks,
            residual_infinity=float(np.max(np.abs(fresh))), repeated_residual_difference=repeated,
            baseline_residual_difference=parity, elapsed_seconds=time.monotonic()-start)
    else:
        summary = dict(converged=False, probe_only=True, stages=stages, calls=calls,
                       baseline_residual_difference=parity)
    summary["inputs_unchanged"] = all(digest(path) == value for path, value in pins.items())
    write_json(output/"summary.json", summary)
    event("completed", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--nist", type=Path, default=Path("/tmp/escape-five-cell-root-evidence/nist_row.json"))
    parser.add_argument("--seed-dir", type=Path, default=Path("/tmp/escape-solar-molecular-continuation/current8_e_mdot_refined_trust"))
    parser.add_argument("--observer-inputs", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", choices=["none", "spherical", "slab"], default="spherical")
    parser.add_argument("--blends", type=float, nargs="+", default=[1.])
    parser.add_argument("--wing-extent", type=float, default=30.)
    parser.add_argument("--angle-order", type=int, default=32)
    parser.add_argument("--frequency-order", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--solver", choices=["broyden", "subspace"], default="subspace")
    parser.add_argument("--reset-inverse", action="store_true",
                        help="Start current-direction measurements from the identity map.")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Warm start in a new output directory, retaining the original baseline check.")
    run(parser.parse_args())
