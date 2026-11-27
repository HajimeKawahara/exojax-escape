"""Continue a prescribed, force-supported outer density enhancement.

This spherical radial precursor solves the full native population/radiation
and thermal residual, but prescribes the extra momentum support. It is not a
magnetic equilibrium or a partially covering sector model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from exojax.utils.constants import G, MJ, kB

from escape.parker import PROTON_MASS_CGS
from escape.hydrogen_forward import PLANET_RADIUS_CM
from halpha_equilibrium import (build_model, digest, physical_checks,
                                solve_subspace, write_json)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)

KEYS = ("temperature_k", "n_h_cm3", "bound_cm3", "electron_cm3",
        "radial_velocity_kms", "thermal_net", "thermal_scale",
        "thermal_advection", "thermal_expansion", "molecule_cm3", "helium_cm3",
        "metal_ion_cm3", "metal_levels_cm3", "proton_cm3", "element_density_cm3",
        "pressure_dyn_cm2", "mean_molecular_weight", "valid",
        "reference_domain_valid", "atomic_lte_mask", "thermal_balance_required")


def enhancement(radius_cm, log_factor, start_rp=1.3, end_rp=1.725):
    """C1 log-density ramp, with exact derivative for the flow energy terms."""
    x = jnp.clip((radius_cm / PLANET_RADIUS_CM-start_rp)/(end_rp-start_rp), 0., 1.)
    weight = x*x*(3.-2.*x)
    derivative = 6.*x*(1.-x)/((end_rp-start_rp)*PLANET_RADIUS_CM)
    return jnp.exp(log_factor*weight), log_factor*derivative


def evaluate_dense(model, value, reference_t, mass_loss, boundary, log_factor):
    """Trace the native equations with a changed wind closure only.

    The temporary instance hook is restored during tracing; the compiled
    function takes log_factor as an ordinary dynamic scalar. No production
    module, reaction, residual normalization or physical guard is changed.
    """
    reference_wind = model._wind

    def supported_wind(state, temperature, mass_loss_g_s):
        wind, info = reference_wind(state, temperature, mass_loss_g_s)
        q, gradient_q = enhancement(model.radius_cm, log_factor)
        velocity = wind.velocity_kms/q
        mu = model.unpack(state, jnp.ones(model.cells))["mean_molecular_weight"]
        sound2 = kB*temperature/(mu*PROTON_MASS_CGS)
        v2, ref2 = (velocity*1e5)**2, (wind.velocity_kms*1e5)**2
        support = ((v2-ref2)*info["log_velocity_gradient_cm1"]
                   +(sound2-v2)*gradient_q)
        reference_defect = (ref2*info["log_velocity_gradient_cm1"]
            +sound2*(-2/model.radius_cm-info["log_velocity_gradient_cm1"])
            +info["sound_squared_gradient_cm_s2"]
            +G*MJ*model.planet_mass_mj/model.radius_cm**2
            -model.tidal_coefficient_s2*model.radius_cm)
        modified = wind._replace(velocity_kms=velocity, density_g_cm3=wind.density_g_cm3*q)
        return modified, {**info,
            "log_velocity_gradient_cm1": info["log_velocity_gradient_cm1"]-gradient_q,
            "reference_velocity_kms": wind.velocity_kms,
            "density_enhancement": q,
            "log_enhancement_gradient_cm1": gradient_q,
            "additional_support_acceleration_cm_s2": support,
            "reference_momentum_defect_cm_s2": reference_defect,
            "required_support_acceleration_cm_s2": support+reference_defect}

    with patch.object(model, "_wind", supported_wind):
        return model.evaluate_atmosphere(value, reference_t, mass_loss,
            boundary["incident_mean_intensity"],
            hydrogen_line_intensity=boundary["hydrogen_line_intensity"],
            metal_line_intensity=boundary["metal_line_intensity"],
            overlap_intensity=boundary["overlap_intensity"],
            infrared_intensity=boundary["infrared_intensity"],
            halpha_intensity=boundary["halpha_intensity"], halpha_blend=1.)


def compact_physical(p):
    selected = {key: p[key] for key in KEYS}
    selected["wind_info"] = {key: p["wind_info"][key] for key in
        ("valid", "converged", "reference_velocity_kms", "density_enhancement",
         "log_enhancement_gradient_cm1", "required_support_acceleration_cm_s2")}
    selected["wind_info"].update({key:p["wind_info"][key] for key in
        ("additional_support_acceleration_cm_s2", "reference_momentum_defect_cm_s2")})
    return selected


def checks(p, residual, radius, mass_loss):
    result = physical_checks(p, residual)
    for key in ("wind_valid", "wind_converged"):
        result["flags"]["reference_"+key] = result["flags"].pop(key)
    flux = (4*np.pi*np.asarray(radius)**2*np.asarray(p["n_h_cm3"])
            *np.asarray(p["mass_per_h"])*np.asarray(p["radial_velocity_kms"])*1e5)
    result["mass_flux_relative_error"] = float(np.max(np.abs(flux/mass_loss-1)))
    result["flags"]["mass_continuity"] = result["mass_flux_relative_error"] < 1e-12
    result["flags"]["finite_prescribed_force"] = bool(np.isfinite(
        p["wind_info"]["required_support_acceleration_cm_s2"]).all())
    thermal = (p["thermal_net"]+p["thermal_advection"]+p["thermal_expansion"])
    scale = p["thermal_scale"]+np.abs(p["thermal_advection"])+np.abs(p["thermal_expansion"])
    result["required_thermal_scaled_max"] = float(np.max(np.abs(
        (thermal/scale)[np.asarray(p["thermal_balance_required"])])))
    result["passed"] = all(result["flags"].values())
    result["momentum_scope"] = "Extra support is required, not physically derived; wind flags refer to the reference Parker solution."
    return result


def run(args):
    if any(not np.isfinite(x) or x<1 for x in args.factors) or any(
            b<=a for a,b in zip(args.factors,args.factors[1:])):
        raise ValueError("Use increasing finite factors >= 1.")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/"configuration.json").exists():
        raise ValueError("Use a new output directory; previous attempts are immutable.")
    start, calls = time.monotonic(), 0

    def event(kind, details):
        row = dict(details, event=kind, elapsed_seconds=time.monotonic()-start)
        line = json.dumps(row, allow_nan=False)
        with (out/"events.jsonl").open("a") as stream:
            stream.write(line+"\n")
        print(line, flush=True)

    root = Path(__file__).resolve().parents[1]
    inputs = [args.recipe, args.stellar, args.nist, args.baseline, args.observer_inputs,
              Path(__file__).resolve(), root/"examples/halpha_equilibrium.py",
              *sorted((root/"src/escape").glob("*.py"))]
    if args.initial_checkpoint:
        inputs.append(args.initial_checkpoint)
    pins = {str(p): digest(p) for p in inputs}
    with np.load(args.baseline) as saved:
        theta, baseline_f = saved["theta"].copy(), saved["residual"].copy()
        left, right = saved["inverse_L"].copy(), saved["inverse_R"].copy()
    log_mass, x = float(theta[-1]), theta[:-1]
    mass_loss = np.exp(log_mass)
    model, boundary, recipe = build_model(args.recipe, args.stellar, args.nist, "spherical")
    reference_t = jnp.asarray(recipe["temperature_reference_k"])
    config = dict(source_and_input_sha256=pins, factors=args.factors,
        radius_ramp_rp=[1.3, 1.725], ramp="C1 cubic smoothstep in log density",
        mass_loss_g_s=float(mass_loss), tolerance=args.tolerance,
        backend=jax.default_backend(), x64=jax.config.x64_enabled,
        halpha="Spherical static CRD with thermal/natural broadening; 32 angles, 8 frequency nodes, 30 wings.",
        scope="Full native population/radiation and thermal root with prescribed spherical momentum support; no MHD or angular covering solution.")
    write_json(out/"configuration.json", config)
    event("construct", dict(dimension=len(x), backend=jax.default_backend()))

    @jax.jit
    def native(value, log_density):
        f, p = evaluate_dense(model, value, reference_t, mass_loss, boundary, log_density)
        selected = compact_physical(p)
        selected["mass_per_h"] = model.mass_per_h
        return f, selected

    log_density, current, last_physical = 0., out, None

    def evaluate(value, kind):
        nonlocal calls, last_physical
        calls += 1
        f, last_physical = jax.tree.map(np.asarray, native(jnp.asarray(value), jnp.asarray(log_density)))
        guard = checks(last_physical, f, model.radius_cm, mass_loss)
        finite = np.isfinite(f).all()
        event("evaluation", dict(call=calls, kind=kind, factor=float(np.exp(log_density)),
            passed=guard["passed"], infinity=float(np.max(np.abs(f))) if finite else None,
            l2=float(np.linalg.norm(f)) if finite else None))
        return f, guard["passed"]

    def save(value, f, l, r):
        with (current/"checkpoint.tmp").open("wb") as stream:
            np.savez_compressed(stream, theta=np.r_[value, log_mass], residual=f,
                inverse_L=l, inverse_R=r, log_density_factor=log_density)
        (current/"checkpoint.tmp").replace(current/"checkpoint.npz")

    def domain(value):
        with np.errstate(over="ignore", invalid="ignore"):
            t = np.asarray(reference_t)*np.exp(value.reshape(model.cells, -1)[:, 0])
        return bool(np.isfinite(value).all() and np.all((t >= 4000.) & (t <= 14000.)))

    f, valid = evaluate(x, "baseline_parity")
    parity = float(np.max(np.abs(f-baseline_f)))
    if not valid or np.max(np.abs(f)) >= args.tolerance or parity >= 1e-10:
        raise RuntimeError("The D=1 spherical baseline was not reproduced.")
    event("baseline_parity", dict(maximum_difference=parity))
    initial_density = 0.
    if args.initial_checkpoint:
        with np.load(args.initial_checkpoint) as saved:
            assert abs(float(saved["theta"][-1])-log_mass) < 1e-13
            x = saved["theta"][:-1].copy()
            left, right = saved["inverse_L"].copy(), saved["inverse_R"].copy()
            initial_density = float(saved["log_density_factor"])
    stages = []
    with np.load(args.observer_inputs) as saved:
        wn = saved["wavenumber_cm"].copy()
    continuum = jax.jit(lambda t, p: model.continuum_coefficients_at(t, p, jnp.asarray(wn*2.99792458e10)))
    for factor in args.factors:
        log_density = np.log(factor)
        if log_density < initial_density-1e-12:
            raise ValueError("Continuation factors must not precede the initial checkpoint.")
        current = out/f"D{factor:g}"
        current.mkdir()
        f, valid = evaluate(x, "initial")
        save(x, f, left, right)
        if not valid or not domain(x):
            result = dict(converged=False, reason="initial_guard")
        else:
            x, f, left, right, result = solve_subspace(evaluate, x, f, left, right,
                domain=domain, save=save, event=event, max_steps=args.max_steps,
                tolerance=args.tolerance, max_basis=args.max_basis)
        if not result["converged"]:
            summary = dict(result, factor=factor, residual_infinity=float(np.max(np.abs(f))),
                           accepted_for_observation=False)
            write_json(current/"summary.json", summary)
            stages.append(summary)
            break
        fresh, valid = evaluate(x, "fresh_final")
        repeated = float(np.max(np.abs(fresh-f)))
        guard = checks(last_physical, fresh, model.radius_cm, mass_loss)
        if not valid or np.max(np.abs(fresh)) >= args.tolerance or repeated >= 1e-10:
            raise RuntimeError("Fresh complete residual or physical guards failed.")
        if not all(digest(p)==h for p,h in pins.items()):
            raise RuntimeError("Source/input changed during this attempt.")
        p = last_physical
        # Export coefficients only after the complete native residual passes.
        extinction, emission = jax.tree.map(np.asarray, continuum(jnp.asarray(p["temperature_k"]),
            jax.tree.map(jnp.asarray,{k:p[k] for k in ("bound_cm3", "proton_cm3", "electron_cm3",
                "molecule_cm3", "helium_cm3", "metal_ion_cm3", "metal_levels_cm3", "atomic_lte_mask")})))
        np.savez_compressed(current/"fixed_inputs.npz", wavenumber_cm=wn,
            continuum_extinction=extinction, continuum_emission_hz=emission,
            **{k:p[k] for k in ("temperature_k", "bound_cm3", "electron_cm3", "radial_velocity_kms")})
        np.savez_compressed(current/"physical_summary.npz",
            **{k:v for k,v in p.items() if k not in ("wind_info", "metal_levels_cm3")},
            **{f"metal_levels__{k}":v for k,v in p["metal_levels_cm3"].items()},
            **{f"wind__{k}":v for k,v in p["wind_info"].items()})
        save(x, fresh, left, right)
        summary = dict(result, factor=factor, residual_infinity=float(np.max(np.abs(fresh))),
            repeated_residual_difference=repeated, checks=guard,
            fixed_inputs_sha256=digest(current/"fixed_inputs.npz"),
            checkpoint_sha256=digest(current/"checkpoint.npz"),
            physical_summary_sha256=digest(current/"physical_summary.npz"),
            accepted_for_observation=True, independent_cpu_check_pending=jax.default_backend()!="cpu",
            physical_realization="unknown: prescribed supporting force")
        write_json(current/"summary.json", summary)
        stages.append(summary)
        event("stage_complete", dict(factor=factor, infinity=summary["residual_infinity"]))
    write_json(out/"summary.json", dict(stages=stages, calls=calls,
        baseline_parity=parity, elapsed_seconds=time.monotonic()-start,
        inputs_unchanged=all(digest(p)==h for p,h in pins.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--nist", type=Path, default=Path("/tmp/escape-five-cell-root-evidence/nist_row.json"))
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/escape-halpha-equilibrium-gpu-cublas/checkpoint.npz"))
    parser.add_argument("--observer-inputs", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz"))
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", type=float, nargs="+", default=[1., 1.2, 1.5, 2., 3., 5., 10., 20.])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-basis", type=int, default=24)
    run(parser.parse_args())
