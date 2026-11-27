"""Independent local continuity, pressure-force and thermal checks for q(r).

This evaluates no new atmosphere. Finite differences use local reference
extensions matching the saved native value and first spatial derivatives;
they check the rescaling identities, not global wind interpolation accuracy.
"""
import hashlib
import json
from pathlib import Path
import pickle

import jax
import numpy as np
from exojax.utils.constants import G, MJ, kB
from escape.atmosphere_wind import thermal_flow_terms
from escape.parker import PROTON_MASS_CGS

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RP = 1.9 * 7.1492e9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enhancement(radius, factor):
    z = np.clip((radius / RP - 1.3) / .425, 0., 1.)
    shape = z*z*(3 - 2*z)
    derivative = 6*z*(1-z) / (.425*RP)
    return np.exp(np.log(factor)*shape), np.log(factor)*derivative


def main():
    folder = Path("/tmp/escape-halpha-equilibrium-final-cpu")
    recipe_path = Path("/tmp/escape-corrected-observer-full-f/recipe.json")
    files = [Path(__file__), recipe_path, folder/"complete_physical.pkl",
             folder/"checkpoint.npz", folder/"summary.json",
             ROOT/"src/escape/atmosphere_wind.py", ROOT/"src/escape/nonisothermal.py",
             ROOT/"src/escape/solar.py"]
    pins = {str(path): digest(path) for path in files}
    state = pickle.loads((folder/"complete_physical.pkl").read_bytes())
    summary = json.loads((folder/"summary.json").read_text())
    assert summary["converged"] and summary["checks"]["passed"] and summary["inputs_unchanged"]
    recipe = json.loads(recipe_path.read_text())
    edge = np.asarray(recipe["edges_rp"])
    radius = (edge[1:] + edge[:-1]) / 2 * RP
    temperature, mu = (np.asarray(state[key]) for key in ("temperature_k", "mean_molecular_weight"))
    v0 = np.asarray(state["radial_velocity_kms"]) * 1e5
    # Recover rho from the saved gas pressure with Solar's own Boltzmann constant.
    rho0 = np.asarray(state["pressure_dyn_cm2"]) * mu * PROTON_MASS_CGS / (1.380649e-16*temperature)
    a2 = kB*temperature/(mu*PROTON_MASS_CGS)
    grad_v = np.asarray(state["wind_info"]["log_velocity_gradient_cm1"])
    grad_a2 = np.asarray(state["wind_info"]["sound_squared_gradient_cm_s2"])
    gravity = G*MJ*2.17/radius**2
    tide = 3*(2*np.pi/(1.48*86400))**2*radius
    reference_force = (v0*v0-a2)*grad_v+grad_a2-2*a2/radius+gravity-tide
    with np.load(folder/"checkpoint.npz") as checkpoint:
        mass_loss = float(np.exp(checkpoint["theta"][-1]))
    reference_adv = -1.5*rho0*v0*grad_a2
    reference_exp = -rho0*v0*a2*(2/radius+grad_v)
    flux_error = float(np.max(abs(4*np.pi*radius**2*rho0*v0/mass_loss-1)))
    native_error = max(float(np.max(abs(expected/np.asarray(state[key])-1))) for expected,key in
                       ((reference_adv,"thermal_advection"),(reference_exp,"thermal_expansion")))
    rows = []
    for factor in (1., 2., 5., 10., 20., 20.19):
        q, grad_q = enhancement(radius, factor)
        density, velocity = rho0*q, v0/q
        modified_gradient = grad_v-grad_q
        force_increment = (velocity**2-v0**2)*grad_v+(a2-velocity**2)*grad_q
        force_total = force_increment+reference_force
        adv = -1.5*density*velocity*grad_a2
        expansion = -density*velocity*a2*(2/radius+modified_gradient)
        expected_expansion_delta = rho0*v0*a2*grad_q
        production = thermal_flow_terms(radius,density,temperature,mu,velocity,
            log_velocity_gradient_cm1=modified_gradient,
            sound_squared_gradient_cm_s2=grad_a2)
        scale = np.maximum.reduce([np.ones_like(radius), abs(force_total), gravity, abs(tide), abs(a2*grad_v)])
        errors = []
        for step_rp in (1e-5, 5e-6):
            step = step_rp*RP
            def local(delta):
                r = radius+delta
                vr = v0*np.exp(grad_v*delta)
                rho = rho0*(radius/r)**2*np.exp(-grad_v*delta)
                q_offset,_ = enhancement(r,factor)
                rho *= q_offset
                return vr/q_offset, rho*(a2+grad_a2*delta)
            vp,pp = local(step)
            vm,pm = local(-step)
            numerical = velocity*(vp-vm)/(2*step)+(pp-pm)/(2*step)/density+gravity-tide
            errors.append(float(np.max(abs(numerical-force_total)/scale)))
        denom = np.maximum(abs(reference_exp),1e-30)
        rows.append(dict(factor=factor,q=q.tolist(),dlogq_dr_cm1=grad_q.tolist(),
            velocity_kms=(velocity/1e5).tolist(),density_g_cm3=density.tolist(),
            reference_momentum_defect_cm_s2=reference_force.tolist(),
            added_support_acceleration_cm_s2=force_increment.tolist(),
            total_support_acceleration_cm_s2=force_total.tolist(),
            advection_erg_cm3_s=adv.tolist(),expansion_erg_cm3_s=expansion.tolist(),
            expansion_delta_erg_cm3_s=expected_expansion_delta.tolist(),
            support_power_erg_cm3_s=(density*velocity*force_increment).tolist(),
            finite_difference_scaled_errors=errors,
            continuity_relative_error=float(np.max(abs(density*velocity/(rho0*v0)-1))),
            advection_relative_error=float(np.max(abs(adv/reference_adv-1))),
            expansion_identity_scaled_error=float(np.max(abs(expansion-reference_exp-expected_expansion_delta)/denom)),
            native_flow_terms_scaled_error=max(float(np.max(abs(np.asarray(production.advection)-adv)/abs(reference_adv))),
                float(np.max(abs(np.asarray(production.expansion)-expansion)/denom))),
            D1_exact_identity=bool(factor != 1 or (np.array_equal(q,np.ones_like(q))
                and np.array_equal(density,rho0) and np.array_equal(velocity,v0)
                and np.array_equal(modified_gradient,grad_v) and np.all(force_increment == 0)))))
    checks = dict(saved_reference_accepted=bool(summary["converged"] and summary["checks"]["passed"]),
        mass_flux_recovery=flux_error<1e-12, saved_thermal_terms_reproduced=native_error<1e-12,
        D1_exact_identity=all(row["D1_exact_identity"] for row in rows),
        continuity_and_thermal_identities=all(max(row[k] for k in
            ("continuity_relative_error","advection_relative_error","expansion_identity_scaled_error","native_flow_terms_scaled_error"))<1e-12 for row in rows),
        independent_pressure_momentum_derivative=all(row["finite_difference_scaled_errors"][-1]<5e-7 for row in rows),
        source_inputs_unchanged=all(digest(path)==value for path,value in pins.items()))
    output = dict(scope=__doc__.strip(),checks=checks,all_passed=all(checks.values()),
        reference_radius_rp=(radius/RP).tolist(),mass_flux_relative_error=flux_error,
        saved_thermal_relative_error=native_error,rows=rows,source_and_input_sha256=pins,
        limitations=["Reference sampled momentum defects are retained; the added force vanishes at D=1, not the native discretization error.",
            "support_power is mechanical work density, not an added heat term. A mechanism must specify dissipation/energy exchange.",
            "The native flow module uses ExoJAX kB=1.38064852e-16; reported Solar gas pressure uses 1.380649e-16. Each is used consistently here.",
            "No populations, radiation, thermal root, geometry, momentum-supported configuration or observable is solved."])
    (HERE/"flow_controls.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(checks=checks,mass_flux_relative_error=flux_error,
        reference_momentum_defect_cm_s2=reference_force.tolist(),
        worst_pressure_derivative_error=max(row['finite_difference_scaled_errors'][-1] for row in rows)),indent=2))
    assert all(checks.values())


if __name__ == "__main__":
    main()
