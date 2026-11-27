"""Recover n=2/3 receiving rates and audit their frozen-state excitation budget.

Local five-state sensitivity calculations hold the external population and
radiation baths fixed. They are not charge, thermal, transport, or wind roots.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.special import voigt_profile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples"))
from escape.hydrogen_atom import make_hydrogen_atom, atom_cross_sections, atom_milne_population, atom_collision_rates
from escape.hydrogen_mixing import mixing_bound_bound_cm3_s
from escape.hydrogen_rates import PLANCK_CGS as H, BOLTZMANN_CGS as K
from halpha_conversion_effect import uv_coefficients
from halpha_geometry_effect import load_state
from cloudy_halpha import NIST_ROW

jax.config.update("jax_enable_x64", True)
FOLDER = Path(__file__).resolve().parent
LEVELS = np.arange(1, 6)
SHELLS = dict(ground=np.array([0]), n2=np.arange(1, 3), n3=np.arange(3, 6), higher=np.arange(6, 18))
CELLS = (4, 5, 6)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(physical, state, atom):
    temperature = jnp.asarray(physical["temperature_k"])

    @jax.jit
    def reduce(frequency, flux):
        cross = atom_cross_sections(atom, frequency)[1:6]
        inverse = jax.vmap(lambda t: atom_milne_population(atom, t, frequency)[1:6])(temperature)
        return flux @ cross.T, jnp.einsum("cf,lf,clf->cl", flux, cross, inverse)

    def packet(frequency, flux):
        keep = frequency >= float(np.min(atom.ionization_energy_erg[1:6])) / H
        frequency, flux = frequency[keep], flux[:, keep]
        photo, rr = np.zeros((8, 5)), np.zeros((8, 5))
        for start in range(0, len(frequency), 4096):
            f, row = frequency[start:start+4096], flux[:, start:start+4096]
            fp, weights = np.full(4096, f[-1]), np.zeros((8, 4096))
            fp[:len(f)], weights[:, :len(f)] = f, row
            p, r = map(np.asarray, reduce(fp, weights))
            photo += p
            rr += r
        return photo, rr

    metal = packet(np.asarray(physical["metal"]["destroyed_line_frequency_hz"]),
                   np.asarray(physical["metal"]["ots_photon_flux_cm2_s"]))
    uv, _, audit = uv_coefficients(state, physical)
    overlap = packet(uv["wavenumber_cm"] * 2.99792458e10,
                     np.asarray(physical["overlap"]["additional_photon_flux_cm2_s"]))
    return dict(metal_photo=metal[0], metal_rr=metal[1], overlap_photo=overlap[0], overlap_rr=overlap[1]), audit


def production_loss(bound, transition, photo, rr, ion, three, ne, proton, indices):
    incoming = (bound[:, None] * transition)[:, indices].sum()
    outgoing = (bound[indices] * transition[indices].sum(axis=1)).sum()
    gain = incoming + ne * proton * (rr[indices] + ne * three[indices]).sum()
    loss = outgoing + (bound[indices] * (photo[indices] + ne * ion[indices])).sum()
    return float(gain), float(loss)


def shell_budget(bound, radiation, collision, photo_parts, rr_parts, ion, three, ne, proton, name):
    indices = LEVELS if name == "n2_n3" else SHELLS[name]
    terms = {}
    for other, targets in SHELLS.items():
        if np.all(np.isin(targets, indices)):
            continue  # Internal l mixing cancels in the shell sum.
        for label, matrix in (("radiative", radiation), ("collisional", collision)):
            incoming = (bound[targets, None] * matrix[np.ix_(targets, indices)]).sum()
            outgoing = (bound[indices, None] * matrix[np.ix_(indices, targets)]).sum()
            terms[label + "_" + other] = dict(in_cm3_s=float(incoming), out_cm3_s=float(outgoing), net_cm3_s=float(incoming-outgoing))
    for label in photo_parts:
        incoming = ne * proton * rr_parts[label][indices].sum()
        outgoing = (bound[indices] * photo_parts[label][indices]).sum()
        terms["bound_free_" + label] = dict(in_cm3_s=float(incoming), out_cm3_s=float(outgoing), net_cm3_s=float(incoming-outgoing))
    incoming = proton * ne**2 * three[indices].sum()
    outgoing = (bound[indices] * ne * ion[indices]).sum()
    terms["collisional_ionization_three_body"] = dict(in_cm3_s=float(incoming), out_cm3_s=float(outgoing), net_cm3_s=float(incoming-outgoing))
    gain, loss = [sum(row[key] for row in terms.values()) for key in ("in_cm3_s", "out_cm3_s")]
    return dict(channels=terms, total_external_production_cm3_s=gain, total_external_loss_cm3_s=loss,
                scaled_net=(gain-loss)/(gain+loss), turnover_seconds=float(bound[indices].sum()/loss))


def main():
    path = Path("/tmp/escape-solar-molecular-continuation/current8_e_mdot_refined_trust/complete_physical.pkl")
    free_path = ROOT / "mceg/evidence/halpha_causes_20261127/sources/free_shape.json"
    physical, state = pickle.loads(path.read_bytes()), load_state()
    for key in ("temperature_k", "bound_cm3", "radial_velocity_kms"):
        np.testing.assert_array_equal(physical[key], state[key])
    files = [Path(__file__), path, free_path, *map(Path, state["input_sha256"]),
        *[ROOT / "src/escape" / name for name in ("hydrogen_atom.py", "hydrogen_rates.py", "hydrogen_bound_free.py", "hydrogen_collisions.py", "hydrogen_mixing.py", "hydrogen_escape_probability.py", "hydrogen_nlte.py", "line_escape_probability.py", "line_overlap.py", "metal_atom.py", "parker.py", "solar.py")],
        *[ROOT / "examples" / name for name in ("halpha_conversion_effect.py", "halpha_geometry_effect.py", "cloudy_halpha.py")]]
    pins = {str(p): digest(p) for p in files}
    atom = make_hydrogen_atom(15, resolved_nmax=3)
    receiving, uv_audit = recover(physical, state, make_hydrogen_atom(3))
    rates = physical["radiation"]["rates"]
    rad, base_collision, base_photo, base_rr, ion, three = [np.asarray(value) for value in rates]
    photo_parts = dict(base_continuum=base_photo-np.asarray(physical["radiation"]["photoionization_ots_s"]),
                       hydrogen_ots=np.asarray(physical["radiation"]["photoionization_ots_s"]))
    rr_parts = dict(base_continuum=base_rr-np.asarray(physical["radiation"]["recombination_ots_cm3_s"]),
                    hydrogen_ots=np.asarray(physical["radiation"]["recombination_ots_cm3_s"]))
    for label in ("metal", "overlap"):
        p, r = np.zeros_like(base_photo), np.zeros_like(base_rr)
        p[:, 1:6], r[:, 1:6] = receiving[label+"_photo"], receiving[label+"_rr"]
        photo_parts[label], rr_parts[label] = p, r
    photo, rr = sum(photo_parts.values()), sum(rr_parts.values())
    bound, ne, proton, temperature = [np.asarray(physical[key]) for key in ("bound_cm3", "electron_cm3", "proton_cm3", "temperature_k")]
    # Match the native electron plus screened proton mixing operator before varying density.
    @jax.jit
    def collisions(t, e, p):
        q = atom_collision_rates(atom, t, low_temperature_extrapolation="cloudy", shell_collision_model="anderson_lebedev")[0]
        q = q.at[1, 2].set(0.).at[2, 1].set(0.)
        return q + mixing_bound_bound_cm3_s(atom, t, e) * p/e
    rebuilt = np.stack([np.asarray(collisions(temperature[i], ne[i], proton[i])) for i in CELLS])
    collision_error = float(np.max(abs(rebuilt-base_collision[list(CELLS)])/np.maximum(abs(base_collision[list(CELLS)]), 1e-100)))
    assert collision_error < 1e-10
    upper, lower = np.asarray(atom.line_upper[2:5]), np.asarray(atom.line_lower[2:5])
    g, energy = np.asarray(atom.statistical_weights), np.asarray(atom.excitation_energy_erg)
    line_energy = energy[upper]-energy[lower]
    free = json.loads(free_path.read_text())["free"]["original"]["at_physical_centers"]
    center_profile = voigt_profile(NIST_ROW["wav"][0]-state["branch_centers"][None], state["branch_sigma"], state["branch_gamma"][None])

    def population_metrics(cell, populations):
        ratio = populations[upper]/populations[lower]*g[lower]/g[upper]
        original = bound[cell, upper]/bound[cell, lower]*g[lower]/g[upper]
        source_factor = ratio/(1-ratio) / (original/(1-original))
        alpha_factor = (populations[lower]*g[upper]/g[lower]-populations[upper])/(bound[cell,lower]*g[upper]/g[lower]-bound[cell,upper])
        alpha = state["branch_alpha"][cell]*center_profile[cell]
        eta = state["branch_eta"][cell]*center_profile[cell]
        opacity_factor = np.sum(alpha*alpha_factor)/np.sum(alpha)
        source_center_factor = np.sum(eta*populations[upper]/bound[cell,upper])/np.sum(eta)/opacity_factor
        return dict(n2_cm3=float(populations[1:3].sum()), n3_cm3=float(populations[3:6].sum()),
            n2_factor=float(populations[1:3].sum()/bound[cell, 1:3].sum()),
            n3_n2=float(populations[3:6].sum()/populations[1:3].sum()),
            branch_source_ratio=source_factor.tolist(), rest_center_source_ratio=float(source_center_factor),
            static_center_opacity_ratio=float(opacity_factor),
            branch_excitation_temperature_k=(line_energy/K/np.log(1/ratio)).tolist())

    def local(cell, density=1., lya_pump=1., photo_fraction=1., halpha_scale=1., lya_escape_scale=1., details=False):
        e, p = ne[cell]*density, proton[cell]*density
        external = bound[cell]*density
        radiation = rad[cell].copy()
        # Scale the effective upward pump and its Einstein-related stimulated reverse only.
        change = rad[cell, 0, 2]*(lya_pump-1)
        radiation[0, 2] += change
        radiation[2, 0] += change*g[0]/g[2]
        radiation[2, 0] += (lya_escape_scale-1)*float(atom.einstein_a_s[0])*float(physical["radiation"]["line_operators"].escape_probability[cell,0])
        radiation[upper, lower] *= halpha_scale
        radiation[lower, upper] *= halpha_scale
        collision = e*np.asarray(collisions(temperature[cell], e, p))
        transition = radiation + collision
        pi = photo[cell].copy()
        pi[1:6] *= photo_fraction
        losses = transition.sum(1)+pi+e*ion[cell]
        matrix = np.diag(losses)-transition.T
        outside = np.r_[0, np.arange(6, 18)]
        rhs = p*e*(rr[cell]+e*three[cell]) + transition[outside].T @ external[outside]
        selected = matrix[np.ix_(LEVELS, LEVELS)]
        solution = np.linalg.solve(selected/losses[LEVELS, None], rhs[LEVELS]/losses[LEVELS])
        external[LEVELS] = solution
        residual = selected@solution-rhs[LEVELS]
        result = population_metrics(cell, external)
        result.update(density_factor=density, effective_lya_pump_factor=lya_pump, photoionization_sink_fraction=photo_fraction,
            halpha_common_up_down_scale=halpha_scale, lya_escape_term_scale=lya_escape_scale,
            scaled_linear_balance_error=float(np.max(abs(residual)/(abs(selected)@solution+rhs[LEVELS]))),
            selected_population_fraction_of_scaled_nH=float(solution.sum()/(density*physical["n_h_cm3"][cell])))
        base_return = float(p*e*base_rr[cell].sum())
        n2_ionization = float(external[1:3]@pi[1:3])
        selected_proton_net = float(external[LEVELS]@(pi+e*ion[cell])[LEVELS]-p*e*(rr[cell]+e*three[cell])[LEVELS].sum())
        original_selected_net = float(bound[cell,LEVELS]@(photo[cell]+ne[cell]*ion[cell])[LEVELS]-proton[cell]*ne[cell]*(rr[cell]+ne[cell]*three[cell])[LEVELS].sum())
        result["ionization_budget_warning"] = dict(n2_photoionization_cm3_s=n2_ionization,
            total_base_H_radiative_recombination_cm3_s=base_return,
            total_base_H_three_body_cm3_s=float(p*e**2*three[cell].sum()),
            known_n2_n3_receiving_recombination_cm3_s=float(p*e*(rr[cell,1:6]-base_rr[cell,1:6]).sum()),
            change_in_n2_n3_net_proton_production_from_saved_cm3_s=selected_proton_net-original_selected_net,
            n2_photo_to_total_base_RR=n2_ionization/base_return)
        assert np.all(solution > 0) and result["scaled_linear_balance_error"] < 1e-12
        if details:
            pieces = {}
            for label, targets in (("ground", np.array([0])), ("higher", np.arange(6,18))):
                for kind, operator in (("radiative", radiation), ("collisional", collision)):
                    pieces[kind+"_"+label] = operator[targets].T @ external[targets]
            for label, coefficient in rr_parts.items():
                pieces["recombination_"+label] = p*e*coefficient[cell]
            pieces["three_body"] = p*e**2*three[cell]
            forcing = {}
            for label, vector in pieces.items():
                response = np.linalg.solve(selected/losses[LEVELS,None], vector[LEVELS]/losses[LEVELS])
                forcing[label] = dict(injected_into_n2_n3_cm3_s=float(vector[LEVELS].sum()),
                    maintained_n2_cm3=float(response[:2].sum()), maintained_n3_cm3=float(response[2:].sum()),
                    fraction_of_n2=float(response[:2].sum()/solution[:2].sum()))
            np.testing.assert_allclose(sum(x["maintained_n2_cm3"] for x in forcing.values()), solution[:2].sum(), rtol=1e-12)
            result["fixed_bath_forcing_decomposition"] = forcing
            result["n2_budget"] = shell_budget(external,radiation,collision,
                {k:v[cell] for k,v in photo_parts.items()},{k:v[cell] for k,v in rr_parts.items()},ion[cell],three[cell],e,p,"n2")
            result["n2_n3_external_budget"] = shell_budget(external,radiation,collision,
                {k:v[cell] for k,v in photo_parts.items()},{k:v[cell] for k,v in rr_parts.items()},ion[cell],three[cell],e,p,"n2_n3")
        return result

    records, sensitivities = [], {}
    for cell in CELLS:
        coll = ne[cell]*base_collision[cell]
        transition = rad[cell]+coll
        total_in, total_out = production_loss(bound[cell], transition, photo[cell], rr[cell], ion[cell], three[cell], ne[cell], proton[cell], LEVELS)
        before_in, before_out = production_loss(bound[cell], transition, base_photo[cell], base_rr[cell], ion[cell], three[cell], ne[cell], proton[cell], LEVELS)
        levels = {}
        for level, name in zip(LEVELS, ("2s", "2p", "3s", "3p", "3d")):
            gain, loss = production_loss(bound[cell], transition, photo[cell], rr[cell], ion[cell], three[cell], ne[cell], proton[cell], np.array([level]))
            levels[name] = dict(population_cm3=float(bound[cell, level]), production_cm3_s=gain, loss_cm3_s=loss, scaled_net=(gain-loss)/(gain+loss),
                base_photoionization_s=float(base_photo[cell, level]), complete_photoionization_s=float(photo[cell, level]))
        boltz = g*np.exp(-(energy-energy[1])/(K*temperature[cell]))
        operators = physical["radiation"]["line_operators"]
        record = dict(cell=cell, radius_rp=float(state["radius_rp"][cell]), temperature_k=float(temperature[cell]), electron_cm3=float(ne[cell]), proton_cm3=float(proton[cell]),
            ground_cm3=float(bound[cell, 0]), populations=population_metrics(cell,bound[cell]),
            native_ground_only_chemical_terms=dict(
                ct_recombination_cm3_s=float(proton[cell]*(physical["metal"]["hydrogen_ct_recombination_s"][cell]+physical["helium"]["hydrogen_ct_recombination_s"][cell])),
                ct_ionization_cm3_s=float(bound[cell,0]*(physical["metal"]["hydrogen_ct_ionization_s"][cell]+physical["helium"]["hydrogen_ct_ionization_s"][cell])),
                molecular_proton_destruction_cm3_s=float(physical["molecule_balance"].proton_destruction_cm3_s[cell]),
                molecular_proton_production_cm3_s=float(physical["molecule_balance"].proton_production_cm3_s[cell])),
            LTE_n3_n2=float(boltz[3:6].sum()/boltz[1:3].sum()),
            required_n2_factor=float(free["n2_cm3"][cell]/bound[cell, 1:3].sum()),
            free_source_ratio_to_physical=float(free["source_ratio_to_physical"][cell]), levels=levels,
            base_five_state_scaled_net=(before_in-before_out)/(before_in+before_out),
            corrected_five_state_scaled_net=(total_in-total_out)/(total_in+total_out),
            n2=shell_budget(bound[cell],rad[cell],coll,{k:v[cell] for k,v in photo_parts.items()},{k:v[cell] for k,v in rr_parts.items()},ion[cell],three[cell],ne[cell],proton[cell],"n2"),
            n3=shell_budget(bound[cell],rad[cell],coll,{k:v[cell] for k,v in photo_parts.items()},{k:v[cell] for k,v in rr_parts.items()},ion[cell],three[cell],ne[cell],proton[cell],"n3"),
            n2_n3_external=shell_budget(bound[cell],rad[cell],coll,{k:v[cell] for k,v in photo_parts.items()},{k:v[cell] for k,v in rr_parts.items()},ion[cell],three[cell],ne[cell],proton[cell],"n2_n3"),
            lya=dict(effective_up_s=float(rad[cell,0,2]), effective_down_s=float(rad[cell,2,0]),
                raw_A_s=float(atom.einstein_a_s[0]), escape_probability=float(operators.escape_probability[cell,0]),
                destruction_probability=float(operators.destruction_probability[cell,0]), electron_escape_probability=float(operators.electron_escape_probability[cell,0]),
                iron_extra_up_s=float(physical["iron_lyalpha_exchange"]["hydrogen_upward_s"][cell]),
                iron_extra_down_s=float(physical["iron_lyalpha_exchange"]["hydrogen_downward_s"][cell])),
            halpha=dict(effective_down_s=rad[cell,upper,lower].tolist(), collisional_down_s=coll[upper,lower].tolist(),
                raw_A_s=np.asarray(atom.einstein_a_s[2:5]).tolist(), collision_to_effective_radiative_ratio=(coll[upper,lower]/rad[cell,upper,lower]).tolist(),
                naive_fixed_radiation_critical_density_factor=(rad[cell,upper,lower]/coll[upper,lower]).tolist(),
                population_weighted_all_n3_to_n2_radiative_s=float((bound[cell,3:6,None]*rad[cell,3:6,1:3]).sum()/bound[cell,3:6].sum()),
                population_weighted_all_n3_to_n2_collisional_s=float((bound[cell,3:6,None]*coll[3:6,1:3]).sum()/bound[cell,3:6].sum())))
        records.append(record)
        baseline = local(cell,details=True)
        assert abs(baseline["n2_factor"]-1) < 1e-8
        target = record["required_n2_factor"]
        density_required = brentq(lambda logd: local(cell,density=np.exp(logd))["n2_factor"]-target, 0., np.log(1000.))
        pump_required = brentq(lambda logp: local(cell,lya_pump=np.exp(logp))["n2_factor"]-target, 0., np.log(1e8))
        sensitivities[str(cell)] = dict(baseline=baseline,
            density=[local(cell,density=f) for f in (3.,10.,30.,100.,300.,1000.)],
            effective_lya_pump=[local(cell,lya_pump=f) for f in (3.,10.,100.)],
            suppressed_photoionization_bound=[local(cell,photo_fraction=f) for f in (.1,0.)],
            suppressed_lya_escape=[local(cell,lya_escape_scale=f) for f in (.1,0.)],
            density_matching_free_n2=local(cell,density=np.exp(density_required),details=True),
            density_target_halpha_escape_sensitivity=[local(cell,density=np.exp(density_required),halpha_scale=f) for f in (.25,.1)],
            pump_matching_free_n2=local(cell,lya_pump=np.exp(pump_required)))
    check = max(abs(v["scaled_net"]) for r in records for v in r["levels"].values())
    assert check < 1e-8
    arrays=FOLDER/"receiving_and_rates.npz"
    np.savez_compressed(arrays, **receiving, complete_photo_n2_n3=photo[:,1:6], complete_rr_n2_n3=rr[:,1:6])
    result=dict(scope=__doc__, source_input_sha256=pins, unchanged_source_inputs=all(digest(p)==v for p,v in pins.items()),
        checks=dict(native_state_identity=True, collision_rebuild_max_relative_error=collision_error,
            receiving_corrected_max_level_balance=check, uv_profile_audit=uv_audit),
        cells=records, fixed_bath_sensitivities=sensitivities, raw_array_sha256=digest(arrays),
        interpretation=[
            "Bound-bound radiation is the actual effective SE operator: A times escape/destruction/electron-escape plus shielded stimulated pumping, with overlap/Fe replacements already present. Its gross terms are not the microscopic resonant absorption/emission counts or an independent local mean intensity.",
            "Shell budgets cancel internal orbital mixing. Radiative inflow from higher shells includes cascades and effective transfer, not solely recombination-origin atoms.",
            "Receiving metal OTS and signed overlap packets are restored for n=2,3. IR photons are below both thresholds. Native molecular/charge-exchange terms are added only to ground; this five-level audit does not omit their direct n=2/3 source terms because no such native terms exist.",
            "Density probes scale ne, proton, ground and higher-level bath populations together and recompute screened mixing. Temperature, escape/destruction, incident pumping, higher-level kinetics, charge, thermal balance and flow are not solved again.",
            "The Ly-alpha probe changes effective upward and Einstein-related stimulated reverse together, not the effective escape sink. It is a required-pump diagnostic without a supplied photon budget or transport realization.",
            "The Ly-alpha escape removal probe reduces only the A*escape term, retaining destruction, electron escape, pumping and Fe exchange. Trapped local self-emission has already been eliminated in these effective equations; it is not an omitted additive pump.",
            "The H-alpha opacity-feedback sensitivity scales each of the three effective upward/downward pairs together at the density target. It is not a recomputed escape probability or nonlocal transfer; real shielding, escape and intercell pumping need not scale together.",
            "The maintained-population decomposition is the linear inverse of the fixed five-level loss matrix applied to each external forcing. It separates internal H-alpha cycles from independent bath injection, but does not trace the earlier origin of fixed higher-shell populations.",
            "Photoionization suppression is an optimistic sink-removal bound with fixed recombination and all other baths, not a spectral attenuation model. Actual shielding also changes recombination, heating, ground ionization, and line transfer.",
            "The total-base-H recombination comparison includes all 18 stored levels but omits receiving corrections outside n=2/3, including IR high-shell stimulated recombination. Native ground-only charge exchange and molecular terms are reported separately at baseline. This is a supply-scale warning, not a reconstructed complete ground/proton balance; all n=2/3 local terms are recovered.",
            "For the fixed-density pump-only probe, the change in n2/n3 net continuum proton production is complete for those levels. Unchanged higher/ground receiving terms cancel in this difference. It would therefore be an unbalanced extra proton source if the other atomic and chemical baths were held fixed, even without recovering their original rates."])
    assert result["unchanged_source_inputs"]
    (FOLDER/"review.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(checks=result["checks"], cells=[{k:r[k] for k in ("cell","populations","LTE_n3_n2","required_n2_factor")} for r in records], sensitivities=sensitivities),indent=2))


if __name__ == "__main__":
    main()
