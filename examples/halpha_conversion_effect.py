"""Conditional H-alpha/Ly-beta/O I photon recycling at a frozen atmosphere.

This is a diagnostic, not a replacement SE solver. Each tracked excitation
either mixes within n=3, emits a transported resonance photon, or leaves this
small network. An O I non-resonant cascade, ionization, or excursion to a
different H shell is terminal only for this conditional network. Recombination
and cascade radiation can subsequently return energy in a full atmosphere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from escape.hydrogen_atom import atom_natural_width_hz, make_hydrogen_atom
from escape.hydrogen_rates import LIGHT_SPEED_CGS as C, PLANCK_CGS as H
from escape.line_overlap import LineOverlapGroup
from escape.line_escape_probability import hydrogen_rayleigh_cross_section
from escape.metal_atom import load_metal_atom, electron_collision_coefficients
from halpha_geometry_effect import (load_state, line_coefficients, photon_fates,
                                   photon_integral, geometry_measure)

jax.config.update("jax_enable_x64", True)

TERMINALS = ("halpha_outer", "halpha_core", "halpha_continuum", "uv_outer",
             "uv_core", "uv_continuum", "oxygen_other_radiative",
             "hydrogen_collision_to_lower", "hydrogen_other_shell",
             "hydrogen_ionization")
EVENTS = ("halpha_emission", "lybeta_emission", "oxygen_resonance_emission",
          "halpha_to_uv", "uv_to_halpha", "hydrogen_orbital_mixing",
          "atomic_exit_with_uv_color", "omitted_scattering_encounters")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absorption_chain(transition, terminal, events):
    """Absorbing Markov chain, with expected gross event counts as rewards.

    No renormalization or probability clipping hides an unclosed ledger.
    A perfectly trapped conservative class has no terminal solution and raises.
    """
    q, d, e = map(np.asarray, (transition, terminal, events))
    if not all(np.isfinite(x).all() and np.min(x) >= -1e-13 for x in (q, d, e)):
        raise ValueError("Nonfinite or negative transition, terminal or event weights.")
    error = np.max(np.abs(q.sum(1) + d.sum(1) - 1))
    if error > 1e-9:
        raise ValueError(f"Unclosed one-step probabilities: {error}")
    radius = float(np.max(np.abs(np.linalg.eigvals(q))))
    if radius >= 1 - 1e-13:
        raise ValueError("A conservative trapped class has no terminal solution.")
    fundamental = np.linalg.solve(np.eye(len(q)) - q, np.eye(len(q)))
    fate, counts = fundamental @ d, fundamental @ e
    error = max(error, float(np.max(np.abs(fate.sum(1) - 1))))
    if error > 1e-8 or np.min(fate) < -1e-10:
        raise ValueError(f"Failed terminal conservation: {error}")
    return fate, counts, dict(probability_error=error, spectral_radius=radius,
                             maximum_expected_steps=float(fundamental.sum(1).max()))


def oxygen_metadata():
    atom = load_metal_atom("O I")
    a = np.asarray(atom.spontaneous)
    upper, lower = np.where(a > 0)
    keep = np.isin(upper, [20, 21, 22]) & (lower < 3)
    upper, lower = upper[keep], lower[keep]
    return atom, dict(upper=upper, lower=lower, a=a[upper, lower],
        frequency=np.asarray(atom.radiative_wavenumber)[upper, lower] * C,
        damping=a.sum(1)[upper] / (4 * np.pi),
        energy=H*C*(np.asarray(atom.energy_wavenumber)[upper]
                    - np.asarray(atom.energy_wavenumber)[lower]))


def uv_coefficients(state, physical, quadrature_order=4, wing_extent=1000.):
    """Rebuild the actual seven H/O profiles; use native Ly-beta continuum.

    The continuum is fixed at its stored Ly-beta line-center value over the
    narrow group. Its frequency variation is a declared approximation, probed
    by a factor-of-two continuum test. Spontaneous profiles are normalized on
    the finite CRD grid. Stimulated emission and continuum scattering are
    omitted from tagged-photon transport; their size is reported separately.
    """
    atom = make_hydrogen_atom(15)
    oxygen, meta = oxygen_metadata()
    g = np.asarray(oxygen.weight)
    group = LineOverlapGroup(state["edges_cm"],
        np.r_[atom.line_frequency_hz[1], meta["frequency"]],
        np.r_[atom.einstein_a_s[1], meta["a"]],
        np.r_[atom.statistical_weights[0], g[meta["lower"]]],
        np.r_[atom.statistical_weights[4], g[meta["upper"]]],
        np.r_[1.67262192369e-24, np.full(6, 15.999*1.6605390666e-24)],
        np.r_[atom_natural_width_hz(atom)[1], meta["damping"]],
        transition_energy_erg=np.r_[atom.excitation_energy_erg[4], meta["energy"]],
        quadrature_order=quadrature_order, wing_extent=wing_extent)
    o = physical["metal_levels_cm3"][3]
    lower = np.column_stack((physical["bound_cm3"][:, 0], o[:, meta["lower"]]))
    upper = np.column_stack((physical["bound_cm3"][:, 4], o[:, meta["upper"]]))
    chi, eta, absorption, stimulated, _ = map(np.asarray, group.coefficients(
        physical["temperature_k"], lower, upper))
    center = np.asarray(physical["radiation"]["line_continuum_absorption_cm1"])[:, 1]
    coeff = dict(wavenumber_cm=np.asarray(group.frequency_hz)/C,
                 spectral_weights=np.asarray(group.weight_hz)/C,
                 line_chi=lower[:, :, None]*absorption, line_eta=eta*C,
                 continuum_chi=np.broadcast_to(center[:, None], (8, len(group.frequency_hz))))
    audit = dict(stimulated_to_absorption_max=float(np.max(
        upper[:, :, None]*stimulated/(lower[:, :, None]*absorption))),
        continuum_center_cm1=center.tolist())
    if quadrature_order == 4 and wing_extent == 1000:
        reference = physical["overlap"]
        audit["native_profile_relative_error"] = max(float(np.max(np.abs(
            value/reference[key]-1))) for value, key in
            ((chi, "line_opacity_cm1"), (eta, "line_emission")))
        if audit["native_profile_relative_error"] > 1e-10:
            raise ValueError("Reconstructed H/O profiles do not match the saved state.")
    return coeff, meta, audit


def network(physical, halpha_fates, uv_fates, oxygen, *, lybeta=True,
            atomic_losses=True, oxygen_cascade=True, mixing=True):
    """Build 8 cells x 6 upper states x 2 last-photon colors.

    The color retains whether the last absorbed photon was H-alpha or UV,
    allowing gross forward and reverse band conversion to be counted even
    after repeated orbital mixing and spatial transport.
    """
    cells, states, colors = 8, 6, 2
    size = cells*states*colors
    q, d, e = np.zeros((size, size)), np.zeros((size, len(TERMINALS))), np.zeros((size, len(EVENTS)))
    idx = lambda cell, state, color: (cell*states+state)*colors+color
    rates = physical["radiation"]["rates"]
    collisions = np.asarray(rates.electron_bound_bound_cm3_s)*physical["electron_cm3"][:, None, None]
    rad = np.asarray(rates.radiative_bound_bound_s)
    hydrogen = make_hydrogen_atom(15)
    ha_a = np.asarray(hydrogen.einstein_a_s)[2:5]
    lyb_a = float(hydrogen.einstein_a_s[1]) if lybeta else 0.
    oatom, _ = oxygen_metadata()
    oa = np.asarray(oatom.spontaneous)

    def emit(row, cell, rate, family, branch, fates, color):
        if rate == 0:
            return
        targets = np.arange(3) if family == 0 else np.r_[1, oxygen["upper"]-20+3]
        for receiver in range(cells):
            for absorbed, target in enumerate(targets):
                q[row, idx(receiver, target, family)] += rate*fates["line"][cell, branch, receiver, absorbed]
        offset = 0 if family == 0 else 3
        d[row, offset] += rate*fates["outer"][cell, branch]
        d[row, offset+1] += rate*fates["core"][cell, branch]
        d[row, offset+2] += rate*fates["continuum"][cell, branch].sum()
        e[row, 0 if family == 0 else (1 if branch == 0 else 2)] += rate
        if family != color:
            e[row, 3 if color == 0 else 4] += rate
        if "scattering_encounters" in fates:
            e[row, 7] += rate*fates["scattering_encounters"][cell, branch]

    for cell in range(cells):
        for state in range(states):
            for color in range(colors):
                row = idx(cell, state, color)
                if state < 3:
                    level = state+3
                    emit(row, cell, ha_a[state], 0, state, halpha_fates, color)
                    if state == 1:
                        emit(row, cell, lyb_a, 1, 0, uv_fates, color)
                    if mixing:
                        for target in range(3):
                            if target != state:
                                value = collisions[cell, level, target+3]
                                q[row, idx(cell, target, color)] += value
                                e[row, 5] += value
                    if atomic_losses:
                        d[row, 7] += collisions[cell, level, :3].sum()
                        d[row, 8] += (collisions[cell, level, 6:] + rad[cell, level, 6:]).sum()
                        d[row, 9] += (rates.photoionization_s[cell, level]
                            + rates.electron_ionization_cm3_s[cell, level]*physical["electron_cm3"][cell])
                else:
                    level = state+17
                    for branch in np.flatnonzero(oxygen["upper"] == level):
                        emit(row, cell, oxygen["a"][branch], 1, branch+1, uv_fates, color)
                    if oxygen_cascade:
                        d[row, 6] += oa[level, 3:].sum()
                if color == 1:
                    e[row, 6] += d[row, 7:].sum()
                total = q[row].sum()+d[row].sum()
                q[row] /= total
                d[row] /= total
                e[row] /= total
    fate, counts, audit = absorption_chain(q, d, e)
    return fate.reshape(cells, states, colors, -1), counts.reshape(cells, states, colors, -1), audit


def summarize(physical, ha, uv, oxygen, **flags):
    fate, count, audit = network(physical, ha, uv, oxygen, **flags)
    # Inner four emitting cells set the relative branches of H-alpha photons
    # absorbed in the principal receiving cell at 1.5125 Rp.
    incoming = np.einsum("eb,ebj->j", ha["emitted"][:4], ha["line"][:4, :, 4, :])
    weights = incoming/incoming.sum()
    primary = weights @ fate[4, :3, 0]
    gross = weights @ count[4, :3, 0]
    # Remaining UV color is carried either by a UV/cascade terminal photon,
    # or by an atom leaving this conditional network while still UV-tagged.
    color_error = abs(gross[3]-gross[4]-primary[3:7].sum()-gross[6])
    if color_error > 1e-10:
        raise ValueError(f"Failed band-conversion accounting: {color_error}")
    # Also launch actual inner-cell spontaneous photons before their first flight.
    first = ha["line"][:4]
    source_weights = ha["emitted"][:4]/ha["emitted"][:4].sum()
    emitted_fate = np.einsum("eb,ebrs,rsk->k", source_weights, first, fate[:, :3, 0])
    for key, term in (("outer", 0), ("core", 1)):
        emitted_fate[term] += np.sum(source_weights*ha[key][:4])
    emitted_fate[2] += np.sum(source_weights*ha["continuum"][:4].sum(2))
    return dict(primary_absorbed_halpha=dict(zip(TERMINALS, primary.tolist())),
        primary_gross_events=dict(zip(EVENTS, gross.tolist())),
        primary_initial_branch_weights=weights.tolist(),
        primary_net_halpha_to_uv=float(gross[3]-gross[4]),
        primary_band_conversion_accounting_error=float(color_error),
        inner_emitted_halpha=dict(zip(TERMINALS, emitted_fate.tolist())),
        all_cell_halpha_terminal_probabilities=fate[:, :3, 0].tolist(), **audit)


def transported(state, coeff, angle, physical, **factors):
    opacity = coeff["line_chi"].copy()
    if factors.get("oxygen", True) is False:
        opacity[:, 1:] = 0.
    result = photon_fates(state["edges_cm"], coeff["wavenumber_cm"], opacity,
        coeff["line_eta"], coeff["continuum_chi"]*factors.get("continuum", 1.),
        geometry="sphere", angle_order=angle,
        spectral_weights=coeff.get("spectral_weights"))
    # Expected scattering encounters along the unscattered paths, including
    # all subsequent resonance cycles through the reward matrix. This bounds
    # the probability that this conditional trajectory meets an omitted
    # scattering event; scattering is not relabeled as true absorption.
    wn = coeff["wavenumber_cm"]
    scatter = (physical["electron_cm3"][:, None]*6.6524587321e-25
        + physical["bound_cm3"][:, :1]*np.asarray(hydrogen_rayleigh_cross_section(
            wn/109737.3156816)))
    volume, _ = geometry_measure(state["edges_cm"], "sphere")
    result["scattering_encounters"] = np.sum(volume*photon_integral(wn,
        result["mean_intensity"]*scatter, coeff.get("spectral_weights")), axis=-1)/result["emitted"]
    return result


def run(args):
    state = load_state(args.input_dir, args.recipe)
    with Path(args.physical).open("rb") as stream:
        physical = pickle.load(stream)
    for key in ("temperature_k", "bound_cm3", "radial_velocity_kms"):
        np.testing.assert_array_equal(state[key], physical[key])
    ha_coeff = line_coefficients(state)
    bound = state["bound_cm3"]
    stimulated_ratio = bound[:, [3, 4, 5]]/bound[:, [2, 1, 2]]*np.array([3., 1/3, .6])
    ha_coeff["line_chi"] /= 1-stimulated_ratio[:, :, None]
    uv_coeff, oxygen, uv_audit = uv_coefficients(state, physical)
    outputs, arrays, checks = {}, {}, {}
    print("Transporting positive H-alpha and seven H/O resonance profiles", flush=True)
    ha = transported(state, ha_coeff, args.angle_order, physical)
    uv = transported(state, uv_coeff, args.angle_order, physical)
    outputs["coupled"] = summarize(physical, ha, uv, oxygen)
    outputs["without_lybeta_branch"] = summarize(physical, ha, uv, oxygen, lybeta=False)
    outputs["without_oxygen_cascade"] = summarize(physical, ha, uv, oxygen, oxygen_cascade=False)
    outputs["without_other_atomic_exits"] = summarize(physical, ha, uv, oxygen, atomic_losses=False)
    outputs["without_orbital_mixing"] = summarize(physical, ha, uv, oxygen, mixing=False)
    for label, factors in (("without_oxygen_opacity", dict(oxygen=False)),
                           ("without_continuum", dict(continuum=0.)),
                           ("double_continuum", dict(continuum=2.))):
        print(label, flush=True)
        h = transported(state, ha_coeff, args.angle_order, physical, continuum=factors.get("continuum", 1.))
        u = transported(state, uv_coeff, args.angle_order, physical, **factors)
        outputs[label] = summarize(physical, h, u, oxygen)
    wn = state["wavenumber_cm"]
    expanded = np.linspace(wn[0]-(wn[-1]-wn[0])/2, wn[-1]+(wn[-1]-wn[0])/2,
                           2*len(wn)-1)
    wide_ha = line_coefficients(state, expanded)
    wide_ha["line_chi"] /= 1-stimulated_ratio[:, :, None]
    print("Doubling H-alpha bandwidth", flush=True)
    wide = transported(state, wide_ha, args.angle_order, physical)
    outputs["halpha_double_window"] = summarize(physical, wide, uv, oxygen)
    outputs["halpha_double_window_without_lybeta"] = summarize(physical, wide, uv, oxygen, lybeta=False)
    checks["halpha_wing_primary_outer_relative_change"] = abs(
        outputs["halpha_double_window"]["primary_absorbed_halpha"]["halpha_outer"]
        /outputs["coupled"]["primary_absorbed_halpha"]["halpha_outer"]-1)
    base_difference = (outputs["without_lybeta_branch"]["primary_absorbed_halpha"]["halpha_outer"]
                       - outputs["coupled"]["primary_absorbed_halpha"]["halpha_outer"])
    wide_difference = (outputs["halpha_double_window_without_lybeta"]["primary_absorbed_halpha"]["halpha_outer"]
                       - outputs["halpha_double_window"]["primary_absorbed_halpha"]["halpha_outer"])
    checks["halpha_wing_lybeta_effect_absolute_change"] = abs(wide_difference-base_difference)
    checks["halpha_wing_lybeta_effect_relative_change"] = abs(wide_difference/base_difference-1)
    # Joint angle/frequency/wing refinement, with the same physical state.
    midpoint = np.sort(np.r_[state["wavenumber_cm"],
        (state["wavenumber_cm"][:-1]+state["wavenumber_cm"][1:])/2])
    fine_ha = line_coefficients(state, midpoint)
    fine_ha["line_chi"] /= 1-stimulated_ratio[:, :, None]
    fine_uv, _, _ = uv_coefficients(state, physical, quadrature_order=8, wing_extent=3000)
    print("Refining angle, frequency and UV wings", flush=True)
    h = transported(state, fine_ha, 2*args.angle_order, physical)
    u = transported(state, fine_uv, 2*args.angle_order, physical)
    outputs["refined"] = summarize(physical, h, u, oxygen)
    primary = lambda item: np.array(list(item["primary_absorbed_halpha"].values()))
    delta = primary(outputs["refined"])-primary(outputs["coupled"])
    checks["primary_terminal_max_absolute_refinement"] = float(np.max(np.abs(delta)))
    checks["primary_halpha_outer_relative_refinement"] = float(abs(delta[0])/primary(outputs["refined"])[0])
    for label, fates in (("halpha", ha), ("uv", uv), ("wide_halpha", wide),
                          ("refined_halpha", h), ("refined_uv", u)):
        checks[label+"_photon_accounting_error"] = fates["accounting_error"]
        for key in ("line", "continuum", "core", "outer", "emitted", "scattering_encounters"):
            arrays[label+"_"+key] = fates[key]
    rates = physical["radiation"]["rates"].radiative_bound_bound_s
    hydrogen = make_hydrogen_atom(15)
    lower, upper = np.array([2, 1, 2]), np.array([3, 4, 5])
    oatom, _ = oxygen_metadata()
    o_collision = np.stack([np.asarray(electron_collision_coefficients(oatom, t))
                           for t in state["temperature_k"]])*physical["electron_cm3"][:, None, None]
    comparison = dict(
        native_halpha_net_emission_cm3_s=(bound[:, upper]*rates[:, upper, lower]
            - bound[:, lower]*rates[:, lower, upper]).sum(1).tolist(),
        native_lybeta_net_emission_cm3_s=(bound[:, 4]*rates[:, 4, 0]-bound[:, 0]*rates[:, 0, 4]).tolist(),
        raw_lybeta_spontaneous_cm3_s=(bound[:, 4]*float(hydrogen.einstein_a_s[1])).tolist(),
        native_oxygen_group_net_excitation_cm3_s=physical["overlap"]["net_excitation_cm3_s"][:, 1:].sum(1).tolist(),
        native_halpha_destruction_probability=physical["radiation"]["line_operators"].destruction_probability[:, 2:5].tolist(),
        omitted_oxygen_electron_collision_over_radiative_max=float(np.max(
            o_collision[:, 20:23].sum(2)/np.asarray(oatom.spontaneous)[20:23].sum(1))),
        halpha_stimulated_ratio_by_cell=stimulated_ratio.max(1).tolist(), **uv_audit)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays.update(edges_cm=state["edges_cm"], radius_rp=state["radius_rp"])
    np.savez_compressed(out/"arrays.npz", **arrays)
    source = Path(__file__).resolve().parents[1]
    review = dict(status="Executed conditional frozen-state recycling network; no SE or fit prediction.",
        state_identity="Exactly equal T, H populations and radial velocity to current PR12/13 state; all rates from that same complete_physical.pkl. No older E8 physical rates used.",
        input_sha256={**state["input_sha256"], str(args.physical):digest(args.physical)},
        source_sha256={str(p.relative_to(source)):digest(p) for p in (
            Path(__file__).resolve(), source/"examples/halpha_geometry_effect.py",
            source/"src/escape/line_overlap.py", source/"src/escape/spherical_radiation.py",
            source/"src/escape/hydrogen_atom.py", source/"src/escape/metal_atom.py",
            source/"src/escape/line_escape_probability.py", source/"src/escape/hydrogen_rates.py",
            source/"src/escape/data/metal_atoms.npz", source/"tests/test_halpha_conversion_effect.py")},
        raw_arrays=dict(path=str(out/"arrays.npz"), sha256=digest(out/"arrays.npz")),
        terminal_names=TERMINALS, event_names=EVENTS, angle_order=args.angle_order,
        principal_receiver=dict(cell=4, radius_rp=float(state["radius_rp"][4])),
        cases=outputs, checks=checks, native_comparison=comparison,
        passed=checks["primary_halpha_outer_relative_refinement"] < .01
            and checks["primary_terminal_max_absolute_refinement"] < .01
            and checks["halpha_wing_primary_outer_relative_change"] < .01
            and max(v for k,v in checks.items() if "accounting" in k) < 1e-9,
        limitations=[
            "Spherical, static, complete redistribution within homogeneous frozen cells; repeated absorptions are redistributed uniformly within the receiving cell.",
            "No stimulated photon multiplication or continuum electron/Rayleigh scattering; positive lower-population absorption is used.",
            "Omitted scattering is bounded by expected encounters along the no-scattering trajectories; this is not a scattering redistribution solve.",
            "Ly-beta continuum is its actual native line-center value, constant across the group; doubling and removing it are sensitivity probes, not convergence proofs.",
            "O I resonance re-emission is followed, but non-resonant O I cascade photons leave this conditional network. O I collisions and photoionization are omitted.",
            "H collisions to lower states, shell excursions and ionization leave the tracked cycle. Subsequent recombination/cascade return is not solved.",
            "The no-Ly-beta-branch control deletes a real A coefficient and is an isolation experiment, not a proposed physical atom.",
            "Tagged input photons are a linear conditional perturbation. Adding these returns on top of all saved spontaneous emission would double count cycles.",
            "Native closure already contains continuum destruction, orbital mixing and H/O overlap. These are not new missing species or new independent sinks.",
            "No conversion probability is interpreted as a source-function multiplier or observational fit improvement."])
    review_path = Path(args.review)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=review["passed"], checks=checks, primary=outputs["coupled"]["primary_absorbed_halpha"])), flush=True)
    if not review["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fine"))
    parser.add_argument("--recipe", type=Path, default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--physical", type=Path, default=Path("/tmp/escape-solar-molecular-continuation/current8_e_mdot_refined_trust/complete_physical.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/escape-halpha-conversion-effect"))
    parser.add_argument("--review", type=Path, default=Path("mceg/evidence/halpha_conversion_effect_20261126/review.json"))
    parser.add_argument("--angle-order", type=int, default=16)
    run(parser.parse_args())
