"""Compare stored full-root H-alpha feedback, without a new atmosphere solve."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from escape.hydrogen_atom import make_hydrogen_atom

H, C = 6.62607015e-27, 2.99792458e10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def characterize(path, cell=5):
    with Path(path).open("rb") as stream:
        p = pickle.load(stream)
    h, rates = p["halpha"], p["hydrogen_complete_rates"]
    if float(h["blend"]) != 1. or not h["converged"]:
        raise ValueError("Use a complete spherical H-alpha state.")
    atom = make_hydrogen_atom(15, resolved_nmax=3)
    lower, upper, a, center = (np.asarray(x)[2:5] for x in
        (atom.line_lower, atom.line_upper, atom.einstein_a_s, atom.line_frequency_hz))
    nu, edges = np.asarray(h["frequency_hz"]), np.asarray(h["frequency_edges_hz"])
    order = len(nu)//(len(edges)-1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    weight = (np.diff(edges)[:, None]/2*weights).ravel()
    reconstructed = ((edges[:-1]+edges[1:])[:, None]/2
                     +np.diff(edges)[:, None]/2*nodes).ravel()
    np.testing.assert_allclose(nu, reconstructed, rtol=1e-14, atol=0)
    bound, ne, proton = p["bound_cm3"][cell], p["electron_cm3"][cell], p["proton_cm3"][cell]
    eta, chi = h["line_emission"][cell], h["line_opacity_cm1"][cell]
    profile = eta*4*np.pi/(H*nu[None, :]*bound[upper, None]*a[:, None])
    normalization = np.sum(profile*weight, axis=1)
    np.testing.assert_allclose(normalization, 1., rtol=0, atol=1e-12)
    pump, down, induced = (np.asarray(h[key])[cell] for key in
                           ("pump_up_s", "radiative_down_s", "stimulated_down_s"))
    net = bound[lower]*pump-bound[upper]*down
    np.testing.assert_allclose(net, h["net_excitation_cm3_s"][cell], rtol=1e-11, atol=1e-10)
    nearest = np.argmin(abs(nu[None, :]-center[:, None]), axis=1)
    bb = np.asarray(rates.radiative_bound_bound_s)[cell]
    collision = ne*np.asarray(rates.electron_bound_bound_cm3_s)[cell]
    n2, n3 = bound[1:3].sum(), bound[3:6].sum()
    return dict(temperature_k=float(p["temperature_k"][cell]),
        n_h_cm3=float(p["n_h_cm3"][cell]), electron_cm3=float(ne), proton_cm3=float(proton),
        n2_cm3=float(n2), n3_cm3=float(n3), n3_over_n2=float(n3/n2),
        branch_profile_mean_J_hz=np.sum(profile*weight*h["transfer"].mean_intensity[cell], axis=1).tolist(),
        branch_center_source_per_wavenumber=(C*eta[np.arange(3), nearest]/chi[np.arange(3), nearest]).tolist(),
        effective_pump_up_s=pump.tolist(), effective_radiative_down_s=down.tolist(),
        effective_stimulated_down_s=induced.tolist(),
        effective_spontaneous_over_A=((down-induced)/a).tolist(),
        radiative_Halpha_up_cm3_s=float(np.sum(bound[lower]*pump)),
        radiative_Halpha_down_cm3_s=float(np.sum(bound[upper]*down)),
        radiative_Halpha_net_n3_cm3_s=float(net.sum()),
        collisional_Halpha_up_cm3_s=float(np.sum(bound[1:3, None]*collision[1:3, 3:6])),
        collisional_Halpha_down_cm3_s=float(np.sum(bound[3:6, None]*collision[3:6, 1:3])),
        n2_radiative_from_ground_cm3_s=float(bound[0]*bb[0, 1:3].sum()),
        n2_radiative_from_higher_than_n3_cm3_s=float(np.sum(bound[6:, None]*bb[6:, 1:3])),
        n2_photoionization_cm3_s=float(np.sum(bound[1:3]*rates.photoionization_s[cell, 1:3])),
        n2_radiative_recombination_cm3_s=float(ne*proton*np.sum(rates.radiative_recombination_cm3_s[cell, 1:3])),
        n2_three_body_recombination_cm3_s=float(ne*ne*proton*np.sum(rates.three_body_recombination_cm6_s[cell, 1:3])),
        n3_photoionization_cm3_s=float(np.sum(bound[3:6]*rates.photoionization_s[cell, 3:6])),
        n3_radiative_recombination_cm3_s=float(ne*proton*np.sum(rates.radiative_recombination_cm3_s[cell, 3:6])),
        n3_three_body_recombination_cm3_s=float(ne*ne*proton*np.sum(rates.three_body_recombination_cm6_s[cell, 3:6])),
        profile_normalization_max_error=float(np.max(abs(normalization-1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/escape-halpha-equilibrium-final-cpu/complete_physical.pkl"))
    parser.add_argument("--cpu-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.cpu_review.read_text())
    final = receipt["cases"][-1]
    if not receipt["passed"] or not receipt["inputs_unchanged"] or not final["passed"]:
        raise ValueError("The selected CPU state must pass all full-root checks.")
    candidate = Path(final["full_physical"]["path"])
    if digest(candidate) != final["full_physical"]["sha256"]:
        raise ValueError("Full physical snapshot does not match the CPU receipt.")
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).resolve(), args.baseline, candidate, args.cpu_review,
             root/"src/escape/hydrogen_atom.py", root/"src/escape/line_overlap.py"]
    pins = {str(p): digest(p) for p in paths}
    baseline, result = characterize(args.baseline), characterize(candidate)
    ratios = {k: (np.asarray(v)/np.asarray(baseline[k])).tolist() for k, v in result.items()
              if k != "profile_normalization_max_error" and np.all(np.asarray(baseline[k]) != 0)}
    output = dict(scope=__doc__, factor=final["factor"], cell=5, radius_rp=1.9375,
        baseline=baseline, candidate=result, ratio_to_D1=ratios,
        source_and_input_sha256=pins,
        inputs_unchanged=all(digest(p)==h for p, h in pins.items()),
        limitations=[
            "Both points solve the complete native equations. This comparison does not isolate a unique cause: T, charge, all populations, transport and flow change together.",
            "Profile-mean J is the full transported intensity, including local trapped emission. Effective pump/down rates eliminate the same branch's local source and cannot be inferred directly from full J alone.",
            "(down-stimulated)/A is an own-branch source-elimination factor including overlapping-branch exchange; it is not a global escape or irreversible photon-loss probability.",
            "Source values sample the native group's nearest branch-center nodes; they are per wavenumber and precede observer amplitude profiling.",
            "Bound-free quantities use hydrogen_complete_rates, including receiving packets. Reported source/sink entries are selected gross channels, not normalized causal fractions; H-alpha cycles largely cancel.",
            "The earlier five-level bath fixed T, external populations, ion/electron densities and radiation operators. Its density response is not a prediction of this coupled atmosphere."])
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(factor=final["factor"], n2_ratio=ratios["n2_cm3"],
                          source_ratios=ratios["branch_center_source_per_wavenumber"])))


if __name__ == "__main__":
    main()
