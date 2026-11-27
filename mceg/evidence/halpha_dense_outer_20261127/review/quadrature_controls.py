"""Reuse H-alpha group-only quadrature controls at one CPU-verified dense state.

No population, temperature, density, velocity or atmospheric equilibrium is
changed. Partial H/free-free receiving heat is not a full thermal residual.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import exojax
import jax
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/"examples"))
from halpha_equilibrium import build_model,digest,write_json
from halpha_equilibrium_controls import diagnose,compare

jax.config.update("jax_enable_x64",True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical",type=Path,required=True)
    parser.add_argument("--cpu-review",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--review",type=Path,required=True)
    parser.add_argument("--recipe",type=Path,default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    parser.add_argument("--stellar",type=Path,default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--nist",type=Path,default=Path("/tmp/escape-five-cell-root-evidence/nist_row.json"))
    args=parser.parse_args()
    if jax.default_backend()!="cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu.")
    cpu_path=args.cpu_review or args.physical.with_name("review.json")
    cpu=json.loads(cpu_path.read_text())
    if not (cpu["passed"] and cpu["inputs_unchanged"]):
        raise ValueError("A successful immutable CPU review is required.")
    matches=[row for row in cpu["cases"] if row.get("full_physical",{}).get("sha256")==digest(args.physical)]
    if len(matches)!=1 or not matches[0]["passed"] or matches[0]["residual_infinity"]>=1e-9:
        raise ValueError("The full physical artifact must identify one exact accepted CPU case.")
    case=matches[0]
    physical=pickle.loads(args.physical.read_bytes())
    if not bool(physical["valid"]) or float(physical["halpha"]["blend"])!=1.:
        raise ValueError("A valid full H-alpha replacement state is required.")
    # Driver/predictor files do not enter this fixed-state group calculation.
    # Check the actual group/model dependencies, including the pinned external tree.
    original=cpu["source_and_input_sha256"]
    required=[args.recipe,args.stellar,args.nist,ROOT/"examples/halpha_equilibrium.py",
              *sorted((ROOT/"src/escape").glob("*.py"))]
    for path in required:
        if original.get(str(path.resolve()))!=digest(path):
            raise ValueError(f"A model dependency differs from the accepted CPU run: {path}")
    dependency_path=ROOT/"mceg/evidence/halpha_equilibrium_20261126/dependency_snapshot.json"
    dependency=json.loads(dependency_path.read_text())
    external=dependency["runtime_exojax"]
    if not Path(exojax.__file__).resolve().is_relative_to(Path(external["path"]).resolve()):
        raise ValueError("Use the archived ExoJAX tree.")
    pins={str(Path(external["path"])/key):value for key,value in external["python_source_sha256"].items()}
    pins.update({str(ROOT/key):value for key,value in dependency["atomic_tables"].items()})
    pins.update({str(path.resolve()):digest(path) for path in [*required,Path(__file__),cpu_path,args.physical,
        dependency_path,ROOT/"examples/halpha_equilibrium_controls.py"]})
    if any(digest(path)!=value for path,value in pins.items()):
        raise ValueError("A source/input pin does not match before evaluation.")
    settings=dict(primary=(32,8,30.),angle=(64,8,30.),frequency=(32,16,30.),wings=(32,8,60.))
    cases,elapsed={},{}
    for label,(angle,frequency,wing) in settings.items():
        start=time.monotonic()
        print(f"Group-only {label}: angle={angle}, frequency={frequency}, wing={wing}",flush=True)
        model,boundary,_=build_model(args.recipe,args.stellar,args.nist,"spherical",wing,angle,frequency)
        if label=="primary":
            np.testing.assert_array_equal(model.halpha_group.frequency_hz,physical["halpha"]["frequency_hz"])
        cases[label]=diagnose(model,boundary,physical)
        elapsed[label]=time.monotonic()-start
        print(f"Finished {label} in {elapsed[label]:.1f} s",flush=True)
    primary=cases["primary"]
    reproduced={name:float(np.max(abs(primary[key]/np.asarray(physical["halpha"][name])-1)))
        for name,key in (("pump_up_s","upward_s"),("radiative_down_s","downward_s"))}
    reproduced["net_scaled_by_effective_transition"]=float(np.max(abs(primary["net_excitation_cm3_s"]-
        np.asarray(physical["halpha"]["net_excitation_cm3_s"]))/primary["effective_transition_scale_cm3_s"]))
    required_mask=~np.asarray(physical["atomic_lte_mask"])
    comparisons={label:compare(primary,value) for label,value in cases.items() if label!="primary"}
    for label,value in cases.items():
        if label=="primary":
            continue
        changes=comparisons[label]
        changes["nonLTE_up_down_max_relative_change"]=max(float(np.max(abs(value[key][required_mask]/primary[key][required_mask]-1)))
            for key in ("upward_s","downward_s"))
        terms=("redistribution_heat_erg_cm3_s","line_hydrogen_kinetic_heat_erg_cm3_s",
               "background_hydrogen_kinetic_heat_erg_cm3_s","line_free_free_heat_erg_cm3_s","background_free_free_heat_erg_cm3_s")
        delta=sum(value[key]-primary[key] for key in terms)
        changes["partial_heat_delta_over_saved_thermal_scale"]=(delta/np.asarray(physical["thermal_scale"])).tolist()
    tests=dict(primary_rates_reproduced=max(reproduced.values())<1e-10,
        transfer_valid=bool(all(row["converged"] for row in cases.values())),
        line_energy_identity=bool(max(np.max(abs(row["line_energy_defect_over_emission"])) for row in cases.values())<1e-8),
        inputs_unchanged=all(digest(path)==value for path,value in pins.items()))
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/"arrays.npz"
    np.savez_compressed(raw,**{f"{label}_{key}":value for label,row in cases.items() for key,value in row.items()})
    scalar=("upward_s","downward_s","net_excitation_cm3_s","n3_net_excitation_cm3_s",
            "redistribution_heat_erg_cm3_s","line_absorbed_power_erg_cm3_s","line_energy_defect_over_emission",
            "line_hydrogen_kinetic_heat_erg_cm3_s","background_hydrogen_kinetic_heat_erg_cm3_s",
            "line_free_free_heat_erg_cm3_s","background_free_free_heat_erg_cm3_s")
    review=dict(scope=__doc__,factor=case["factor"],CPU_case=case["label"],checks=tests,
        all_implementation_checks_passed=all(tests.values()),primary_reproduction_errors=reproduced,
        source_and_input_sha256=pins,settings=settings,elapsed_seconds=elapsed,comparisons=comparisons,
        cases={label:{key:row[key].tolist() for key in scalar} for label,row in cases.items()},
        raw_arrays=dict(path=str(raw),sha256=digest(raw)),
        limitations=["Implementation checks test parity, conservation and immutable inputs; they do not declare refined-transfer or observable adequacy.",
            "Up/down rates are self-eliminated effective operators. Net differences and their cancellation scales are reported separately.",
            "Common-frequency mean intensity uses interpolation and is a display comparison, not the integrated-rate convergence criterion.",
            "Wing enlargement also expands the background correction, including geometry, exact stellar samples and continuum-bin interpolation. Line and background heating remain separate.",
            "Only H and free-free receiving heat is exposed. Other receiving species and the full SE/thermal residual are not re-evaluated here.",
            "The density ramp and eight-cell physical stratification are not radially refined by these controls.",
            "Material changes require a newly solved atmosphere before refined observables may be claimed."])
    args.review.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.review,review)
    print(json.dumps(dict(checks=tests,nonLTE_rate_changes={k:v["nonLTE_up_down_max_relative_change"] for k,v in comparisons.items()}),indent=2))
    assert all(tests.values())


if __name__=="__main__":
    main()
