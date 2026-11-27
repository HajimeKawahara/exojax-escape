"""Describe qualified dense candidates without solving transfer or fitting data.

All line quantities are local rest-frame thermal/natural-profile diagnostics.
Static radial tau=1 is not an observer formation radius or a flux prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import voigt_profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0,str(ROOT/"examples"))
from halpha_geometry_fit import state_line_coefficients, RP_CM
from cloudy_halpha import NIST_ROW
from escape.line_escape_probability import hydrogen_stark_escape


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def columns(edges, value):
    return {str(cut): float(value @ (np.maximum(0.,edges[1:]-np.maximum(cut,edges[:-1]))*RP_CM))
            for cut in (1.3,1.725,2.15)}


def tau_one(edges, chi):
    above = np.r_[np.cumsum((chi*np.diff(edges)*RP_CM)[::-1])[::-1],0.]
    take = np.flatnonzero((above[:-1]>=1)&(above[1:]<1))
    if not len(take):
        return None
    i = take[0]
    radius = edges[i+1]-(1-above[i+1])/(chi[i]*RP_CM)
    width = np.maximum(0.,edges[1:]-np.maximum(radius,edges[:-1]))*RP_CM
    np.testing.assert_allclose(chi @ width,1.,rtol=0,atol=1e-12)
    return float(radius)


def load(folder, pins):
    paths = [folder/name for name in ("summary.json","physical_summary.npz","fixed_inputs.npz")]
    pins.update({str(p):digest(p) for p in paths})
    receipt = json.loads(paths[0].read_text())
    if not receipt["converged"] or not receipt["accepted_for_observation"] or not receipt["checks"]["passed"]:
        raise ValueError(f"Unqualified candidate: {folder}")
    for path in paths[1:]:
        if receipt[path.stem+"_sha256"] != digest(path):
            raise ValueError(f"State/receipt hash mismatch: {path}")
    physical, state = dict(np.load(paths[1])), dict(np.load(paths[2]))
    for key in ("temperature_k","bound_cm3","radial_velocity_kms","electron_cm3"):
        np.testing.assert_array_equal(physical[key],state[key])
    return physical,state,receipt


def characterize(physical,state,edges):
    b = state_line_coefficients(state)
    phi = voigt_profile(NIST_ROW["wav"][0]-b["centers"][None],b["sigma"],b["gamma"][None])
    chi,eta = b["alpha"]*phi,b["eta"]*phi
    if np.any(chi<=0) or np.any(~np.isfinite(chi)):
        raise ValueError("Static positive-opacity diagnostics require positive finite branch extinction")
    n2,n3 = (state["bound_cm3"][:,take].sum(1) for take in (slice(1,3),slice(3,6)))
    total = chi.sum(1)
    at_centers=voigt_profile(b["centers"][None,:,None]-b["centers"][None,None,:],
                            b["sigma"][:,None,:],b["gamma"][None,None,:])
    center_chi=np.sum(b["alpha"][:,None,:]*at_centers,axis=-1)
    depth=center_chi*np.diff(edges)[:,None]*RP_CM
    front=np.cumsum(depth[::-1],axis=0)[::-1]-depth
    # An isolated-line escape prescription, not a pressure-broadened profile.
    stark=np.asarray(hydrogen_stark_escape(3,2,state["electron_cm3"][:,None],
        state["temperature_k"][:,None],front+.5*depth))
    result = dict(n2_cm3=n2,n3_cm3=n3,n3_over_n2=n3/n2,
        nominal_center_branch_extinction_cm1=chi,
        branch_source_per_wavenumber=b["eta"]/b["alpha"],
        nominal_center_total_extinction_cm1=total,
        nominal_center_total_source_per_wavenumber=eta.sum(1)/total,
        branch_center_wavenumber_cm=b["centers"],
        diagnostic_isolated_Stark_escape_increment=stark,
        n2_column_above_radius_cm2=columns(edges,n2),
        static_center_tau_above_radius=columns(edges,total),
        static_center_tau_one_radius_rp=tau_one(edges,total),
        branch_nominal_center_tau_one_radius_rp=[tau_one(edges,chi[:,i]) for i in range(3)])
    for key in ("temperature_k","n_h_cm3","electron_cm3","mean_molecular_weight",
                "pressure_dyn_cm2","radial_velocity_kms","thermal_net","thermal_scale",
                "thermal_advection","thermal_expansion"):
        result[key]=physical[key]
    result["flow"]={key.removeprefix("wind__"):value for key,value in physical.items() if key.startswith("wind__")}
    result["mass_per_h_g"]=float(physical["mass_per_h"])
    mass_flux_density=physical["n_h_cm3"]*physical["mass_per_h"]*physical["radial_velocity_kms"]*1e5
    for name in ("additional","required"):
        result["flow"][name+"_support_mechanical_power_erg_cm3_s"]=mass_flux_density*result["flow"][name+"_support_acceleration_cm_s2"]
    return result


def encode(value):
    if isinstance(value,dict):
        return {key:encode(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [encode(item) for item in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    return value


def plot(output,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    edges=np.asarray(output["edges_rp"])
    quantities=("n2_cm3","nominal_center_total_source_per_wavenumber")
    baseline=output["baseline"]
    for ax,key in zip(axes,quantities):
        ax.stairs(baseline[key],edges,color="k",label="D=1, spherical")
    selected=[r for r in output["candidates"] if r["factor"]>1
              and np.any(np.isclose(r["factor"],(2.,5.,10.,20.),rtol=1e-12,atol=0))]
    if not selected and output["candidates"][-1]["factor"]>1:
        selected=output["candidates"][-1:]
    for row in selected:
        for ax,key in zip(axes,quantities):
            ax.stairs(row[key],edges,label=f"D={row['factor']:g}, spherical")
    free=output["free_model1_reference"]
    for ax,key in zip(axes,("full_n2_cm3","full_source")):
        ax.plot(free["full_radius_rp"],free[key],"--",color=".5",label="Free #1, different geometry")
    for ax in axes:
        ax.set(yscale="log",xlim=(1.3,3.),xlabel="Radius / Rp")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("n=2 population [cm$^{-3}$]")
    axes[1].set_ylabel("Nominal-center line source per wavenumber")
    axes[0].legend(fontsize=8)
    fig.suptitle("Outer population reservoir and local H-alpha source")
    fig.text(.5,.015,"Static local diagnostics; different #1 geometry. These curves do not predict observed flux or fit.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.97))
    for suffix in ("png","pdf"):
        fig.savefig(path.with_suffix("."+suffix),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--candidates",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plot",action="store_true",help="Make one final cross-candidate figure next to the JSON")
    parser.add_argument("--recipe",type=Path,default=Path("/tmp/escape-corrected-observer-full-f/recipe.json"))
    args=parser.parse_args()
    free_path=ROOT/"mceg/evidence/halpha_causes_20261127/sources/free_shape.json"
    sources=[Path(__file__),args.recipe,free_path,ROOT/"examples/halpha_geometry_fit.py",ROOT/"examples/cloudy_halpha.py",
             *[ROOT/"src/escape"/name for name in ("hydrogen_atom.py","nlte_forward.py","hydrogen_rates.py","line_escape_probability.py")]]
    pins={str(p):digest(p) for p in sources}
    edges=np.asarray(json.loads(args.recipe.read_text())["edges_rp"])
    p,s,baseline_receipt=load(args.baseline,pins)
    base=characterize(p,s,edges)
    records=[]
    for folder in args.candidates:
        p,s,receipt=load(folder,pins)
        row=characterize(p,s,edges)
        ratios={key:row[key]/base[key] for key in ("n2_cm3","n3_cm3","n3_over_n2","temperature_k","electron_cm3",
            "n_h_cm3","radial_velocity_kms","branch_source_per_wavenumber",
            "nominal_center_total_extinction_cm1","nominal_center_total_source_per_wavenumber")}
        imposed=row["flow"]["density_enhancement"]
        row.update(directory=str(folder),factor=receipt["factor"],ratio_to_D1=ratios,
            reference_wind_density_response=ratios["n_h_cm3"]/imposed,
            gpu_receipt_cpu_check_pending=receipt.get("independent_cpu_check_pending",None),
            solver_residual_infinity=receipt["residual_infinity"])
        records.append(row)
    free=json.loads(free_path.read_text())["free"]["original"]
    output=dict(scope=__doc__,edges_rp=edges.tolist(),radius_rp=((edges[:-1]+edges[1:])/2).tolist(),
        nominal_center_wavenumber_cm=float(NIST_ROW["wav"][0]),baseline_directory=str(args.baseline),
        baseline=base,candidates=records,source_and_input_sha256=pins,
        free_model1_reference={key:free[key] for key in ("full_radius_rp","full_n2_cm3","full_source",
            "n2_column_above_radius_cm2","static_center_tau_one_radius_rp")},
        inputs_unchanged=all(digest(path)==sha for path,sha in pins.items()),
        limitations=["The q factor is relative to the current-T/mu reference wind. Actual density/D1 also includes the reference-wind response.",
            "Sources are per wavenumber; opacities include stimulated emission. No continuum or bulk-velocity shift is included in static line diagnostics.",
            "Spherical halo emission and phase-dependent covering require the separate observer calculation.",
            "Support mechanical power rho*v*a is not added to heat. The force energy supply and dissipation are unspecified; required support includes the reference interpolant's momentum defect.",
            "Free model #1 is a prescribed unresolved line with a different plane/no-halo geometry. Its structural overlay is not a matched physical or spectral counterfactual.",
            "The isolated Stark diagnostic uses observer thermal/natural branch profiles and radial midpoint optical depth. It is not an exact native quadrature-average escape, a line width, or an added rate in the coupled H-alpha group. Its magnitude cannot establish observed pressure-wing dominance.",
            "gpu_receipt_cpu_check_pending preserves the immutable GPU receipt status. Subsequent independent CPU acceptance has its own receipt; this helper does not establish it."])
    assert output["inputs_unchanged"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(encode(output),indent=2,allow_nan=False)+"\n")
    if args.plot:
        plot(output,args.output.with_name("radial_structure"))
    print(json.dumps([dict(factor=row["factor"],n2_cell5=row["n2_cm3"][5],
        source_ratio_cell5=row["ratio_to_D1"]["nominal_center_total_source_per_wavenumber"][5],
        outer_N2=row["n2_column_above_radius_cm2"]["1.725"],tau_one=row["static_center_tau_one_radius_rp"])
        for row in records],indent=2))


if __name__ == "__main__":
    main()
