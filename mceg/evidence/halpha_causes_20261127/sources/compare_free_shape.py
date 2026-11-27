"""Compare prescribed model-1 structure with the saved physical atmosphere.

No spectrum or atmosphere is fitted. Local source and static radial optical
depths are descriptive: the original plane columns and spherical observer
have different geometries, continuum, branches and velocity projection.
"""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import voigt_profile

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples"))
from cloudy_halpha import NIST_ROW
from escape.hydrogen_forward import HydrogenForwardModel, PLANET_RADIUS_CM
from exojax.opacity.lpf.line_coefficients import line_coefficients
from exojax.database.core.broadening import doppler_sigma


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upper_columns(edges, values):
    return {str(cut): float(values @ (np.maximum(0., edges[1:] - np.maximum(cut, edges[:-1])) * PLANET_RADIUS_CM))
        for cut in (1.3, 1.725, 2.15)}


def tau_one_radius(edges, chi):
    above = np.r_[np.cumsum((chi * np.diff(edges) * PLANET_RADIUS_CM)[::-1])[::-1], 0.]
    matches = np.flatnonzero((above[:-1] >= 1) & (above[1:] < 1))
    if not len(matches):
        return None
    i = matches[0]
    return float(edges[i + 1] - (1 - above[i + 1]) / (chi[i] * PLANET_RADIUS_CM))


def main():
    archive = Path("/home/kawahara/exojax-escape/results/contrast_response_20261126")
    parameters = Path("/home/kawahara/exojax-escape/results/three_models/parameters.json")
    physical_path = Path("/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz")
    branch_path = physical_path.with_name("branch_arrays.npz")
    recipe_path = Path("/tmp/escape-corrected-observer-full-f/recipe.json")
    source_paths = [ROOT / "src/escape" / name for name in ("hydrogen_forward.py", "parker.py")]
    for path in source_paths:
        assert digest(path) == digest(Path("/tmp/escape-phase-night-pr/src/escape") / path.name)
    inputs = [Path(__file__), parameters, physical_path, branch_path, recipe_path, physical_path.with_name("plan.json"),
        archive / "free_coarse.json", archive / "native_fit.py", *source_paths,
        ROOT / "examples/cloudy_halpha.py"]
    pins = {str(path): digest(path) for path in inputs}
    original = json.loads(parameters.read_text())["models"]["1"]["best"]["coordinates"][:5]
    fitted = json.loads((archive / "free_coarse.json").read_text())["best"]["parameters"][:5]
    with patch("exojax.database.nist.api.fetch_nist_lines", lambda *a, **kw: (pd.DataFrame(NIST_ROW), [])):
        model = HydrogenForwardModel(nlayer=64, nlon=1, nlat=1, nnu=3)
    state, branch = dict(np.load(physical_path)), dict(np.load(branch_path))
    edges = np.asarray(json.loads(recipe_path.read_text())["edges_rp"])
    radius, thickness = (edges[:-1] + edges[1:]) / 2, np.diff(edges) * PLANET_RADIUS_CM
    center = NIST_ROW["wav"][0]
    phi = voigt_profile(center - branch["branch_centers"][None], branch["branch_sigma"], branch["branch_gamma"][None])
    chi, eta = np.sum(branch["branch_alpha"] * phi, axis=1), np.sum(branch["branch_eta"] * phi, axis=1)
    n2 = state["bound_cm3"][:, 1:3].sum(axis=1)
    physical = dict(radius_rp=radius.tolist(), thickness_rp=np.diff(edges).tolist(),
        temperature_k=state["temperature_k"].tolist(), radial_velocity_kms=state["radial_velocity_kms"].tolist(),
        zonal_kms=json.loads((physical_path.with_name("plan.json")).read_text())["zonal_kms"],
        n2_cm3=n2.tolist(), n2_column_cm2=float(n2 @ thickness),
        upper_n2_column_cm2=float(n2[radius >= 1.3] @ thickness[radius >= 1.3]),
        n2_column_above_radius_cm2=upper_columns(edges, n2),
        static_center_tau_above_radius=upper_columns(edges, chi),
        static_center_tau_one_radius_rp=tau_one_radius(edges, chi),
        thermal_sigma_kms=(np.mean(branch["branch_sigma"] / branch["branch_centers"][None], axis=1) * 299792.458).tolist(),
        rest_center_source=(eta / chi).tolist(), branch_source=(branch["branch_eta"] / branch["branch_alpha"]).tolist(),
        static_radial_center_tau_per_cell=(chi * thickness).tolist(),
        upper_static_radial_center_tau=float(chi[radius >= 1.3] @ thickness[radius >= 1.3]))
    free = {}
    for name, p in (("original", original), ("native_refitted_shape", fitted)):
        ncolumn, floor, ttop, twind, zonal = np.exp(p[0]) * 1e14, np.exp(p[1]), p[2] * 1e4, p[3] * 1e4, p[4] * 10
        temperature = model.temperature_profile(ttop)
        ratio = model.prescribed_ratio(floor, ttop)
        lower = model.n2_reference * ncolumn / 3e14
        phi_free = jax.vmap(model.opa.line_profiles)(temperature)[:, :, 1:2]
        alpha, emission = line_coefficients(phi_free, model.adb.nu_lines, model.adb.A,
            model.adb.glower, model.adb.gupper, lower[:, None], (lower * ratio)[:, None])
        tau = np.asarray(alpha[:, 0] * model.thickness_cm)
        source = np.asarray(emission[:, 0] / alpha[:, 0])
        r = np.asarray(model.radius_rp)
        free_edges = np.linspace(1., 3., 65)
        upper_thickness = np.maximum(0., free_edges[1:] - np.maximum(1.3, free_edges[:-1]))[::-1] * PLANET_RADIUS_CM
        sigma_kms = jax.vmap(lambda t: doppler_sigma(model.adb.nu_lines, t, model.adb.line_masses)[0])(temperature) * 299792.458 / center
        at_physical = lambda value: np.interp(radius, r[::-1], np.asarray(value)[::-1])
        free[name] = dict(n2_column_cm2=float(ncolumn), departure_floor=float(floor),
            kinetic_temperature_top_k=float(ttop), wind_temperature_k=float(twind), zonal_kms=float(zonal),
            n2_exponential_scale_rp=.35, source_departure_sigmoid_midpoint_rp=1.45,
            source_departure_sigmoid_scale_rp=.08, layer_thickness_rp=2 / 64,
            upper_n2_column_cm2=float(np.asarray(lower) @ upper_thickness),
            upper_static_radial_center_tau=float(np.asarray(alpha[:, 0]) @ upper_thickness), total_static_radial_center_tau=float(tau.sum()),
            n2_column_above_radius_cm2=upper_columns(free_edges, np.asarray(lower)[::-1]),
            static_center_tau_above_radius=upper_columns(free_edges, np.asarray(alpha[:, 0])[::-1]),
            static_center_tau_one_radius_rp=tau_one_radius(free_edges, np.asarray(alpha[:, 0])[::-1]),
            at_physical_centers=dict(n2_cm3=at_physical(lower).tolist(), temperature_k=at_physical(temperature).tolist(),
                radial_velocity_kms=at_physical(model.radial_velocity(twind, 2/3)).tolist(),
                thermal_sigma_kms=at_physical(sigma_kms).tolist(),
                source=at_physical(source).tolist(), source_ratio_to_physical=at_physical(source).tolist()),
            full_radius_rp=r[::-1].tolist(), full_n2_cm3=np.asarray(lower)[::-1].tolist(),
            full_source=source[::-1].tolist(), full_static_radial_center_tau=tau[::-1].tolist())
        free[name]["at_physical_centers"]["source_ratio_to_physical"] = (at_physical(source) / (eta / chi)).tolist()
    result = dict(scope=__doc__.strip(), physical=physical, free=free, source_and_input_sha256=pins,
        unchanged_inputs=all(digest(Path(path)) == value for path, value in pins.items()),
        limitations=["Static optical depths are at the common nominal H-alpha center with no bulk shift or continuum. They are not observer line-of-sight optical depths.",
            "Free-model values at physical centers are interpolations of its 64 prescribed layers. Its source is an unresolved n=2/3 transition, whereas the physical state has three resolved branches.",
            "The free shape includes a smooth independent excited column, a prescribed departure sigmoid, separate kinetic/wind temperatures and zonal velocity, plane columns, and no atmospheric continuum or off-limb emission. No single cause is isolated by comparing these models.",
            "The historical additive display baseline is omitted from all current native-data comparisons. No parameters are refitted here."])
    destination = Path(__file__).with_name("free_shape.json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(physical=physical, free={name: {key: value for key, value in row.items() if not key.startswith("full_")} for name,row in free.items()})))


if __name__ == "__main__":
    main()
