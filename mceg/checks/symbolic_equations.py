"""Author and check the seven explicit SymPy relations in the scoped graph.

This trusted authoring script constructs expressions; MCEG never imports it.
The serialized expressions in the contract are the mathematical source.
Opaque transport operators and statistical specifications remain in LaTeX.
Run --runtime for finite comparisons with the unchanged escape implementation.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

import sympy as sp

from mceg.schema_v05 import mathematical_model_digest
from mceg.symbolic import from_sympy, to_sympy
from mceg.validator import load_contract


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "mceg/contracts/escape.mathematical_model.yaml"
MODEL_REVISIONS = {"composition_wind": "2", "continuum_feedback": "2", "conditional_gp": "1"}


def definitions():
    """Construct expressions using exactly the contract's local symbol IDs."""
    r, kB, mH, GMp, kappa, mdot, mass_per_h = sp.symbols("r kB mH GMp kappa mdot mass_per_h")
    T, mu, T_eff, a2, v, rho, nH = (sp.Function(name)(r) for name in
                                    ("T", "mu", "T_eff", "a2", "v", "rho", "nH"))
    M, J_used, J_slab, J_shared, J_pump, J_fwd = sp.symbols(
        "M J_used J_slab J_shared J_pump J_fwd")
    return {
        "effective_temperature": sp.Eq(T_eff, T / mu, evaluate=False),
        "pressure_ratio": sp.Eq(a2, kB * T_eff / mH, evaluate=False),
        "wind_momentum": sp.Eq((v - a2 / v) * sp.Derivative(v, r),
            2 * a2 / r - sp.Derivative(a2, r) - GMp / r**2 + kappa * r, evaluate=False),
        "mass_continuity": sp.Eq(rho, mdot / (4 * sp.pi * r**2 * v), evaluate=False),
        "hydrogen_mass_conversion": sp.Eq(nH, rho / mass_per_h, evaluate=False),
        "continuum_field_selection": sp.Eq(J_used, (1 - M) * J_slab + M * J_shared, evaluate=False),
        "continuum_pump_selection": sp.Eq(J_pump, (1 - M) * J_fwd + M * J_shared, evaluate=False),
    }


def verify():
    contract = load_contract(CONTRACT)
    equations = {(equation.id, equation.revision): equation for equation in contract.equations}
    checks = []

    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    for name, expression in definitions().items():
        equation = equations[name, "2"]
        require(f"{name}:stored_source_matches_authoring", equation.sympy == from_sympy(expression))
        require(f"{name}:round_trip", sp.srepr(to_sympy(equation.sympy)) == sp.srepr(expression))
        names = {str(symbol) for symbol in expression.free_symbols}
        names.update(str(node.func) for node in expression.atoms(sp.Function))
        require(f"{name}:declared_symbol_identities", names == set(equation.symbols))
        require(f"{name}:latex_is_not_a_second_source", "latex" not in equation.model_dump(mode="json"))
    require("twelve_current_relations_retain_latex_sources",
            sum((equation.id not in definitions()) and equation.sympy is None
                for equation in contract.equations) == 12)
    require("nineteen_historical_equations_remain_latex_sources",
            sum(equation.revision == "1" and equation.sympy is None
                for equation in contract.equations) == 19)
    models = {model.id: model for model in contract.mathematical_models
              if MODEL_REVISIONS[model.id] == model.revision}
    wind = {relation.id: relation for relation in models["composition_wind"].relations}
    radiation = {relation.id: relation for relation in models["continuum_feedback"].relations}
    require("pressure_retains_distinct_effective_temperature",
            wind["pressure_ratio"].quantities["T_eff"] == wind["effective_temperature"].quantities["T_eff"]
            != wind["effective_temperature"].quantities["T"])
    require("continuity_and_hydrogen_conversion_share_mass_density",
            wind["mass_continuity"].quantities["rho"] == wind["hydrogen_mass_conversion"].quantities["rho"])
    selection = radiation["continuum_field_selection"].quantities
    require("selected_mean_drives_photoionization",
            radiation["continuum_photoionization"].quantities["J_used"] == selection["J_used"])
    require("three_radiation_fields_remain_distinct",
            len({selection[key] for key in ("J_used", "J_slab", "J_shared")}) == 3)
    changed = contract.model_dump(mode="json")
    altered_model = next(model for model in changed["mathematical_models"]
                         if model["id"] == "continuum_feedback" and model["revision"] == "2")
    relation = next(relation for relation in altered_model["relations"]
                    if relation["id"] == "continuum_photoionization")
    relation["quantities"]["J_used"] = "slab_mean"
    altered = type(contract).model_validate(changed)
    require("wrong_same_unit_field_changes_model_pin",
            mathematical_model_digest(contract, "continuum_feedback", "2")
            != mathematical_model_digest(altered, "continuum_feedback", "2"))
    missing = deepcopy(changed)
    del next(model for model in missing["mathematical_models"]
             if model["id"] == "composition_wind" and model["revision"] == "2")["relations"][0]["quantities"]["T_eff"]
    try:
        type(contract).model_validate(missing)
    except ValueError:
        require("missing_symbol_mapping_is_rejected", True)
    else:
        raise AssertionError("missing_symbol_mapping_is_rejected")
    imports = []
    for path in (ROOT / "src/escape").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.append(node.module.split(".")[0])
    require("runtime_sources_have_no_mceg_or_sympy_import", not {"mceg", "sympy"}.intersection(imports))
    require("runtime_metadata_has_no_mceg_or_sympy_dependency",
            not re.search(r'''["'](?:mceg|sympy)(?:[\s\[<>=!~;"']|$)''',
                          (ROOT / "pyproject.toml").read_text(), re.IGNORECASE))
    return {
        "scope": "Stored symbolic sources, explicit connections and source independence; not complete code correspondence.",
        "checks": checks,
        "status": "passed",
        "model_digests": {f"{name}@{revision}": mathematical_model_digest(contract, name, revision)
                          for name, revision in MODEL_REVISIONS.items()},
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)):
                          sha256(Path(__file__).read_bytes()).hexdigest()},
    }


def verify_runtime():
    """Compare stored symbolic expressions with small, declared native probes."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exojax.utils.constants import RJ, kB
    from escape.atmosphere_wind import _effective_temperature, thermal_flow_terms, wind_from_profiles
    from escape.hydrogen_escape_probability import _replace_complete_mean
    from escape.parker import PROTON_MASS_CGS
    from escape.solar import SolarAtmosphereModel

    jax.config.update("jax_enable_x64", True)
    contract = load_contract(CONTRACT)
    equations = {equation.id: to_sympy(equation.sympy) for equation in contract.equations
                 if equation.sympy is not None}
    r = sp.Symbol("r")
    field = lambda name: sp.Function(name)(r)
    checks = []

    def compare(name, actual, expected, rtol=3e-14):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)
        scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
        checks.append({"name": name, "maximum_relative_error":
                       float(np.max(np.abs(np.asarray(actual) - expected) / scale)), "rtol": rtol})

    radius = np.geomspace(1.9 * RJ, 3 * 1.9 * RJ, 6)
    temperature = np.array([9000., 9500., 10000., 11000., 10500., 10000.])
    mu = np.array([.9, .85, .8, .75, .7, .65])
    symbolic_temperature = np.array([float(equations["effective_temperature"].rhs.subs(
        {field("T"): t, field("mu"): m})) for t, m in zip(temperature, mu)])
    parameters = (jnp.log(radius), jnp.log(temperature / mu))
    compare("effective_temperature_at_profile_knots", jax.vmap(
        lambda q: _effective_temperature(q, parameters))(jnp.asarray(radius)), symbolic_temperature)

    symbolic_pressure = np.array([float(equations["pressure_ratio"].rhs.subs(
        {field("T_eff"): t, sp.Symbol("kB"): kB, sp.Symbol("mH"): PROTON_MASS_CGS}))
        for t in symbolic_temperature])
    # Read the pressure factor through the actual expansion term with supplied
    # d(log v)/dr=0; then d(rho)/dr=-2 rho/r by native mass conservation.
    density = np.full(6, 1e-14)
    terms = thermal_flow_terms(radius, density, temperature, mu, np.ones(6),
        log_velocity_gradient_cm1=np.zeros(6), sound_squared_gradient_cm_s2=np.zeros(6))
    compare("pressure_ratio_in_native_expansion_term", np.asarray(terms.expansion) / (-2 * density / radius),
            symbolic_pressure)

    wind, info = wind_from_profiles(radius, np.full(6, 10000.), np.full(6, .7), 1e13, 2.17,
        base_radius_cm=radius[0], outer_radius_cm=32 * radius[0], return_info=True)
    assert bool(info["valid"]) and bool(info["converged"])
    symbolic_density = np.array([float(equations["mass_continuity"].rhs.subs(
        {r: q, field("v"): float(v) * 1e5, sp.Symbol("mdot"): 1e13}))
        for q, v in zip(radius, wind.velocity_kms)])
    compare("native_wind_mass_density_in_cgs", wind.density_g_cm3, symbolic_density)

    # Invoke the real atmosphere method, capturing its density input before
    # chemistry/thermal work. This tests only the density conversion line.
    mass_per_h = 1.4 * PROTON_MASS_CGS
    captured = {}

    class CapturedDensity(Exception):
        pass

    def capture(composition, density, temperature, incident_mean_intensity, **kwargs):
        captured["density"] = density
        raise CapturedDensity

    provider = SimpleNamespace(cells=6, state_size=1, mass_per_h=mass_per_h, atomic_lte_mask=None,
                               _wind=lambda *args: (wind, info), evaluate=capture)
    try:
        SolarAtmosphereModel.evaluate_atmosphere(provider, jnp.zeros(12), 10000., 1e13, 0.)
    except CapturedDensity:
        pass
    expected_nh = np.array([float(equations["hydrogen_mass_conversion"].rhs.subs(
        {field("rho"): d, sp.Symbol("mass_per_h"): mass_per_h})) for d in symbolic_density])
    compare("hydrogen_density_passed_to_native_composition", captured["density"], expected_nh)

    original = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    complete = 10 * original
    mask = jnp.array([True, False, True])
    for name, original_name in (("continuum_field_selection", "J_slab"), ("continuum_pump_selection", "J_fwd")):
        expression = equations[name].rhs
        expected = np.array([[float(expression.subs({sp.Symbol(original_name): float(old),
            sp.Symbol("J_shared"): float(new), sp.Symbol("M"): int(selected)}))
            for old, new, selected in zip(row, shared, mask)] for row, shared in zip(original, complete)])
        selected, _, valid = _replace_complete_mean(original, (mask, complete))
        assert bool(valid)
        compare(f"{name}:native_helper_values", selected, expected, rtol=0)
        gradients = jax.grad(lambda old, new: _replace_complete_mean(old, (mask, new))[0].sum(),
                             argnums=(0, 1))(original, complete)
        for derivative, symbol in zip(gradients, (original_name, "J_shared")):
            symbolic_gradient = np.broadcast_to([float(sp.diff(expression, sp.Symbol(symbol)).subs(
                sp.Symbol("M"), int(value))) for value in mask], original.shape)
            compare(f"{name}:gradient_{symbol}", derivative, symbolic_gradient, rtol=0)
    return {"status": "passed", "checks": checks,
            "scope": "Six converted scalar relations at finite probes, including helper gradients. Momentum correspondence, full atmosphere and observer remain unproved."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check stored sources and connections (default).")
    mode.add_argument("--definitions", action="store_true", help="Print serialized definitions for deliberate new-revision authoring.")
    args = parser.parse_args()
    if args.definitions:
        result = {name: from_sympy(expression).model_dump(mode="json") for name, expression in definitions().items()}
    else:
        result = verify()
        if args.runtime:
            result["runtime"] = verify_runtime()
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
