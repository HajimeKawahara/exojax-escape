"""Check declared component identities without claiming complete model equivalence.

Run in an authoring environment with MCEG schema 0.5. By default this script
does not import escape; --runtime adds a finite escape/JAX helper probe.
Neither mode executes an atmosphere, Cloudy or NUTS calculation.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

from mceg.schema_v05 import ContractV05, mathematical_model_digest
from mceg.validator import load_contract


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "mceg/contracts/escape.mathematical_model.yaml"


def verify():
    contract = load_contract(CONTRACT)
    models = {model.id: model for model in contract.mathematical_models}
    wind = {relation.id: relation for relation in models["composition_wind"].relations}
    radiation = {relation.id: relation for relation in models["continuum_feedback"].relations}
    gp = {relation.id: relation for relation in models["conditional_gp"].relations}
    checks = []

    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "status": "passed"})

    require("gas_and_effective_temperature_are_distinct",
            wind["effective_temperature"].quantities["T"]
            != wind["effective_temperature"].quantities["T_eff"])
    require("pressure_uses_effective_temperature",
            wind["pressure_ratio"].quantities["T_eff"]
            == wind["effective_temperature"].quantities["T_eff"])
    require("continuity_and_hydrogen_conversion_share_mass_density",
            wind["mass_continuity"].quantities["rho"]
            == wind["hydrogen_mass_conversion"].quantities["rho"])
    require("hydrogen_nuclei_are_not_mass_density",
            wind["hydrogen_mass_conversion"].quantities["nH"]
            != wind["hydrogen_mass_conversion"].quantities["rho"])
    selection = radiation["continuum_field_selection"].quantities
    require("slab_shared_and_used_fields_are_distinct", len({
        selection["J_slab"], selection["J_shared"], selection["J_used"]}) == 3)
    require("photoionization_uses_selected_field",
            radiation["continuum_photoionization"].quantities["J_used"] == selection["J_used"])
    require("se_and_photoionization_share_rate",
            radiation["local_bound_balance"].quantities["Gamma"]
            == radiation["continuum_photoionization"].quantities["Gamma"])
    require("continuum_pumping_is_not_aliased_to_rate_mean",
            radiation["continuum_pump_selection"].quantities["J_pump"] != selection["J_used"])
    require("gp_mean_and_covariance_share_projected_design",
            gp["gp_project_data"].quantities["G"] == gp["gp_project_covariance"].quantities["G"])
    require("gp_prior_is_a_distribution_relation",
            gp["conditional_planet_prior"].kind == "distribution"
            and gp["conditional_planet_prior"].role == "prior")
    require("gp_planet_prior_parameters_reach_native_model",
            all(gp["conditional_planet_prior"].quantities[symbol]
                == gp["conditional_native_model"].quantities[symbol]
                for symbol in ("rv", "log_Ap")))
    require("gp_likelihood_uses_derived_native_model",
            gp["gp_project_data"].quantities["m"] == gp["conditional_native_model"].quantities["m"]
            and models["conditional_gp"].quantities["log_model"].role == "derived")

    # A scientifically wrong but unit-compatible connection is structurally
    # legal. A declared model pin must change; units cannot catch this mistake.
    changed = contract.model_dump(mode="json")
    for model in changed["mathematical_models"]:
        if model["id"] == "continuum_feedback":
            for relation in model["relations"]:
                if relation["id"] == "continuum_photoionization":
                    relation["quantities"]["J_used"] = "slab_mean"
    altered = ContractV05.model_validate(changed)
    require("wrong_field_reconnection_changes_model_pin",
            mathematical_model_digest(contract, "continuum_feedback", "1")
            != mathematical_model_digest(altered, "continuum_feedback", "1"))

    missing = deepcopy(changed)
    del missing["mathematical_models"][0]["relations"][0]["quantities"]["T_eff"]
    try:
        ContractV05.model_validate(missing)
    except ValueError:
        require("missing_symbol_mapping_is_rejected", True)
    else:
        raise AssertionError("missing_symbol_mapping_is_rejected")

    require("runtime_metadata_has_no_mceg_dependency",
            not re.search(r'''["']mceg(?:[\s\[<>=!~;"']|$)''',
                          (ROOT / "pyproject.toml").read_text(), re.IGNORECASE))
    imports = []
    for path in (ROOT / "src/escape").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.append(node.module.split(".")[0])
    require("runtime_sources_do_not_import_mceg", "mceg" not in imports)

    return {
        "scope": "Finite structural and declared-identity probes only; no complete physical/code equivalence.",
        "checks": checks,
        "model_digests": {model.id: mathematical_model_digest(contract, model.id, model.revision)
                          for model in contract.mathematical_models},
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)):
                          sha256(Path(__file__).read_bytes()).hexdigest()},
    }


def verify_runtime_field_selection():
    """Exercise the actual replacement helper at one finite two-cell probe."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from escape.hydrogen_escape_probability import _replace_complete_mean

    jax.config.update("jax_enable_x64", True)
    original = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    complete = original * 10
    mask = jnp.array([True, False, True])
    selected, returned_mask, valid = _replace_complete_mean(original, (mask, complete))
    expected = np.where(np.asarray(mask)[None, :], np.asarray(complete), np.asarray(original))
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(returned_mask, mask)
    assert bool(valid)
    original_gradient, override_gradient = jax.grad(
        lambda old, new: _replace_complete_mean(old, (mask, new))[0].sum(), argnums=(0, 1)
    )(original, complete)
    np.testing.assert_array_equal(original_gradient, np.broadcast_to(~np.asarray(mask), original.shape))
    np.testing.assert_array_equal(override_gradient, np.broadcast_to(np.asarray(mask), original.shape))
    unchanged, empty_mask, valid_without_override = _replace_complete_mean(original, None)
    np.testing.assert_array_equal(unchanged, original)
    assert not np.asarray(empty_mask).any() and bool(valid_without_override)
    bad = complete.at[0, 0].set(-1.)
    assert not bool(_replace_complete_mean(original, (mask, bad))[2])
    return {
        "scope": "One two-cell, three-frequency helper probe; not the full population-state data path.",
        "checks": ["complete_field_replaces_instead_of_adds", "unselected_frequencies_preserved",
                   "gradients_select_the_correct_source", "absent_override_preserves_original",
                   "negative_selected_intensity_is_invalid"],
        "maximum_value_error": float(np.max(np.abs(np.asarray(selected) - expected))),
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", action="store_true",
                        help="Also run the finite field-selection helper probe using escape/JAX.")
    args = parser.parse_args()
    result = verify()
    if args.runtime:
        result["runtime_field_selection"] = verify_runtime_field_selection()
    result = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
