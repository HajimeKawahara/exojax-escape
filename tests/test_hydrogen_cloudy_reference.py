"""Measure external Cloudy disagreement separately from numerical correctness.

Cloudy is not a runtime/test dependency: independently executed reference
outputs, inputs and reporting-only patches are retained with the fixture.
The provider workflow applies the physical accuracy criterion to these
metrics; a passing pytest result does not assert agreement with Cloudy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_atom import make_hydrogen_atom, solve_atom_populations
from escape.hydrogen_rates import EV_CGS, PLANCK_CGS, planck_occupation

jax.config.update("jax_enable_x64", True)


def test_cloudy_thin_reference(record_property):
    fixture = json.loads((Path(__file__).parent / "data"
                          / "cloudy_hydrogen_thin_reference.json").read_text())
    evidence = Path(__file__).parents[1] / fixture["evidence_directory"]
    provenance_bytes = (evidence / "provenance.json").read_bytes()
    assert hashlib.sha256(provenance_bytes).hexdigest() == fixture["provenance_sha256"]
    provenance = json.loads(provenance_bytes)
    for name, expected in provenance["retained_file_sha256"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == expected
    conditions = fixture["conditions"]
    temperature = conditions["temperature_k"]
    radiation_temperature = conditions["radiation_temperature_k"]
    dilution = conditions["radiation_dilution"]
    for name in ("temperature_k", "n_h_cm3", "radiation_temperature_k", "radiation_dilution"):
        record_property(name, conditions[name])
    record_property("external_population_tolerance", fixture["relative_population_tolerance"])
    nodes, weights = np.polynomial.legendre.leggauss(48)
    for model in fixture["models"]:
        nmax = model["nmax"]
        assert model == next(row for row in provenance["all_runs"]
                             if row["run"] == model["run"])
        saved = np.loadtxt(evidence / (model["run"] + ".levels"))
        np.testing.assert_allclose([model["n2_cm3"], model["n3_cm3"]],
                                   [np.sum(saved[2:4]), np.sum(saved[4:7])], rtol=1e-12)
        atom = make_hydrogen_atom(nmax)
        edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                                100 * EV_CGS / PLANCK_CGS])
        widths = np.diff(np.log(edges))
        frequency = edges[:-1, None] * np.exp(widths[:, None] * (nodes + 1) / 2)
        integration_weights = frequency * widths[:, None] * weights / 2
        frequency = jnp.asarray(frequency.ravel())
        solution = solve_atom_populations(
            conditions["n_h_cm3"], atom, temperature,
            planck_occupation(atom.line_frequency_hz, radiation_temperature, dilution),
            planck_occupation(frequency, radiation_temperature, dilution), frequency,
            jnp.asarray(integration_weights.ravel()),
        )
        assert solution.converged
        for shell, indices in ((2, slice(1, 3)), (3, slice(3, 6))):
            actual = float(jnp.sum(solution.bound_cm3[indices]))
            reference = model[f"n{shell}_cm3"]
            assert actual > 0 and np.isfinite(actual)
            record_property(f"escape_n{shell}_cm3_nmax{nmax}", actual)
            record_property(f"cloudy_n{shell}_cm3_nmax{nmax}", reference)
            record_property(f"n{shell}_relative_error_cloudy_nmax{nmax}",
                            abs(actual / reference - 1))
