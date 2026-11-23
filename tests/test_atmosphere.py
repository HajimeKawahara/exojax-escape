"""Conservation, energy closure and AD of the complete selected observer path.

The small atom/grid is a numerical experiment, not the KELT-9b reference model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.atmosphere import HydrogenAtmosphereModel
from escape.hydrogen_forward import PLANET_RADIUS_CM
from escape.parker import PROTON_MASS_CGS
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    return HydrogenAtmosphereModel(
        nlayer=4, nmax=3, nnu=61, nlon=4, nlat=2, wind_ngrid=33,
        tidal_coefficient_s2=3 * (2 * np.pi / (1.48 * 86400))**2,
    )


def inputs(model):
    return (jnp.full(4, 8000.), model.radiation.boundary_blackbody(4500.),
            model.radiation.boundary_blackbody(10000., .05))


def test_joint_root_closes_mass_charge_and_thermal_flow(model, record_property):
    result = model.solve(1e15, *inputs(model), iterations=24, continuation_steps=4)
    assert result["converged"]
    density = result["n_h_cm3"]
    np.testing.assert_allclose(result["bound_cm3"].sum(1) + result["electron_cm3"], density, rtol=2e-14)
    flux = (4 * np.pi * (model.radius_rp * PLANET_RADIUS_CM)**2 * PROTON_MASS_CGS
            * density * result["radial_velocity_kms"] * 1e5)
    np.testing.assert_allclose(flux, 1e15, rtol=2e-14)
    np.testing.assert_allclose(result["mean_molecular_weight"], 1 / (1 + result["ionized_fraction"]), rtol=2e-14)
    assert np.ptp(result["mean_molecular_weight"]) > .2
    assert np.ptp(result["temperature_k"]) > 1000
    heating, cooling = result["thermal_heating"], result["thermal_cooling"]
    adv, expansion = result["thermal_advection"], result["expansion"]
    error = (heating - cooling + adv + expansion) / (heating + cooling + abs(adv) + abs(expansion))
    assert np.max(np.abs(error[:-1])) < 1e-8  # bottom temperature is a boundary
    np.testing.assert_allclose(result["temperature_k"][-1], 8000., rtol=2e-14)
    record_property("joint_atmosphere_residual", float(result["residual_norm"]))
    record_property("thermal_flow_balance", float(jnp.max(abs(error[:-1]))))


def test_complete_spectrum_implicit_gradient(model, record_property):
    temperature, inner, outer = inputs(model)
    weights = jnp.exp(-.5 * ((model.velocity_kms - 12.) / 35.)**2)

    @jax.jit
    def signal(log_mass):
        return jnp.mean(weights * model.spectrum(
            1e15 * jnp.exp(log_mass), temperature, inner, outer,
            iterations=24, continuation_steps=4,
        ))

    step = 2e-4
    automatic = jax.grad(signal)(0.)
    finite = (signal(step) - signal(-step)) / (2 * step)
    assert np.isfinite(automatic) and abs(float(finite)) > 1e-12
    np.testing.assert_allclose(automatic, finite, rtol=5e-4, atol=1e-12)
    record_property("atmosphere_spectrum_gradient_relative_error", float(abs(automatic / finite - 1)))


def test_failed_atmosphere_is_not_replaced_by_an_uncoupled_spectrum(model):
    result = model.solve(-1., *inputs(model), iterations=2, continuation_steps=1)
    assert not result["converged"]
    assert np.isnan(result["temperature_k"]).all()
    assert np.isnan(result["bound_cm3"]).all()
    with pytest.raises(ValueError, match="both atmosphere and observer"):
        model.spectrum(1e15, *inputs(model), include_hminus=True)
