"""Branch damping, coefficient summation, and the supported transfer domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_atom import atom_natural_width_hz, make_hydrogen_atom
from escape.hydrogen_forward import HydrogenForwardModel
from escape.hydrogen_rates import LIGHT_SPEED_CGS
from test_hydrogen_forward import offline_nist  # noqa: F401

jax.config.update("jax_enable_x64", True)


def test_halpha_natural_width_includes_both_level_lifetimes():
    widths = np.asarray(atom_natural_width_hz(make_hydrogen_atom(15)))[2:5]
    # 3p has a Lyman-beta branch; 2p decays through Lyman-alpha.
    upper_decay = np.array([6.3143e6, 1.6725e8 + 2.2448333333333332e7, 6.46514e7])
    lower_decay = np.array([6.264833333333333e8, 8.2, 6.264833333333333e8])
    np.testing.assert_allclose(widths, (upper_decay + lower_decay) / (4 * np.pi),
                               rtol=2e-15)
    assert np.all(widths / LIGHT_SPEED_CGS > 44101000 / (4 * np.pi * LIGHT_SPEED_CGS))


def test_branch_sum_matches_common_profile_and_rejects_unsupported_extinction():
    class ThreeBranches(HydrogenForwardModel):
        def _line_profiles(self, temperature_k, velocity_kms):
            return jnp.repeat(super()._line_profiles(temperature_k, velocity_kms), 3, axis=0)

    options = dict(nlayer=4, nlon=2, nlat=2, nnu=101)
    one, three = HydrogenForwardModel(**options), ThreeBranches(**options)
    extinction = jnp.full((4, 1), 1e-11)
    source = jnp.full((4, 1), 1e5)
    velocity = jnp.zeros(4)
    common = (velocity, 0., 0., one.temperature_k)
    expected = one._transfer_coefficients(extinction, source, *common)
    # A negative individual branch is allowed when the summed opacity is positive.
    weights = extinction * jnp.array([-.1, .4, .7])
    emission = extinction * source * jnp.array([.2, .3, .5])
    actual = three._transfer_line_coefficients(weights, emission, *common)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-16)
    for invalid in (-extinction, jnp.zeros_like(extinction)):
        spectrum = one._transfer_line_coefficients(invalid, extinction * source, *common)
        assert np.isnan(spectrum).all()


def test_zero_column_derivative_preserves_emission_and_absorption():
    model = HydrogenForwardModel(nlayer=4, nlon=4, nlat=2, nnu=101)
    extinction = jnp.full((4, 1), 1e-12)
    source = jnp.full((4, 1), 1e5)

    def spectrum(scale):
        return model._transfer_coefficients(
            scale * extinction, source, jnp.zeros(4), 0., 0., model.temperature_k,
        )

    derivative = jax.jacfwd(spectrum)(0.)
    step = 1e-4
    finite = (spectrum(step) - spectrum(0.)) / step
    np.testing.assert_allclose(spectrum(0.), 0., atol=1e-16)
    np.testing.assert_allclose(derivative, finite, rtol=3e-5, atol=2e-12)
