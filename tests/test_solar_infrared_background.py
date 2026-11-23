"""One explicit spherical IR reservoir, independent of selected line windows."""

from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.solar import SolarCompositionModel

jax.config.update("jax_enable_x64", True)


def model(**options):
    return SolarCompositionModel(
        [1e10, 1.1e10, 1.4e10],
        nmax=6,
        atoms=(("C I", 6, 0),),
        oxygen_overlap=False,
        quadrature_order=2,
        angle_order=2,
        line_sampling="average",
        line_quadrature_order=2,
        infrared_continuum="spherical_shell_average",
        infrared_continuum_angle_order=32,
        resolved_infrared_lines=((6, 5),),
        infrared_options=dict(quadrature_order=2),
        **options,
    )


def test_complete_solar_ir_background_is_shared_and_line_window_independent():
    enabled = model()
    no_lines = copy(enabled)
    no_lines.infrared_groups = []
    no_lines.infrared_indices = []
    legacy = copy(no_lines)
    legacy.infrared_shell_geometry = None
    t = jnp.array([8000.0, 8500.0])
    nh = jnp.array([1e10, 1e8])
    initial = enabled.initialize(nh, t)
    incident = enabled.hydrogen.boundary_blackbody(10000.0, 0.025)

    def evaluate(m):
        _, p = m.evaluate(initial, nh, t, incident)
        return dict(
            valid=p["valid"],
            mean=p["radiation"]["mean_intensity"],
            pump=p["radiation"]["line_pump_mean_intensity"],
            background=p["infrared_background"],
            groups=p["infrared"],
            net=p["thermal_net"],
            ledger=p["thermal_ledger"],
        )

    result = jax.jit(lambda: evaluate(enabled))()
    bare = jax.jit(lambda: evaluate(no_lines))()
    old = jax.jit(lambda: evaluate(legacy))()
    assert result["valid"] and bare["valid"] and old["valid"]
    np.testing.assert_array_equal(result["mean"], bare["mean"])
    mask = enabled.infrared_continuum_mask
    np.testing.assert_array_equal(result["mean"][:, ~mask], old["mean"][:, ~mask])
    bg = result["background"]["solution"]
    np.testing.assert_allclose(
        result["mean"][:, mask], bg.radiation.mean_intensity, rtol=3e-15
    )
    expected = bg.emission_mean_intensity + bg.outer_boundary_response * incident[mask]
    np.testing.assert_allclose(result["mean"][:, mask], expected, rtol=3e-13)
    assert np.max(abs(result["mean"][:, mask] - old["mean"][:, mask])) > 0
    assert result["groups"][0]["converged"]
    np.testing.assert_allclose(
        result["net"], sum(result["ledger"].values()), rtol=3e-15
    )
    assert all("scattering" not in name for name in result["ledger"])
    # At fixed opacity, exact line-boundary changes use the linear response,
    # not an interpolated spectrum already contaminated by old direct pumping.
    frequency = enabled.hydrogen.atom.line_frequency_hz
    stellar = jnp.interp(frequency, enabled.grid.frequency_hz, incident)

    def shared_pump(scale):
        return enabled._shared_infrared_pump(
            result["background"], frequency, stellar * scale
        )[1]

    mean, derivative = jax.jvp(shared_pump, (jnp.array(1.0),), (jnp.array(1.0),))
    np.testing.assert_allclose(
        derivative, shared_pump(1.0) - shared_pump(0.0), rtol=3e-13
    )
    assert np.isfinite(mean).all()


def test_shared_band_is_fixed_and_must_cover_every_line_window():
    with pytest.raises(ValueError, match="fixed shared IR band"):
        model(infrared_band_min_wavelength_micron=8.0)
    with pytest.raises(ValueError, match="IR continuum"):
        SolarCompositionModel([0.0, 1e5], infrared_continuum="unknown")
