"""A complete shared mean replaces, rather than adds to, metal pumping."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.metal_network import MetalNetwork

jax.config.update("jax_enable_x64", True)


def setup(sampling, quadrature):
    frequency = jnp.geomspace(2e13, 3e16, 40)
    model = MetalNetwork(frequency, jnp.gradient(frequency), jnp.array([1e6, 2e6]),
        atoms=(("C I", 6, 0), ("C II", 6, 1)), line_sampling=sampling,
        line_quadrature=quadrature, line_quadrature_order=8,
        shielding_model="continuous_core", core_escape_model="continuous")
    temperature = jnp.array([6000., 9000.])
    abundance = jnp.array([[1e7], [2e7]])
    state = model.initialize(temperature, abundance)
    radiation = dict(mean_intensity=jnp.full((2, 40), 1e-8),
        pump_mean_intensity=jnp.full((2, 40), 1e-8),
        continuum_opacity_cm1=jnp.full((2, 40), 1e-9),
        ots_photon_flux_cm2_s=jnp.zeros((2, 1)))
    lines = len(model.line_frequency_hz)
    options = dict(line_incident_mean_intensity=jnp.linspace(1e-8, 3e-8, lines),
        line_continuum_opacity_cm1=jnp.full((2, lines), 1e-9),
        line_diffuse_mean_intensity=jnp.full((2, lines), 2e-8))
    if quadrature == "optical_depth":
        options["line_diffuse_field"] = dict(frequency_hz=frequency,
            incoming=jnp.full((1, 2, 40), 2e-8),
            optical_depth=jnp.full((1, 2, 40), .5),
            emitted=jnp.full((1, 2, 40), 1e-8),
            angular_weight=jnp.array([.5]))
    elif sampling == "average":
        options["line_node_diffuse_mean_intensity"] = (
            jnp.arange(1, 9)[:, None, None]*jnp.ones((8, 2, lines))*1e-9)

    def evaluate(opts, override=None):
        return model.evaluate(temperature, state, abundance,
            jnp.array([1e10, 2e10]), jnp.array([1e12, 2e12]),
            jnp.array([1e10, 2e10]), jnp.array([1e11, 2e11]),
            jnp.array([1e5, 2e5]), radiation, jnp.array([2.47e15]),
            line_mean_intensity_override=override, **opts)
    return model, options, evaluate


@pytest.mark.parametrize("sampling,quadrature", [
    ("front", "gauss"), ("average", "gauss"), ("average", "optical_depth")])
def test_complete_mean_replaces_every_point_field_and_keeps_photon_ledger(sampling, quadrature):
    model, options, evaluate = setup(sampling, quadrature)
    lines = len(model.line_frequency_hz)
    mask = jnp.ones(lines, dtype=bool)
    mean = jnp.zeros((2, lines))
    dark = dict(options, line_incident_mean_intensity=jnp.zeros(lines),
                line_diffuse_mean_intensity=mean)
    if "line_diffuse_field" in options:
        field = options["line_diffuse_field"]
        dark["line_diffuse_field"] = dict(field, incoming=field["incoming"]*0,
                                          emitted=field["emitted"]*0)
    if "line_node_diffuse_mean_intensity" in options:
        dark["line_node_diffuse_mean_intensity"] = options["line_node_diffuse_mean_intensity"]*0
    original = jax.jit(lambda: evaluate(options))()
    replaced = jax.jit(lambda: evaluate(options, (mask, mean)))()
    expected = jax.jit(lambda: evaluate(dark))()
    assert replaced["valid"] and expected["valid"]
    np.testing.assert_allclose(replaced["target_state"], expected["target_state"], rtol=2e-13)
    assert np.max(abs(np.asarray(replaced["target_state"]-original["target_state"]))) > 1e-5
    # Fixed populations/opacity own these events and kinetic exchanges. The
    # new mean only changes pumping, never adds a second OTS/thermal ledger.
    np.testing.assert_array_equal(replaced["ots_photon_flux_cm2_s"], original["ots_photon_flux_cm2_s"])
    for key in original["energy_erg_cm3_s"]:
        np.testing.assert_array_equal(replaced["energy_erg_cm3_s"][key], original["energy_erg_cm3_s"][key])
    unselected = jax.jit(lambda: evaluate(options,
        (jnp.zeros(lines, dtype=bool), jnp.full((2, lines), jnp.nan))))()
    np.testing.assert_array_equal(unselected["target_state"], original["target_state"])


def test_partial_mask_slices_atoms_and_preserves_unselected_fields():
    model, options, evaluate = setup("average", "gauss")
    lines = len(model.line_frequency_hz)
    mask = jnp.arange(lines) % 3 == 0
    mean = jnp.full((2, lines), 4e-8)
    changed = dict(options,
        line_incident_mean_intensity=jnp.where(mask, 100*options["line_incident_mean_intensity"],
                                               options["line_incident_mean_intensity"]),
        line_node_diffuse_mean_intensity=jnp.where(mask[None, None, :], 1e-3,
                                                   options["line_node_diffuse_mean_intensity"]))
    first = evaluate(options, (mask, mean))
    second = evaluate(changed, (mask, mean))
    np.testing.assert_array_equal(first["target_state"], second["target_state"])
    assert not evaluate(options, (mask, mean.at[:, 0].set(-1.)))["valid"]
    assert not evaluate(options, (mask, mean.at[:, 0].set(jnp.nan)))["valid"]
    with pytest.raises(ValueError, match="boolean"):
        evaluate(options, (mask.astype(float), mean))
    with pytest.raises(ValueError, match="cell,line"):
        evaluate(options, (mask, mean[0]))


def test_optical_depth_override_derivative_reaches_local_population_solution():
    model, options, evaluate = setup("average", "optical_depth")
    lines = len(model.line_frequency_hz)
    mask = jnp.arange(lines) % 2 == 0
    def observable(theta):
        result = evaluate(options, (mask, jnp.full((2, lines), 3e-8)*jnp.exp(theta)))
        return jnp.sum(result["target_state"][:, 2:])
    derivative = jax.jit(jax.grad(observable))(0.)
    step = 1e-5
    finite = (observable(step)-observable(-step))/(2*step)
    assert np.isfinite(derivative) and abs(derivative) > 1e-4
    np.testing.assert_allclose(derivative, finite, rtol=2e-5)
