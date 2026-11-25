"""Controlled observation effects, not a recovered publication reduction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.observation import C_KMS, process_phase_spectra

jax.config.update("jax_enable_x64", True)


def inputs(*, shifts=(-50., 0., 50.), phases=(.3, .375, .45), nights=(0, 0, 0)):
    velocity = jnp.linspace(-400., 400., 1601)
    stellar = jnp.linspace(-220., 220., 1761)
    output = jnp.linspace(-80., 80., 321)
    count = len(phases)
    normalization = jnp.broadcast_to((jnp.abs(stellar) > 180).astype(float), (count, stellar.size))
    return dict(
        model_phase=jnp.array([.3, .45]), model_velocity_kms=velocity,
        model_contrast=jnp.zeros((2, velocity.size)), exposure_phase=jnp.array(phases),
        orbital_velocity_kms=jnp.array(shifts), night_index=jnp.array(nights),
        stellar_velocity_kms=stellar, output_velocity_kms=output,
        stellar_reference=jnp.ones(stellar.size), continuum_weights=normalization,
        master_weights=jnp.ones((count, stellar.size)), stellar_keep=jnp.ones((count, stellar.size), dtype=bool),
        coadd_weights=jnp.ones((count, output.size)),
    )


def gaussian(args, amplitude=1e-3, width=8.):
    return amplitude * jnp.exp(-.5 * (args["model_velocity_kms"] / width)**2)


def test_constant_continuum_and_stationary_stellar_line_divide_out():
    args = inputs()
    stellar = args["stellar_velocity_kms"]
    args["stellar_reference"] = (1 - .5 * jnp.exp(-.5 * (stellar / 30)**2))[None, :] * jnp.array([.8, 1., 1.2])[:, None]
    result = process_phase_spectra(**args)
    assert result.valid.all()
    np.testing.assert_allclose(result.contrast, 0., atol=5e-16)


def test_moving_line_is_partly_self_subtracted_by_nightly_master():
    args = inputs()
    profile = gaussian(args)
    args["model_contrast"] = jnp.stack((profile, profile))
    result = process_phase_spectra(**args)
    center = np.argmin(np.abs(args["output_velocity_kms"]))
    # Other exposures contribute shifted line copies to the same master.
    assert .64e-3 < float(result.contrast[center]) < .68e-3
    assert float(result.contrast.min()) < -1e-4
    excluded = process_phase_spectra(**{**args, "master_weights": args["master_weights"].at[0].set(0.)})
    # Excluding the first exposure from its master removes its own central imprint.
    np.testing.assert_allclose(excluded.exposure_contrast[0, center], 1e-3, rtol=2e-3)
    args["orbital_velocity_kms"] = jnp.zeros(3)
    stationary = process_phase_spectra(**args)
    np.testing.assert_allclose(stationary.contrast, 0., atol=5e-16)


def test_nights_do_not_share_a_master_and_coadd_selection_does_not_remove_references():
    args = inputs()
    profile = gaussian(args)
    args["model_contrast"] = jnp.stack((profile, profile))
    args["coadd_weights"] = args["coadd_weights"].at[1:].set(0.)
    result = process_phase_spectra(**args)
    assert float(result.contrast.max()) > 6e-4
    args["night_index"] = jnp.arange(3)
    separate = process_phase_spectra(**args)
    # A singleton night divides by its own spectrum, even though other nights exist.
    np.testing.assert_allclose(separate.contrast, 0., atol=5e-16)


def test_mask_and_phase_dependence_make_an_asymmetric_coadd():
    args = inputs(shifts=(140., 110., 80.))
    profile = gaussian(args, width=25.)
    args["model_contrast"] = jnp.stack((.4 * profile, 1.6 * profile))
    unmasked = process_phase_spectra(**args)
    args["stellar_keep"] = jnp.broadcast_to(jnp.abs(args["stellar_velocity_kms"]) >= 110., args["stellar_keep"].shape)
    masked = process_phase_spectra(**args)
    velocity = np.asarray(args["output_velocity_kms"])
    red, blue = np.argmin(abs(velocity - 20)), np.argmin(abs(velocity + 20))
    assert masked.valid[red] and masked.valid[blue]
    assert masked.weight[red] > masked.weight[blue]
    assert abs(float(masked.contrast[red] - masked.contrast[blue])) > 2e-4
    assert not np.allclose(masked.contrast, unmasked.contrast, equal_nan=True)


def test_doppler_composition_phase_interpolation_and_explicit_pixel_weights():
    args = inputs(shifts=(80., 120., 160.), phases=(.3, .375, .45))
    # Exactly linear model makes the two interpolation steps analytically testable.
    velocity = args["model_velocity_kms"]
    args["model_contrast"] = jnp.stack((1e-5 * velocity, 2e-5 * velocity))
    result = process_phase_spectra(**args)
    stellar = np.asarray(args["stellar_velocity_kms"])
    slopes = np.array([1., 1.5, 2.]) * 1e-5
    rest = C_KMS * ((1 + stellar[None, :] / C_KMS) / (1 + np.array([80., 120., 160.])[:, None] / C_KMS) - 1)
    raw = 1 + slopes[:, None] * rest
    normalization = np.asarray(args["continuum_weights"])
    normalized = raw / ((raw * normalization).sum(1) / normalization.sum(1))[:, None]
    np.testing.assert_allclose(result.normalized_stellar_flux, normalized, rtol=2e-15)
    np.testing.assert_allclose(result.nightly_master, np.broadcast_to(normalized.mean(0), normalized.shape), rtol=2e-15)
    args["coadd_weights"] = args["coadd_weights"].at[1:].set(0.)
    selected = process_phase_spectra(**args)
    np.testing.assert_allclose(selected.contrast, selected.exposure_contrast[0], rtol=1e-14)


def test_mask_holes_are_not_bridged_and_missing_support_is_visible():
    args = inputs(shifts=(.1, .1, .1))
    stellar = args["stellar_velocity_kms"]
    args["stellar_keep"] = jnp.broadcast_to(jnp.abs(stellar) > 2., args["stellar_keep"].shape)
    result = process_phase_spectra(**args)
    center = np.argmin(np.abs(args["output_velocity_kms"]))
    assert not result.valid[center] and result.weight[center] == 0
    assert np.isnan(result.contrast[center])
    args["exposure_phase"] = jnp.array([.1, .2, .5])
    unsupported = process_phase_spectra(**args)
    assert not unsupported.valid.any() and np.isnan(unsupported.contrast).all()


def test_complete_operator_gradients_match_finite_differences():
    args = inputs(shifts=(37.2, 68.3, 100.4))
    args["stellar_keep"] = jnp.broadcast_to(jnp.abs(args["stellar_velocity_kms"]) > 20., args["stellar_keep"].shape)
    output = args["output_velocity_kms"]
    test_weight = jnp.exp(-.5 * ((output - 12) / 25)**2)

    def observable(parameters):
        amplitude, width, offset = parameters
        profile = gaussian(args, amplitude=amplitude * 1e-3, width=width * 12.)
        result = process_phase_spectra(**{**args, "model_contrast": jnp.stack((profile, 1.3 * profile)),
                                           "orbital_velocity_kms": args["orbital_velocity_kms"] + offset})
        return jnp.sum(test_weight * result.contrast)

    point = np.array([1., 1., .03])
    automatic = np.asarray(jax.grad(observable)(point))
    step = 1e-4
    finite = np.array([(observable(point + step * axis) - observable(point - step * axis)) / (2 * step)
                       for axis in np.eye(3)])
    assert np.isfinite(automatic).all() and np.all(np.abs(automatic) > 1e-8)
    np.testing.assert_allclose(automatic, finite, rtol=3e-5, atol=1e-10)


def test_bad_shapes_raise_and_bad_weights_do_not_silently_pass():
    args = inputs()
    with pytest.raises(ValueError, match="integer night"):
        process_phase_spectra(**{**args, "night_index": jnp.zeros(3)})
    result = process_phase_spectra(**{**args, "master_weights": -args["master_weights"]})
    assert not result.valid.any()
    result = process_phase_spectra(**{**args, "model_velocity_kms": args["model_velocity_kms"][::-1]})
    assert not result.valid.any()
