"""The external full-Cloudy observer keeps the native atom limit explicit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    spec = importlib.util.spec_from_file_location("cloudy_halpha", EXAMPLES / "cloudy_halpha.py")
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
finally:
    sys.path.pop(0)


def test_packed_population_adapter_keeps_low_levels_and_names_the_high_shells():
    native = np.arange(36.).reshape(2, 18)
    kept, principal, tail = observer.split_observer_populations(native)
    np.testing.assert_array_equal(kept, native)
    assert principal.size == 0 and tail.shape == (2, 0)
    full = np.arange(56.).reshape(2, 28)
    kept, principal, tail = observer.split_observer_populations(full)
    np.testing.assert_array_equal(kept, full[:, :23])
    np.testing.assert_array_equal(principal, np.arange(21, 26))
    np.testing.assert_array_equal(tail, full[:, 23:])


def test_external_tail_obeys_kirchhoff_with_native_milne_units():
    frequency = np.array([4.56e14, 4.57e14])
    temperature = np.array([6000., 8000.])
    electron, proton = np.full(2, 1e9), np.full(2, 3e8)
    h, k, c, me = 6.62607015e-27, 1.380649e-16, 2.99792458e10, 9.1093837015e-28
    shell = np.arange(21, 26)
    volume = (h*h/(2*np.pi*me*k))**1.5
    threshold = 13.5984345*1.602176634e-12/shell**2
    bound = np.ones((2, 28))
    bound[:, 23:] = (electron*proton)[:, None]*shell[None, :]**2*volume/temperature[:, None]**1.5*np.exp(
        threshold[None, :]/(k*temperature[:, None]))
    profile = dict(bound_cm3=bound, temperature_k=temperature, electron_cm3=electron,
                   n_h_cm3=proton, ionized_fraction=np.ones(2))
    fixture = dict(frequency_hz=frequency, principal_n=np.arange(20, 26),
                   cross_section_cm2=np.full((6, 2), 1e-21))
    extinction, emissivity = observer.high_n_continuum(profile, frequency, fixture)
    blackbody = 2*h/c**2*frequency[None, :]**3/np.expm1(
        h*frequency[None, :]/(k*temperature[:, None]))
    assert np.all(extinction > 0)
    np.testing.assert_allclose(emissivity/extinction, blackbody, rtol=2e-14)


def test_raw_projection_retains_resolved_peak_and_records_only_boundary_stubs():
    projected = dict(radius_cm=np.array([1., 1.2, 1.5, 2.]),
                     total_heating_cooling_ratio=np.array([.8, 1., 1.2, 1.5]))
    raw = dict(radius_cm=np.array([1.1, 1.4, 1.8]), n2_cm3=np.array([10., 100., 20.]),
               principal_n=np.array([1, 2, 2, 3, 3, 3]))
    prescribed = dict(radius_cm=projected["radius_cm"], velocity_cm_s=np.array([100., 120., 150., 200.]))
    data, receipt = observer.raw_observer_profiles(projected, raw, prescribed)
    np.testing.assert_allclose(data["radius_cm"], [1., 1.1, 1.4, 1.8, 2.])
    np.testing.assert_allclose(data["n2_cm3"], [10., 10., 100., 20., 20.])
    np.testing.assert_allclose(data["velocity_cm_s"], [100., 110., 140., 180., 200.])
    np.testing.assert_allclose(data["principal_n"], raw["principal_n"])
    np.testing.assert_allclose([receipt["inner_endpoint_stub_cm"], receipt["outer_endpoint_stub_cm"]], [.1, .2])
