"""The same microscopic energy flow can have different gross heat ledgers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).parents[1] / "tools/cloudy_thermal_bookkeeping.py"
SPEC = importlib.util.spec_from_file_location("cloudy_thermal_bookkeeping", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def test_reversible_cycles_change_gross_power_without_changing_net_energy():
    arguments = dict(ground_photo=3., excited_photo=9., spontaneous_recombination=7.,
                     induced_recombination=1., collisional_ionization=6.,
                     three_body_recombination=4., line_net=-1.)
    base = reference.hydrogen_heat_groups(**arguments)
    # Adding the same reversible power to both directions is a bookkeeping
    # identity, not an increase in physical thermal imbalance.
    cycled = reference.hydrogen_heat_groups(**dict(
        arguments, excited_photo=1e6+9., spontaneous_recombination=1e6+7.,
        collisional_ionization=1e6+6., three_body_recombination=1e6+4.))
    assert base["heating"] == 5.
    assert base["cooling"] == 4.
    assert base["net"] == 1.
    for key in ("heating", "cooling", "net"):
        np.testing.assert_array_equal(cycled[key], base[key])


def test_signed_channels_broadcast_and_invalid_power_is_rejected():
    arguments = dict(ground_photo=0., excited_photo=0., spontaneous_recombination=0.,
                     induced_recombination=0., collisional_ionization=0.,
                     three_body_recombination=0., line_net=np.array([-2., 3.]))
    result = reference.hydrogen_heat_groups(**arguments)
    np.testing.assert_array_equal(result["heating"], [0., 3.])
    np.testing.assert_array_equal(result["cooling"], [2., 0.])
    with pytest.raises(ValueError, match="nonnegative"):
        reference.hydrogen_heat_groups(**dict(arguments, ground_photo=-1.))
