"""Compare hydrogen heat ledgers using C23.01's process grouping.

This external diagnostic supplies no physical rates or radiation field. Input
powers must already refer to the same populations, transfer closure and units.
In particular, a thin recombination rate is not interchangeable with a net
escaping rate unless the corresponding local photoheating is also accounted
for. This grouping never changes the net kinetic energy exchange.
"""

from __future__ import annotations

import numpy as np


def hydrogen_heat_groups(*, ground_photo, excited_photo, spontaneous_recombination,
                         induced_recombination, collisional_ionization,
                         three_body_recombination, line_net):
    """Group powers in erg cm^-3 s^-1; only line_net may be negative.

    C23 iso_cool.cpp nets CI against three-body heating, all bound-bound
    collisions within the atom, and excited photoheating against spontaneous
    recombination cooling. Ground photoheating and induced recombination
    remain separate. The caller must match the retained upper-level treatment
    and continuum escape convention before interpreting agreement physically.
    Scalars or arrays with a common broadcastable shape are accepted.
    """
    positive = np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in (
        ground_photo, excited_photo, spontaneous_recombination,
        induced_recombination, collisional_ionization, three_body_recombination)])
    line = np.asarray(line_net, dtype=float)
    if any(np.any(~np.isfinite(value)) or np.any(value < 0) for value in positive):
        raise ValueError("Unsigned process powers must be finite and nonnegative.")
    if np.any(~np.isfinite(line)):
        raise ValueError("Net line power must be finite.")
    ground, excited, spontaneous, induced, ionization, three_body = positive
    groups = dict(ground_photo=ground, excited_photo_minus_rr=excited-spontaneous,
                  induced_recombination=-induced,
                  three_body_minus_ci=three_body-ionization, line=line)
    powers = np.stack(np.broadcast_arrays(*groups.values()))
    heating = np.maximum(powers, 0).sum(axis=0)
    cooling = -np.minimum(powers, 0).sum(axis=0)
    return dict(groups=groups, heating=heating, cooling=cooling,
                net=powers.sum(axis=0))
