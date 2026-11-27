"""A deliberately limited optically thick projected-area diagnostic.

This is not a renderer, a line-profile prediction or an inferred photosphere.
It tests why an extended absorbing shell need not dim the unresolved source.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def run():
    here = Path(__file__).resolve().parent
    path = here.parent / 'halpha_causes_20261127/sources/free_shape.json'
    shape = json.loads(path.read_text())
    physical, free = shape['physical'], shape['free']['original']
    old_r = physical['static_center_tau_one_radius_rp']
    new_r = free['static_center_tau_one_radius_rp']
    # Piecewise constant local coefficient in the cell containing radial tau=1.
    radii, widths = np.array(physical['radius_rp']), np.array(physical['thickness_rp'])
    index = np.flatnonzero((old_r >= radii-widths/2) & (old_r < radii+widths/2))
    assert len(index) == 1
    old_s = physical['rest_center_source'][int(index[0])]
    free_centers, free_width = np.array(free['full_radius_rp']), free['layer_thickness_rp']
    new_index = np.flatnonzero((new_r >= free_centers-free_width/2) & (new_r < free_centers+free_width/2))
    assert len(new_index) == 1
    new_s = free['full_source'][int(new_index[0])]
    area_ratio = (new_r/old_r)**2
    result = dict(
        scope='Uniform optically thick disk approximation F_nu proportional to R_eff^2 S_nu; no spectrum or fitted flux prediction.',
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_radii_rp=[old_r,new_r],
        local_source_per_wavenumber=[old_s,new_s],
        projected_area_ratio=float(area_ratio),
        source_ratio=float(new_s/old_s),
        approximate_area_times_source_ratio=float(area_ratio*new_s/old_s),
        source_required_for_smaller_area_times_source=float(old_s/area_ratio),
        example_covered_emitter_radius_rp=1.3,
        thick_full_shell_dim_condition_S_over_Iinner_lt=float((1.3/new_r)**2),
        physical_condition='A pure absorbing foreground patch can remove a core; a whole emitting shell also adds projected halo emission. Both need a population and photon budget.',
        limitations=[
            'The radial tau=1 locations are static nominal-center diagnostics, not actual observer formation radii.',
            'The actual observer has a dayside, occultation, ray-dependent optical depths, three branches, rotation/winds, continuum and observation processing.',
            'Model 1 originally has plane columns and no off-limb emission. Applying a disk-area proxy to its radial profile does not convert it into the spherical model.',
            'The proxy is a counterexample to automatic dimming, not a quantitative correction or rejection of the extended-absorber hypothesis.'])
    assert result['projected_area_ratio'] > 1
    assert result['approximate_area_times_source_ratio'] > 1
    here.joinpath('geometry_budget.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('projected_area_ratio','source_ratio','approximate_area_times_source_ratio','thick_full_shell_dim_condition_S_over_Iinner_lt')},indent=2))


if __name__ == '__main__':
    run()
