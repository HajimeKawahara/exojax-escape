"""Executed independent interface and legacy-display checks for phase maps."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

repository = next(path for path in Path(__file__).resolve().parents if (path / 'examples/halpha_phase_localization.py').is_file())
sys.path.insert(0, str(repository / 'examples'))
from halpha_data_consistency import evaluate, bands, load_groups, solve_group
from halpha_phase_localization import legacy_coefficients, project_reference

directory = Path('/tmp/escape-halpha-phase-localization')
report = json.loads((directory / 'rows.json').read_text())
archive = dict(np.load('/tmp/escape-e8-mdot-sphere/fit_arrays.npz'))
surface = dict(np.load('/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz'))
noise = {k: np.median(v, axis=(0, 1)) for k, v in np.load('/tmp/escape-gp-nuts-20261125/gp/samples.npz').items()}
display_edges = np.load('/tmp/escape-gp-combined-20261125/combined_arrays.npz')['edges']
paths = [f'/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5' for day in ['08', '20']]
arrays = np.load(directory / 'maps.npz')
checks = {}
for selection, offset in [('primary', 0), ('alternate', 1)]:
    groups, _ = load_groups(paths, archive, surface, offset=offset)
    fields = [solve_group(group, noise) for group in groups]
    _, profiles, _ = evaluate(groups, noise, display_edges)
    old = bands(profiles['all'])
    maximum_projection_error = 0.
    maximum_coarse_error = 0.
    coordinates = []
    for i, (group, field) in enumerate(zip(groups, fields)):
        prefix = f'{selection}__g{i}__'
        for key in ['phase', 'stellar_velocity', 'pixel_index', 'weight', 'science', 'eclipse', 'exposure_id', 'time_min', 'log_data', 'model']:
            np.testing.assert_array_equal(arrays[prefix + key], group[key])
        np.testing.assert_array_equal(arrays[prefix + 'gp'], field['gp'])
        mapping, raw = arrays[prefix + 'maps'], arrays[prefix + 'raw_maps']
        assert np.min(raw) >= 0
        weight = group['weight']
        common = weight / np.where(weight.sum(axis=0) > 0, weight.sum(axis=0), 1)
        expected = raw - raw.sum(axis=1)[:, None] * common[None]
        maximum_projection_error = max(maximum_projection_error, float(np.max(abs(mapping - expected))))
        for r, meta in enumerate(report['selections'][selection]['rows']):
            if not meta['supported']:
                assert np.count_nonzero(mapping[r]) == 0
        for row in report['rows']:
            if row['grid'] != 'coarse' or row['product'] == 'legacy_display':
                continue
            indices = [x['row_id'] for x in report['rows'] if x['grid'] == 'fine' and x['product'] == row['product'] and x['scope'] == 'all' and x['band'] == row['band'] and x['phase_low'] >= row['phase_low'] and x['phase_high'] <= row['phase_high']]
            counts = np.array([sum(report['selections'][selection]['rows'][j]['group_exposure_counts']) for j in indices])
            combined = np.einsum('r,rep->ep', counts, mapping[indices]) / max(counts.sum(), 1)
            maximum_coarse_error = max(maximum_coarse_error, float(np.max(abs(combined - mapping[row['row_id']]))))
        orbital = 244.8 * np.sin(2 * np.pi * group['phase'])
        coordinates.append(299792.458 * ((1 + group['stellar_velocity'][None] / 299792.458) / (1 + orbital[:, None] / 299792.458) - 1))
    legacy_errors = {}
    for band in ['blue', 'core', 'red']:
        row = dict(phase_low=.30, phase_high=.435, scope='all', band=band)
        raw = legacy_coefficients(groups, fields, coordinates, row, display_edges)
        maps = [project_reference(g, c) for g, c in zip(groups, raw)]
        legacy_errors[band] = {key: float(sum(np.sum(m * f[key]) for m, f in zip(maps, fields)) * 1e6 - old[band][key]) for key in ['observed', 'planet', 'gp', 'residual']}
    assert maximum_projection_error < 1e-14
    assert maximum_coarse_error < 1e-14
    assert max(abs(v) for row in legacy_errors.values() for v in row.values()) < 1e-8
    checks[selection] = dict(native_arrays_exactly_match_original_loader=True, frozen_GP_exact=True, unsupported_maps_zero=True,
        raw_coefficients_nonnegative=True, maximum_common_projection_difference=maximum_projection_error,
        coarse_maps_equal_exposure_weighted_fine_maps_max_difference=maximum_coarse_error,
        full_phase_legacy_parity_difference_ppm=legacy_errors)
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
output = repository / 'mceg/evidence/halpha_phase_localization_20261127/localization/controls.json'
output.write_text(json.dumps(dict(passed=True, checks=checks, input_sha256={str(p): digest(p) for p in [directory / 'maps.npz', directory / 'rows.json', Path(__file__), repository / 'examples/halpha_phase_localization.py']}), indent=2) + '\n')
print(output)
