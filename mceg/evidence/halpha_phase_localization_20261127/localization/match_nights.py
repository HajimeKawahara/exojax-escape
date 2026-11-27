"""Replay the common-phase night check without changing the sealed phase maps.

Run from the repository with PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src.
The window is set by overlapping complete-core support, not by the residuals.
"""
import json
from pathlib import Path
import sys

import numpy as np

repository = next(path for path in Path(__file__).resolve().parents if (path / 'examples/halpha_phase_localization.py').is_file())
sys.path.insert(0, str(repository / 'examples'))
from halpha_phase_localization import construct_maps, values_from_maps, sha256, WINDOWS, SCOPES

raw_directory = Path('/tmp/escape-halpha-phase-localization')
output = Path(__file__).resolve().parent
original_maps = raw_directory / 'maps.npz'
original_rows = raw_directory / 'rows.json'
source = repository / 'examples/halpha_phase_localization.py'
noise_path = Path('/tmp/escape-gp-nuts-20261125/gp/samples.npz')
display_path = Path('/tmp/escape-gp-combined-20261125/combined_arrays.npz')
pins = {str(path): sha256(path) for path in [Path(__file__), original_maps, original_rows, source, noise_path, display_path]}
saved = dict(np.load(original_maps))
original = json.loads(original_rows.read_text())
noise = {key: np.median(value, axis=(0, 1)) for key, value in np.load(noise_path).items()}
edges = np.load(display_path)['edges']
lower, upper = .40320277222508594, .4180600419917937
rows = [dict(row_id=i, grid='matched_nights', product='band_complete', scope=scope,
             phase_index=0, phase_low=lower, phase_high=float(np.nextafter(upper, np.inf)), band=band)
        for i, (scope, band) in enumerate((scope, band) for scope in SCOPES for band in WINDOWS)]
arrays = {key: value for key, value in saved.items() if not key.endswith('__maps') and not key.endswith('__raw_maps')}
selections = {}
for selection in ['primary', 'alternate']:
    groups, fields = [], []
    for i in range(4):
        prefix = f'{selection}__g{i}__'
        group = {key: saved[prefix + key] for key in ['phase', 'stellar_velocity', 'pixel_index', 'weight', 'science', 'eclipse', 'exposure_id', 'time_min', 'log_data', 'model']}
        group.update(night=i // 2, order=i % 2, label=f'night{i // 2}_order{i % 2}')
        groups.append(group)
        fields.append(dict(observed=group['log_data'], planet=group['model'], gp=saved[prefix + 'gp'],
                           residual=group['log_data'] - group['model'] - saved[prefix + 'gp'],
                           white_scale=float(np.exp(noise['log_white'][i]))))
    maps, raw, metadata, _ = construct_maps(groups, fields, rows, edges)
    values = values_from_maps(groups, fields, maps, metadata)
    for i, (mapping, coefficient) in enumerate(zip(maps, raw)):
        arrays[f'{selection}__g{i}__maps'] = mapping
        arrays[f'{selection}__g{i}__raw_maps'] = coefficient
    night_ids = [next(row['row_id'] for row in rows if row['scope'] == f'night{n}' and row['band'] == 'core') for n in [0, 1]]
    selections[selection] = dict(rows=metadata, values_ppm=values,
        core_night1_minus_night0=dict(row_ids=night_ids,
            values_ppm={key: values[night_ids[1]][key] - values[night_ids[0]][key] for key in ['observed', 'planet', 'gp', 'residual']}))
maps_path = raw_directory / 'matched_nights_maps.npz'
np.savez_compressed(maps_path, **arrays)
report = dict(schema_version=1, scope='Band-complete night/order comparison in the same support-defined phase window. Original full-data native baseline, reference and conditional GP are unchanged. This is not an intrinsic absolute night-to-night line measurement.',
    phase_window_inclusive=[lower, upper], upper_edge_implementation='nextafter(upper,+inf) to include the support endpoint',
    parameters=original['parameters'], rows=rows, selections=selections,
    inputs=pins, raw_maps=dict(path=str(maps_path), sha256=sha256(maps_path)),
    limitations=['Different nights have separate full-data common references. Matching displayed phase limits does not remove that reference dependence.',
                 'The matched window still has no complete blue band on either night.',
                 'Noise/control sensitivities must use these maps with the original full-data groups, not retrain the reference or GP on only the matched exposures.'])
assert all(sha256(path) == expected for path, expected in pins.items())
(output / 'match_nights.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({selection: value['core_night1_minus_night0'] for selection, value in selections.items()}, indent=2))
