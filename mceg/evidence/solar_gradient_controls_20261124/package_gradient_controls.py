"""Retain completed derivative and observer controls without private inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

import numpy as np

scratch = Path('/tmp/escape-solar-molecular-continuation')
out = Path('/tmp/escape-thermal-gradient-controls-evidence')
out.mkdir(exist_ok=False)

def copy(source, destination):
    target = out/destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

cases = {
    'fixed_original': 'current5_fixed_temperature_derivative',
    'fixed_refined': 'current5_fixed_derivative_refinement',
    'thermal_interrupted': 'current5_thermal_derivative',
    'thermal_resumed': 'current5_thermal_derivative_resumed',
}
for destination, source in cases.items():
    for path in sorted((scratch/source).iterdir()):
        if path.suffix in ('.json', '.npz'):
            copy(path, Path(destination)/path.name)
    copy(scratch/(source+'.log'), Path(destination)/'execution.log')
for name in ('audit_current5_implicit.py', 'audit_current5_implicit_fast_observer.py',
             'audit_current5_implicit_refinement.py', 'resume_current5_thermal_derivative.py'):
    copy(scratch/name, Path('executed_drivers')/name)
copy(scratch/'source_current_flow_gradient/manifest.json', 'source_manifest.json')
copy(scratch/'current5_flow_gradient/configuration.json', 'configuration.json')
copy('/tmp/escape-fast-observer-parity/receipt.json', 'observer_parity.json')

checks = {}
for name, directory, implicit, step in (
    ('fixed_original', 'fixed_original', 'fixed_original', .001),
    ('fixed_refined', 'fixed_refined', 'fixed_refined', .00025),
    ('thermal_resumed', 'thermal_resumed', 'thermal_interrupted', .00025),
):
    plus = np.load(out/directory/f'finite_+{step:.6f}.npz')['contrast']
    minus = np.load(out/directory/f'finite_-{step:.6f}.npz')['contrast']
    tangent = np.load(out/implicit/'implicit_spectrum.npz')['derivative']
    finite = (plus-minus)/(2*step)
    error = float(np.max(abs(tangent-finite))/np.max(abs(finite)))
    summary = json.loads((out/directory/'summary.json').read_text())
    np.testing.assert_allclose(error, summary['maximum_derivative_relative_error'], rtol=1e-12)
    checks[name] = dict(recomputed_spectral_error=error, threshold=1e-4,
                        passed=bool(error<1e-4))
assert not checks['fixed_original']['passed']
assert checks['fixed_refined']['passed'] and checks['thermal_resumed']['passed']

observer = Path('/tmp/escape-native-thermal-observer-refinements')
copy(observer/'summary.json', 'observer/summary.json')
copy('/tmp/refine_thermal_observer.py', 'observer/executed_refinement.py')
copy('/tmp/observe_native_solar_root.py', 'observer/observe_native_solar_root.py')
copy('/tmp/escape-native-current5-thermal-observer/derived_spectrum.npz',
     'observer/baseline/derived_spectrum.npz')
copy('/tmp/escape-native-current5-thermal-observer/receipt.json', 'observer/baseline/receipt.json')
baseline = np.load(out/'observer/baseline/derived_spectrum.npz')
velocity = baseline['observed_velocity_kms']
prediction = np.interp(velocity, baseline['output_velocity_kms'], baseline['processed'])
observer_summary = json.loads((observer/'summary.json').read_text())
for row in observer_summary['controls']:
    name = row['label']
    for filename in ('derived_spectrum.npz', 'receipt.json'):
        copy(observer/name/filename, Path('observer')/name/filename)
    copy(observer/(name+'.log'), Path('observer')/name/'execution.log')
    data = np.load(out/'observer'/name/'derived_spectrum.npz')
    difference = np.interp(velocity,data['output_velocity_kms'],data['processed'])-prediction
    error = float(np.max(abs(difference)/baseline['supplied_error']))
    np.testing.assert_allclose(error,row['maximum_difference_over_supplied_error'],rtol=1e-12)
    assert error < .1
    checks['observer_'+name] = dict(recomputed_maximum_supplied_errors=error,threshold=.1,passed=True)
copy('/tmp/escape-native-thermal-observer-evidence/ORIGINAL_OBSERVATION_NOTICE.txt',
     'observer/ORIGINAL_OBSERVATION_NOTICE.txt')
(out/'certificate_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(out/'README.md').write_text('''# Full thermal sensitivity and fixed-atmosphere observer refinement

The original fixed-temperature derivative failed the unchanged 1e-4
observable criterion. Tightening only its implicit linear solve from 1e-7
to 1e-10, while retaining the original independently re-solved finite
differences, reduces the error from 2.5951e-4 to 1.3832e-6. A separate
smaller finite step gives 7.7412e-7. This is a conditioned-observable
accuracy control, not a claim of global smoothness across all rate guards.

At the full five-cell thermal/radiation/composition/wind root, the exact
implicit spectral JVP agrees with two independently re-solved perturbed
roots to 2.3939e-6. The true sensitivity equation residual is 4.6131e-11;
both finite-difference roots satisfy F-infinity below 2.02e-12. A saved
implicit tangent initializes those solves but does not determine their
acceptance. The original process terminated with exit 143 and no recorded
numerical exception; its cause is unknown. Its products and log are kept
alongside the resumed calculation and its independently checked products.

These are host-orchestrated exact JAX-product controls at one mass-loss
point, not a completed production reverse-mode or NUTS-domain test.
The coarse actual observer used for the derivative has 101 frequencies
and 2 by 2 surface nodes. Its fast evaluation is bitwise identical in
spectrum and directional derivative to the original complete evaluation.

Separate longitude, latitude, frequency and phase refinements and their
joint refinement all pass 0.1 supplied observation errors at the unchanged
full thermal root. Their largest change is 1.0225 ppm, 0.00547 supplied
errors. This does not test radial, atomic or radiation-grid convergence,
nor establish the exact unpublished observational reduction.

The included arrays and receipts are from executed controls. The packaging
script recomputes their reported finite-difference and observer errors;
it does not rerun the atmosphere equations. Original drivers retain their
executed paths. For a new physical replay restore the exact source manifest,
the root/input metadata in evidence/solar-thermal-root-20261124 and the
retrieved PHOENIX input with its recorded SHA256. Restore ExoJAX revision
38cb2e44f153ad6d729e3253e235702de52920cf. Preconditioner factors were
numerical accelerators and are not redistributed: regenerate a checked
inverse approximation or solve the original residual equations directly.
No private stellar samples, compiler caches, binaries or duplicate atomic
tables are included. The observed data are attributed under CC BY 4.0 in
observer/ORIGINAL_OBSERVATION_NOTICE.txt. No extra plots duplicate the
existing thermal observer figure archive.
''')
inventory = {}
for path in sorted(out.rglob('*')):
    if not path.is_file():
        continue
    row = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    if path.suffix == '.npz':
        with np.load(path,allow_pickle=False) as data:
            assert all(not data[key].dtype.hasobject for key in data.files)
            assert all(not any(term in key.lower() for term in
                ('surface_flux','stellar_reference','incident','boundary')) for key in data.files)
            row['fields'] = data.files
    inventory[str(path.relative_to(out))] = row
(out/'manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
archive = out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as packed:
    packed.add(out,arcname=out.name)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                     bytes=archive.stat().st_size,files=len(inventory)+1,checks=checks),indent=2))
