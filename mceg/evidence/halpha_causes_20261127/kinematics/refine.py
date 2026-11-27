"""Replay numerical controls and figures for the frozen kinematic probes."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import voigt_profile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'examples'))
import halpha_cause_kinematics as diagnostic
from halpha_geometry_fit import PHASE, sha256

RAW = Path('/tmp/escape-halpha-cause-kinematics')
OUTPUT = Path(__file__).resolve().parent
ARCHIVE = Path('/home/kawahara/exojax-escape/results/contrast_response_20261126')


def render_los(index):
    """Fine-grid upper +5 km/s observer-LOS shift; frozen alpha and eta."""
    import escape.longitude_map as geometry
    assert sha256(Path(geometry.__file__)) == sha256(Path('/tmp/escape-phase-night-pr/src/escape/longitude_map.py'))
    sys.path.insert(0, str(ARCHIVE))
    renderer = importlib.import_module('render_response')
    branch = renderer.A
    dx = .05
    table_velocity = np.arange(-250., 250. + dx / 2, dx)
    table = np.stack([voigt_profile(table_velocity[None] * branch['branch_centers'][:, None] / renderer.C,
        branch['branch_sigma'][i, :, None], branch['branch_gamma'][:, None]) for i in range(8)])
    offset = renderer.C * (renderer.WN[None] / branch['branch_centers'][:, None] - 1)
    packed = list(renderer.pack_rays(PHASE[index], renderer.PLAN['grid']))
    packed[4] = packed[4] + np.where(renderer.INNER[packed[1]], 0., 5.)
    assert np.max(abs(offset)) + np.max(abs(packed[3]) + abs(packed[4])) < 249.
    raw = renderer.integrate(*packed, table, table_velocity[0], dx, offset,
        branch['branch_alpha'], branch['branch_eta'], renderer.F['continuum_extinction'],
        renderer.F['continuum_emission_hz'] * renderer.C * 1e5,
        renderer.BOUNDARY, renderer.INNER, 1., 1.)
    spectrum = np.sum([.08**2 * np.convolve(row, renderer.KERNEL, mode='same') / renderer.STAR for row in raw], axis=0)[::-1]
    destination = RAW / f'los_+5_fine_phase_{index:02d}.npz'
    np.savez_compressed(destination, spectrum=spectrum, phase=PHASE[index])
    paths = [Path(__file__), ARCHIVE / 'render_response.py', Path(geometry.__file__),
             Path('/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz'), Path('/tmp/escape-e8-mdot-sphere/fine/branch_arrays.npz')]
    destination.with_suffix('.json').write_text(json.dumps(dict(inputs={str(p):sha256(p) for p in paths}, raw_sha256=sha256(destination)), indent=2)+'\n')


def controls():
    review = json.loads((OUTPUT / 'review.json').read_text())
    original = dict(np.load('/tmp/escape-e8-mdot-sphere/fit_arrays.npz'))
    baseline = original['fine_intrinsic_phases'][:7]
    cached = np.load(RAW / 'arrays.npz')
    maps = np.load('/tmp/escape-halpha-phase-localization/maps.npz')
    rows = json.loads(Path('/tmp/escape-halpha-phase-localization/rows.json').read_text())['rows']
    noise = {k:np.median(v,axis=(0,1)) for k,v in np.load('/tmp/escape-gp-nuts-20261125/gp/samples.npz').items()}
    paths = [Path(f'/tmp/escape-observation-processing/KELT9b_2024-08-{d}_line_w6565.h5') for d in ['08','20']]
    surface = dict(np.load('/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz'))
    velocity = original['model_velocity_kms']
    ev = diagnostic.prepare_evaluator('primary',0,PHASE,velocity,paths,surface,noise,maps,rows)
    target = [i for i,r in enumerate(ev['row_ids']) if rows[r]['scope']=='all' and rows[r]['product']=='common_complete' and rows[r]['phase_index']==3]
    records = {}
    def compare(name, first, second, second_ev=None):
        other = ev if second_ev is None else second_ev
        a,b = ev['evaluate'](first),other['evaluate'](second)
        ma = ev['problem']['models'](first*np.exp(diagnostic.PARAMETERS[1]),diagnostic.PARAMETERS[0])
        mb = other['problem']['models'](second*np.exp(diagnostic.PARAMETERS[1]),diagnostic.PARAMETERS[0])
        maximum = max(float(np.max(abs(np.asarray(x)-np.asarray(y))*np.sqrt(g['weight']))) for x,y,g in zip(ma,mb,ev['problem']['groups']))
        records[name] = dict(Q_first=a['Q'],Q_second=b['Q'],Q_second_minus_first=b['Q']-a['Q'],
            target_residual_second_minus_first_BCR_ppm=(b['residual'][target]-a['residual'][target]).tolist(),
            maximum_native_change_in_archived_white_sigma=maximum)
        return a,b
    bank,_ = diagnostic.longitude_bank('/tmp/escape-longitude-complete-20261125/fine')
    coarse,_ = diagnostic.longitude_bank('/tmp/escape-longitude-complete-20261125/coarse')
    fitted = review['cases']['all_data_longitude_fit']['parameters']['coefficients']
    for name,kind,value in [('patch_+30_9','patch',[30.,9.]),('substellar_+2','substellar',2.),('all_data_longitude_fit','fitted_map',fitted)]:
        weights = diagnostic.positive_map(bank['longitude'],kind,value)
        full = cached[name+'__spectrum']
        compare(name+'_ray_grid',full,baseline+np.einsum('plv,l->pv',coarse['kernel'],weights)-coarse['kernel'].sum(axis=1))
        node49 = diagnostic.positive_map(bank['longitude'][::2],kind,value)
        compare(name+'_longitude_nodes',full,np.einsum('plv,l->pv',bank['coarse_kernel'],node49))
    midpoint = [dict(np.load(p)) for p in sorted(Path('/tmp/escape-longitude-complete-20261125/midpoint').glob('phase_*.npz'))[:6]]
    midphase = np.array([float(z['phase']) for z in midpoint])
    phase13 = np.sort(np.r_[PHASE,midphase])
    ev13 = diagnostic.prepare_evaluator('primary',0,phase13,velocity,paths,surface,noise,maps,rows)
    for name,kind,value in [('patch_+30_9','patch',[30.,9.]),('all_data_longitude_fit','fitted_map',fitted)]:
        weights=diagnostic.positive_map(bank['longitude'],kind,value)
        spectrum=np.einsum('plv,l->pv',coarse['kernel'],weights)
        mids=np.array([np.einsum('lv,l->v',z['kernel'],weights) for z in midpoint])
        refined=np.array([spectrum[np.flatnonzero(abs(PHASE-p)<1e-12)[0]] if np.any(abs(PHASE-p)<1e-12) else mids[np.flatnonzero(abs(midphase-p)<1e-12)[0]] for p in phase13])
        compare(name+'_phase_sampling_same_coarse_rays',spectrum,refined,ev13)
    radial_coarse=np.array([np.load(RAW/f'radial_+13_coarse_phase_{i:02d}.npz')['spectrum'] for i in range(7)])
    compare('radial_+13_ray_grid',cached['upper_radial_add_+13__spectrum'],baseline+radial_coarse-original['coarse_intrinsic_phases'][:7])
    los=np.array([np.load(RAW/f'los_+5_fine_phase_{i:02d}.npz')['spectrum'] for i in range(7)])
    _,los_fine=compare('upper_LOS_+5_ray_grid',cached['upper_LOS_+5__spectrum'],los)
    baseline_record=ev['evaluate'](baseline)
    records['upper_LOS_+5_fine_result']=dict(fixed=diagnostic.summarize(los_fine,baseline_record,ev['row_ids'],rows),
        profiled=diagnostic.summarize(ev['profile'](los),ev['profile'](baseline),ev['row_ids'],rows))
    parity={grid:float(np.max(abs(np.load(RAW/f'radial_+0_{grid}_phase_03.npz')['spectrum']-original[grid+'_intrinsic_phases'][3]))) for grid in ['coarse','fine']}
    # The independent table interpolator differs from the archived kernels by
    # 0.0066 ppm; recomputed alpha/eta agree with saved branches at 2e-16 relative.
    # This is a numerical reproduction check, not bitwise identity.
    assert max(parity.values())<1e-8
    inputs=[Path(__file__),Path(diagnostic.__file__),OUTPUT/'review.json',RAW/'arrays.npz',*RAW.glob('radial_*.npz'),*RAW.glob('los_*.npz')]
    (OUTPUT/'controls.json').write_text(json.dumps(dict(scope='Fixed-nuisance native GP/map controls. Phase refinement uses identical coarse ray geometry; it is not a second independent fine-ray convergence test.',
        zero_radial_increment_maximum_absolute_spectrum_difference=parity,
        zero_radial_increment_acceptance_absolute=1e-8,
        zero_radial_increment_interpretation='Independent table-renderer reproduction within 0.01 ppm; not bitwise parity. Recomputed alpha and eta match the saved branch coefficients to 2e-16 relative.',comparisons=records,
        inputs={str(p):sha256(p) for p in inputs}),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-los-phase',type=int)
    args=parser.parse_args()
    if args.render_los_phase is not None:
        render_los(args.render_los_phase)
    else:
        controls()
