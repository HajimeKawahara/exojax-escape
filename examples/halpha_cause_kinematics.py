"""Finite velocity and longitude-source probes of the phase-local H-alpha gap.

All atmospheres are frozen. Longitude maps use actual spherical source kernels;
upper velocity probes use saved finite-transfer spectra. Full-data GP covariance
and the complete-band phase maps are shared. Scores are descriptive, not a
discovery claim or a self-consistent flow/thermal solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from halpha_geometry_fit import setup, sha256, PHASE
from halpha_data_consistency import PARAMETERS
from halpha_noise_audit import covariance, coefficient_mean, centered
from escape.longitude_map import longitude_weights

jax.config.update('jax_enable_x64', True)


def longitude_bank(directory):
    paths = sorted(Path(directory).glob('phase_*.npz'))
    rows = [dict(np.load(path)) for path in paths]
    order = np.argsort([float(row['phase']) for row in rows])
    rows, paths = [rows[i] for i in order], [paths[i] for i in order]
    rows, paths = rows[:len(PHASE)], paths[:len(PHASE)]
    np.testing.assert_allclose([row['phase'] for row in rows], PHASE, atol=1e-14, rtol=0)
    return dict(kernel=np.array([row['kernel'] for row in rows]),
                coarse_kernel=np.array([row['coarse_kernel'] for row in rows]),
                longitude=rows[0]['longitude'], velocity=rows[0]['velocity_kms']), paths


def positive_map(nodes, kind, parameter):
    if kind in ['east_west', 'substellar', 'fitted_map']:
        a, b = ((0., parameter) if kind == 'east_west' else
                (parameter, 0.) if kind == 'substellar' else parameter)
        return np.asarray(longitude_weights(jnp.asarray(nodes), a, b))
    center, amplitude = parameter
    width = np.deg2rad(15.)
    quadrature, weights = np.polynomial.legendre.leggauss(64)
    mean = np.sum(weights / 2 * np.exp(-.5 * ((quadrature * np.pi / 2 - np.deg2rad(center)) / width)**2))
    return (1 + amplitude * np.exp(-.5 * ((nodes - np.deg2rad(center)) / width)**2)) / (1 + amplitude * mean)


def map_summary(kind, parameter):
    nodes = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    weights = positive_map(nodes, kind, parameter)
    low, high = nodes <= 0, nodes >= 0
    return dict(minimum=float(weights.min()), maximum=float(weights.max()),
                brightest_to_dimmest=float(weights.max() / weights.min()),
                positive_to_negative_longitude_source_integral=float(np.trapezoid(weights[high], nodes[high]) / np.trapezoid(weights[low], nodes[low])))


def prepare_evaluator(selection, offset, phase, velocity, paths, surface, noise, map_arrays, labels):
    problem = setup(phase, velocity, paths, surface, noise, pixel_offset=offset)
    covs = [covariance(state, group, noise, i) for i, (state, group) in enumerate(zip(problem['states'], problem['groups']))]
    selected = [row['row_id'] for row in labels if row['grid'] == 'fine' and
                ((row['product'] == 'common_complete' and (row['scope'] == 'all' or row['phase_index'] == 3)) or
                 (row['product'] == 'band_complete' and row['scope'] == 'all'))]
    maps = [map_arrays[f'{selection}__g{i}__maps'][selected] for i in range(4)]
    support = np.any([np.any(mapping != 0, axis=(1, 2)) for mapping in maps], axis=0)
    for i, group in enumerate(problem['groups']):
        for key in ['phase', 'stellar_velocity', 'weight', 'pixel_index']:
            np.testing.assert_array_equal(group[key], map_arrays[f'{selection}__g{i}__{key}'])
    observed = sum(np.einsum('rep,ep->r', mapping, group['log_data']) for mapping, group in zip(maps, problem['groups'])) * 1e6
    gp_maps = [np.einsum('rep,mp->rme', mapping, np.asarray(state.basis)).reshape(len(selected), -1)
               for mapping, state in zip(maps, problem['states'])]

    def evaluate(spectrum, parameters=PARAMETERS):
        models = problem['models'](jnp.asarray(spectrum) * np.exp(parameters[1]), parameters[0])
        planet, gp, q = np.zeros(len(selected)), np.zeros(len(selected)), []
        for group, state, cov, mapping, gp_map, model in zip(problem['groups'], problem['states'], covs, maps, gp_maps, models):
            model = np.asarray(model)
            data = group['log_data'] - model
            coefficients = coefficient_mean(group, state, cov, data)
            planet += np.einsum('rep,ep->r', mapping, model) * 1e6
            gp += gp_map @ coefficients.ravel() * 1e6
            residual = centered(group, data)
            product = np.einsum('ep,ep,mp->me', group['weight'], residual, np.asarray(state.basis)).ravel()
            q.append(float(np.sum(group['weight'] * residual**2) / cov['scale']**2 - product @ cov['correction'] @ product))
        return dict(parameters=np.asarray(parameters).tolist(), Q=float(sum(q)), group_Q=q,
                    observed=observed, planet=planet, gp=gp, residual=observed - planet - gp, support=support)

    objective = jax.jit(jax.value_and_grad(lambda p, spectrum: problem['q'](spectrum * jnp.exp(p[1]), p[0]), argnums=0))

    def profile(spectrum):
        value = jnp.asarray(spectrum)
        results = [minimize(lambda p: objective(p, value), seed, jac=True, method='L-BFGS-B',
                    bounds=[(-15, 15), (-3, 3)], options=dict(maxiter=120, ftol=1e-12, gtol=1e-6))
                   for seed in [PARAMETERS, np.array([0., 0.])]]
        best = min(results, key=lambda fit: fit.fun)
        record = evaluate(spectrum, best.x)
        record.update(success=bool(best.success), message=str(best.message), start_Q=[float(fit.fun) for fit in results])
        return record

    return dict(evaluate=evaluate, profile=profile, problem=problem, covs=covs, row_ids=selected)


def render_radial_probe(args):
    """Actual spherical observer transfer with a prescribed upper radial increment."""
    from halpha_geometry_fit import render_state, GRIDS
    state_path = Path('/tmp/escape-e8-mdot-sphere/fine/fixed_inputs.npz')
    recipe_path = Path('/tmp/escape-corrected-observer-full-f/recipe.json')
    state = dict(np.load(state_path))
    edges = np.asarray(json.loads(recipe_path.read_text())['edges_rp'])
    radius = (edges[:-1] + edges[1:]) / 2
    state['radial_velocity_kms'] = state['radial_velocity_kms'].copy()
    state['radial_velocity_kms'][radius >= 1.3] += args.render_radial_kms
    spectrum = render_state(state, edges, PHASE[args.phase_index], GRIDS[args.grid], 19.99871490128246)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    destination = args.raw_output / f'radial_{args.render_radial_kms:+g}_{args.grid}_phase_{args.phase_index:02d}.npz'
    np.savez_compressed(destination, spectrum=spectrum, phase=PHASE[args.phase_index],
                        radial_velocity_kms=state['radial_velocity_kms'])
    destination.with_suffix('.json').write_text(json.dumps(dict(
        scope='Frozen original T/populations/continuum; recomputed Doppler transfer with prescribed added upper radial speed, no SE or dynamics re-solve.',
        added_upper_radial_kms=args.render_radial_kms, phase_index=args.phase_index, grid=GRIDS[args.grid],
        inputs={str(path): sha256(path) for path in [state_path, recipe_path, Path(__file__), Path(__file__).with_name('halpha_geometry_fit.py')]},
        raw_sha256=sha256(destination)), indent=2)+'\n')
    print(destination, flush=True)


def summarize(record, reference, row_ids, labels):
    target = {}
    support = record['support']
    for row_id, value, previous, supported in zip(row_ids, record['residual'], reference['residual'], support):
        row = labels[row_id]
        if row['product'] == 'common_complete' and row['phase_index'] == 3:
            target.setdefault(row['scope'], {})[row['band']] = dict(supported=bool(supported),
                residual_ppm=float(value) if supported else None,
                change_in_planet_plus_GP_ppm=float(previous - value) if supported else None,
                absolute_residual_reduction_ppm=float(abs(previous) - abs(value)) if supported else None)
    all_target = target['all']
    target = {scope: values if any(entry['supported'] for entry in values.values()) else None for scope, values in target.items()}
    phase = {}
    for row_id, value, previous, supported in zip(row_ids, record['residual'], reference['residual'], support):
        row = labels[row_id]
        if row['scope'] == 'all':
            phase.setdefault(row['product'], {}).setdefault(str(row['phase_index']), {})[row['band']] = dict(
                supported=bool(supported), residual_ppm=float(value) if supported else None,
                change_in_planet_plus_GP_ppm=float(previous - value) if supported else None)
    compact_phase = {product: {key: [[values[str(p)][band][field] for band in ['blue','core','red']] for p in range(6)]
                              for key, field in [('residual_BCR_ppm','residual_ppm'),('model_plus_GP_change_BCR_ppm','change_in_planet_plus_GP_ppm')]}
                     for product, values in phase.items()}
    return dict(parameters=record['parameters'], Q=record['Q'], group_Q=record['group_Q'],
        Q_improvement=reference['Q'] - record['Q'], target=target, phase=compact_phase,
        target_blue_and_core_absolute_residuals_reduce=all(all_target[band]['absolute_residual_reduction_ppm'] > 0 for band in ['blue', 'core']),
        target_blue_up_core_down=all_target['blue']['change_in_planet_plus_GP_ppm'] > 0 and all_target['core']['change_in_planet_plus_GP_ppm'] < 0,
        nuisance_success=record.get('success'), nuisance_message=record.get('message'))


def fit_longitude_map(evaluator, bank):
    kernels, nodes = jnp.asarray(bank['kernel']), jnp.asarray(bank['longitude'])
    objective = jax.jit(jax.value_and_grad(lambda p: evaluator['problem']['q'](
        jnp.einsum('plv,l->pv', kernels, longitude_weights(nodes, p[2], p[3])) * jnp.exp(p[1]), p[0])))
    fits = [minimize(objective, np.r_[PARAMETERS, seed], jac=True, method='L-BFGS-B',
        bounds=[(-15,15),(-3,3),(-3,3),(-3,3)], options=dict(maxiter=160,ftol=1e-12,gtol=1e-6))
        for seed in [[0.,0.],[-1.,-1.],[1.,1.]]]
    best = min(fits, key=lambda fit: fit.fun)
    return dict(parameters=best.x.tolist(), Q=float(best.fun), success=bool(best.success), message=str(best.message),
                all_starts=[dict(Q=float(fit.fun),success=bool(fit.success),parameters=fit.x.tolist()) for fit in fits])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-output', type=Path, required=True)
    parser.add_argument('--render-radial-kms', type=float)
    parser.add_argument('--phase-index', type=int, default=3)
    parser.add_argument('--grid', choices=['coarse','fine'], default='coarse')
    args = parser.parse_args()
    if args.render_radial_kms is not None:
        render_radial_probe(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    cache = Path('/tmp/escape-longitude-complete-20261125')
    response = Path('/home/kawahara/exojax-escape/results/contrast_response_20261126')
    baseline_path = Path('/tmp/escape-e8-mdot-sphere/fit_arrays.npz')
    surface_path = Path('/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz')
    noise_path = Path('/tmp/escape-gp-nuts-20261125/gp/samples.npz')
    maps_path = Path('/tmp/escape-halpha-phase-localization/maps.npz')
    rows_path = Path('/tmp/escape-halpha-phase-localization/rows.json')
    paths = [Path(f'/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5') for day in ['08','20']]
    original = dict(np.load(baseline_path))
    baseline = original['fine_intrinsic_phases'][:len(PHASE)]
    bank, kernel_paths = longitude_bank(cache / 'fine')
    coarse, coarse_paths = longitude_bank(cache / 'coarse')
    parity = float(np.max(abs(bank['kernel'].sum(axis=1) - baseline)))
    assert parity < 1e-14
    surface = dict(np.load(surface_path))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in np.load(noise_path).items()}
    map_arrays = np.load(maps_path)
    map_metadata = json.loads(rows_path.read_text())
    labels = map_metadata['rows']
    evaluators = {selection: prepare_evaluator(selection, offset, PHASE, bank['velocity'], paths, surface, noise, map_arrays, labels)
                  for selection, offset in [('primary',0),('alternate',1)]}
    references = {selection: evaluator['evaluate'](baseline) for selection, evaluator in evaluators.items()}
    profile_references = {selection: evaluator['profile'](baseline) for selection, evaluator in evaluators.items()}
    assert abs(references['primary']['Q'] - 38326.64894003179) < 1e-6
    spectra = {'baseline': baseline}
    parameters = {'baseline': dict(family='baseline')}
    for family, values in [('east_west',[-2.,-1.,-.5,.5,1.,2.]),('substellar',[-2.,-1.,1.,2.])]:
        for value in values:
            name = f'{family}_{value:+g}'
            spectra[name] = np.einsum('plv,l->pv', bank['kernel'], positive_map(bank['longitude'],family,value))
            parameters[name] = dict(family=family, coefficient=value, source_map=map_summary(family,value))
    for center in [-60.,-45.,-30.,0.,30.,45.,60.]:
        for amplitude in [3.,9.]:
            name = f'patch_{center:+g}_{amplitude:g}'
            spectra[name] = np.einsum('plv,l->pv',bank['kernel'],positive_map(bank['longitude'],'patch',[center,amplitude]))
            parameters[name] = dict(family='patch',longitude_deg=center,sigma_longitude_deg=15.,excess_over_background=amplitude,
                                    source_map=map_summary('patch',[center,amplitude]))
    velocity_paths = [response / 'coarse' / f'velocity_{i:02d}.npz' for i in range(len(PHASE))]
    velocity_cache = [dict(np.load(path)) for path in velocity_paths]
    shifts = velocity_cache[0]['shifts']
    velocity_spectra = np.stack([row['components'].sum(axis=1) for row in velocity_cache], axis=1)
    zero_index = np.flatnonzero(shifts == 0)[0]
    for i, shift in enumerate(shifts):
        if shift == 0: continue
        name = f'upper_LOS_{shift:+g}'
        spectra[name] = baseline + velocity_spectra[i] - velocity_spectra[zero_index]
        parameters[name] = dict(family='upper_LOS',relative_velocity_kms=float(shift),
            convention='Uniform additional observer line-of-sight velocity in line opacity and emissivity above 1.3 Rp. Continuum and all populations fixed; not a global vector wind solution.',
            numerical='Finite coarse-grid transfer difference added to the archived fine baseline.')
    radial_paths = [response / 'coarse' / f'phase_{i:02d}.npz' for i in range(len(PHASE))]
    radial = [dict(np.load(path)) for path in radial_paths]
    for label, scale in [('v_zero',0.),('v_double',2.)]:
        spectra[label] = baseline + np.array([(row[label] - row['base']).sum(axis=0) for row in radial])
        parameters[label] = dict(family='upper_radial_scale',scale=scale,
            convention='Multiply the actual saved upper radial flow; density/populations/continuum fixed.')
    extra_paths = []
    for increment in [5.5, 13.]:
        grid = 'fine' if increment == 13 else 'coarse'
        available = [args.raw_output / f'radial_{increment:+g}_{grid}_phase_{i:02d}.npz' for i in range(len(PHASE))]
        if not all(path.exists() for path in available):
            continue
        value = np.array([np.load(path)['spectrum'] for path in available])
        if grid == 'coarse':
            value += baseline - original['coarse_intrinsic_phases'][:len(PHASE)]
        name = f'upper_radial_add_{increment:+g}'
        spectra[name] = value
        parameters[name] = dict(family='upper_radial_add',added_velocity_kms=increment,grid=grid,
            convention='Actual radial vector projected along each spherical ray; all upper cell radial speeds receive this increment at frozen populations/density/continuum.')
        extra_paths.extend(available)
    map_fit = fit_longitude_map(evaluators['primary'], bank)
    fitted = map_fit['parameters'][2:]
    spectra['all_data_longitude_fit'] = np.einsum('plv,l->pv',bank['kernel'],positive_map(bank['longitude'],'fitted_map',fitted))
    parameters['all_data_longitude_fit'] = dict(family='fitted_map',coefficients=fitted,source_map=map_summary('fitted_map',fitted),fit=map_fit)
    reports, raw = {}, dict(phase=PHASE,velocity=bank['velocity'])
    for name, spectrum in spectra.items():
        reports[name] = dict(parameters=parameters[name],selections={})
        raw[name + '__spectrum'] = spectrum
        for selection,evaluator in evaluators.items():
            fixed = evaluator['evaluate'](spectrum)
            profiled = profile_references[selection] if name == 'baseline' else evaluator['profile'](spectrum)
            reports[name]['selections'][selection] = dict(fixed=summarize(fixed,references[selection],evaluator['row_ids'],labels),
                profiled=summarize(profiled,profile_references[selection],evaluator['row_ids'],labels))
            for mode, record in [('fixed',fixed),('profiled',profiled)]:
                for key in ['observed','planet','gp','residual']:
                    raw[f'{name}__{selection}__{mode}__{key}'] = record[key]
        print(json.dumps(dict(case=name,primary=reports[name]['selections']['primary']['fixed']['target']['all'],
                              Q_improvement=reports[name]['selections']['primary']['fixed']['Q_improvement'])),flush=True)
    for shift in [-5.,-2.,2.,5.]:
        name = f'common_RV_{shift:+g}'
        reports[name] = dict(parameters=dict(family='common_RV',shift_kms=shift,
            scope='Common planetary shift control, exactly degenerate with the existing shared RV nuisance.'),selections={})
        for selection,evaluator in evaluators.items():
            fixed = evaluator['evaluate'](baseline, PARAMETERS + [shift,0.])
            reports[name]['selections'][selection] = dict(fixed=summarize(fixed,references[selection],evaluator['row_ids'],labels),
                profiled=reports['baseline']['selections'][selection]['profiled'])
    np.savez_compressed(args.raw_output / 'arrays.npz',**raw)
    inputs = [Path(__file__),*[Path(__file__).with_name(name) for name in ['halpha_geometry_fit.py','halpha_data_consistency.py','halpha_noise_audit.py','gp_halpha.py']],
        Path(__file__).resolve().parents[1]/'src/escape/longitude_map.py',baseline_path,surface_path,noise_path,maps_path,rows_path,*paths,*kernel_paths,*coarse_paths,*velocity_paths,*radial_paths,*extra_paths]
    target_support = map_metadata['selections']['primary']['rows'][9]
    review = dict(scope=__doc__,parameters=PARAMETERS.tolist(),target_actual_phase_range=[target_support['phase_min'],target_support['phase_max']],
        target_definition='Sealed common_complete fine phase index 3, same 12 August-8 exposures in B/C/R. All phases and both nights remain in the score and GP conditioning.',
        longitude_scope='Positive normalized redistribution of local source and lower-boundary terms at fixed opacity and velocity, including the existing spherical occultation/visibility. No new SE/thermal solution.',
        longitude_uniform_parity_max_abs=parity,row_ids={selection:evaluator['row_ids'] for selection,evaluator in evaluators.items()},
        selected_rows=[labels[i] for i in evaluators['primary']['row_ids']],cases=reports,
        limitations=['All scores are conditional, descriptive and use the existing fixed GP family.',
                     'Longitude source redistribution changes emissivity and emitting boundary at fixed extinction; no corresponding thermal or population balance is solved.',
                     'Observer line-profile velocity effects are not the frozen pumping/net-excitation effects studied previously.',
                     'The target phase block has large correlated uncertainty; direction ranking does not identify a physical cause.',
                     'No complete late blue data exist; unsupported projected rows must not be interpreted as measured zeros.'],
        inputs={str(path):sha256(path) for path in inputs},raw=dict(path=str(args.raw_output/'arrays.npz'),sha256=sha256(args.raw_output/'arrays.npz')))
    (args.output/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
