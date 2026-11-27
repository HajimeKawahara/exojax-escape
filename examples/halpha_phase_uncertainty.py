"""Conditional covariance of predeclared native phase-band residual maps.

The fixed planet, full common reference, and GP use the original samples.
Global RV/amplitude linear refitting and empirical time-block multipliers are
separate sensitivities, not calibrated confidence intervals or detections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

from gp_halpha import load_gp_data, make_gp_problem
from halpha_noise_audit import (
    band_maps, band_record, block_sensitivity, centered, coefficient_mean,
    covariance, digest, fit_factory, simulate,
)
from escape.observation import C_KMS
from escape.time_gp import time_gp_posterior

PHASE_EDGES = [.30, .325, .35, .375, .40, .425, .435]
BANDS = dict(blue=(-40., -16.), core=(-8., 8.), red=(16., 40.))


def apply_precision(group, state, cov, values):
    """Apply the common-reference and GP marginal precision to (E,P[,K])."""
    values = np.asarray(values)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[:, :, None]
    w = group["weight"]
    total = np.where(w.sum(axis=0) > 0, w.sum(axis=0), 1.)

    def project(array):
        mean = np.sum(w[:, :, None] * np.where(w[:, :, None] > 0, array, 0.), axis=0) / total[:, None]
        return np.where(w[:, :, None] > 0, array - mean, 0.)

    projected = project(values)
    product = np.einsum("ep,mp,epk->mek", w, state.basis, projected).reshape(len(cov["correction"]), -1)
    coefficients = (cov["correction"] @ product).reshape(len(state.basis), len(w), -1)
    field = np.einsum("mek,mp->epk", coefficients, state.basis)
    result = w[:, :, None] * (projected / cov["scale"]**2 - project(field))
    return result[:, :, 0] if squeeze else result


def propagate(problem, covs, maps):
    count = maps[0].shape[0]
    result = np.zeros((count, count))
    for group, state, cov, mapping in zip(problem["groups"], problem["states"], covs, maps):
        inverse = np.divide(cov["scale"]**2, group["weight"], out=np.zeros_like(group["weight"]), where=group["weight"] > 0)
        response = np.einsum("rep,mp->rme", mapping, state.basis).reshape(count, -1)
        weighted = mapping.reshape(count, -1) * np.sqrt(inverse.ravel())[None]
        result += weighted @ weighted.T
        result += response @ cov["prior"] @ response.T
    return (result + result.T) / 2


def responses(problem, covs, maps, parameters):
    """Linear residual response with fixed planet and with global local refitting."""
    count = maps[0].shape[0]
    models = problem["native_log_models"](jnp.asarray(parameters))
    jacobians = jax.jacfwd(problem["native_log_models"])(jnp.asarray(parameters))
    fixed, precisions, innovations = [], [], []
    information, derivative = np.zeros((2, 2)), np.zeros((count, 2))
    for group, state, cov, model, jac, mapping in zip(problem["groups"], problem["states"], covs, models, jacobians, maps):
        w, basis, jac = group["weight"], np.asarray(state.basis), np.asarray(jac)
        product_map = (np.einsum("mp,ep,ef->mefp", basis, w, np.eye(len(w)))
                       - np.einsum("mp,ep,fp,p->mefp", basis, w, w, 1 / w.sum(axis=0)))
        product_map = product_map.reshape(len(cov["estimator"]), *w.shape)
        gp_map = np.einsum("rep,mp->rme", mapping, basis).reshape(count, -1)
        response = mapping - (gp_map @ cov["estimator"] @ product_map.reshape(len(cov["estimator"]), -1)).reshape(mapping.shape)
        precision_jac = apply_precision(group, state, cov, jac)
        information += np.einsum("epk,epl->kl", jac, precision_jac)
        derivative += response.reshape(count, -1) @ jac.reshape(-1, 2)
        fixed.append(response)
        precisions.append(precision_jac)
        mean = coefficient_mean(group, state, cov, group["log_data"] - model)
        innovations.append(centered(group, group["log_data"] - model - mean.T @ basis))
    inverse = la.inv(information)
    local = [r - (derivative @ inverse @ p.reshape(-1, 2).T).reshape(r.shape) for r, p in zip(fixed, precisions)]
    leakage = sum(np.einsum("rep,epk->rk", r, j) for r, j in zip(local, jacobians))
    return dict(fixed=fixed, local=local, innovations=innovations, models=models,
                information=information, derivative=derivative, precisions=precisions,
                nuisance_derivative_leakage_max=float(np.max(abs(leakage))),
                fixed_covariance=propagate(problem, covs, fixed), local_covariance=propagate(problem, covs, local))


def evaluate(problem, covs, maps, parameters):
    count = maps[0].shape[0]
    data, planet, gp = (np.zeros(count) for _ in range(3))
    models = problem["native_log_models"](jnp.asarray(parameters))
    for group, state, cov, model, mapping in zip(problem["groups"], problem["states"], covs, models, maps):
        mean = coefficient_mean(group, state, cov, group["log_data"] - model)
        data += np.einsum("rep,ep->r", mapping, group["log_data"])
        planet += np.einsum("rep,ep->r", mapping, model)
        gp += np.einsum("rep,ep->r", mapping, mean.T @ state.basis)
    return dict(observed=data, planet=planet, gp=gp, residual=data - planet - gp)


def blocks(problem, response, innovations):
    """Simultaneous two-order residual multiplier covariance, with offsets."""
    groups = problem["groups"]
    influence = [np.einsum("rep,ep->er", mapping, error) for mapping, error in zip(response, innovations)]
    records, matrices, effective = [], [], []
    for size in (1, 2, 4, 8):
        for offset in range(size):
            contributions = []
            for night in (0, 1):
                left, right = 2 * night, 2 * night + 1
                if not np.array_equal(groups[left]["phase"], groups[right]["phase"]):
                    raise ValueError("Simultaneous order cadence mismatch")
                values = influence[left] + influence[right]
                time = groups[left]["time_min"]
                edges = np.r_[0, np.flatnonzero(np.diff(time) > 1.6 * np.median(np.diff(time))) + 1, len(time)]
                for low, high in zip(edges[:-1], edges[1:]):
                    label = (np.arange(high - low) + offset) // size
                    contributions.extend(values[low:high][label == k].sum(axis=0) for k in np.unique(label))
            values = np.asarray(contributions)
            matrices.append(values.T @ values)
            square = np.sum(values**2, axis=0)
            effective.append(np.divide(square**2, np.sum(values**4, axis=0), out=np.zeros_like(square), where=square > 0))
            records.append(dict(block_exposures=size, offset=offset, blocks=len(values)))
    return records, np.asarray(matrices), np.asarray(effective)


def map_validation(groups, maps, rows, supported):
    """Recover the positive source weights using exposures outside each row."""
    row_sum = np.zeros(len(rows))
    common_error = 0.
    negative, outside, unresolved = 0., 0., 0
    for group, mapping in zip(groups, maps):
        w = group["weight"]
        reference = w / w.sum(axis=0)
        orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
        velocity = C_KMS * ((1 + group["stellar_velocity"][None] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)
        common_error = max(common_error, float(np.max(abs(mapping.sum(axis=1)))))
        for i, row in enumerate(rows):
            phase = (group["phase"] >= row["phase_low"]) & (group["phase"] < row["phase_high"]) & group["science"]
            outside_reference = np.sum(reference[~phase], axis=0)
            total = np.divide(-np.sum(mapping[i, ~phase], axis=0), outside_reference,
                              out=np.zeros_like(outside_reference), where=outside_reference > 0)
            source = mapping[i] + reference * total
            resolved = outside_reference > 0
            unresolved += int(np.sum(~resolved & np.any(mapping[i] != 0, axis=0)))
            low, high = BANDS[row["band"]]
            allowed = phase[:, None] & (velocity >= low) & (velocity < high) & (w > 0)
            outside = max(outside, float(np.max(abs(np.where(allowed | ~resolved[None], 0., source)))))
            negative = max(negative, float(np.max(np.maximum(-source[:, resolved], 0.))))
            row_sum[i] += source[:, resolved].sum()
    return dict(common_mode_leakage_max=common_error, recovered_source_negative_max=negative,
                recovered_source_outside_declared_phase_band_max=outside, unresolved_supported_columns=unresolved,
                recovered_source_normalization_error_max=float(np.max(abs(row_sum - supported))),
                unsupported_map_max=max(float(np.max(abs(m[~supported]))) if np.any(~supported) else 0. for m in maps))


def injection_check(problem, covs, maps, response, count):
    """Compare explicit production GP fits with linear maps on actual support."""
    rng = np.random.default_rng(20261201)
    production = jax.jit(time_gp_posterior)
    fixed_samples, local_samples = [], []
    parity = 0.
    inverse = la.inv(response["information"])
    for _ in range(count):
        direct, mapped, score = np.zeros(maps[0].shape[0]), np.zeros(maps[0].shape[0]), np.zeros(2)
        local = np.zeros(maps[0].shape[0])
        for group, state, cov, mapping, fixed_map, local_map, precision in zip(problem["groups"], problem["states"], covs, maps, response["fixed"], response["local"], response["precisions"]):
            noise = np.asarray(simulate(group, state, cov, rng))
            # Recover amplitudes and length from the already prepared prior.
            size = len(group["phase"])
            amplitudes = np.sqrt(np.diag(cov["prior"])[::size] / (1 + 1e-10))
            # These are retained explicitly by main; the matrix itself need not
            # be inverted to infer a kernel length.
            post = production(noise, state, group["time_min"], amplitudes, cov["length"], cov["scale"])
            direct += np.einsum("rep,ep->r", mapping, noise - np.asarray(post.mean).T @ state.basis)
            mapped += np.einsum("rep,ep->r", fixed_map, noise)
            local += np.einsum("rep,ep->r", local_map, noise)
            score += np.einsum("epk,ep->k", precision, noise)
        direct_local = direct - response["derivative"] @ inverse @ score
        parity = max(parity, float(np.max(abs(direct - mapped))), float(np.max(abs(direct_local - local))))
        fixed_samples.append(direct)
        local_samples.append(direct_local)
    fixed_samples, local_samples = np.asarray(fixed_samples), np.asarray(local_samples)
    diagnostics = {}
    for label, values in (("fixed", fixed_samples), ("local", local_samples)):
        analytic = np.diag(response[label + "_covariance"])
        keep = analytic > 1e-24
        ratio = np.var(values, axis=0, ddof=1)[keep] / analytic[keep]
        diagnostics[label] = dict(variance_ratio_min_median_max=np.quantile(ratio, [0, .5, 1]).tolist(),
            maximum_variance_relative_error=float(np.max(abs(ratio - 1))))
    return dict(draws=count, direct_pipeline_linear_map_max_absolute_error_ppm=parity * 1e6,
                empirical_variance=diagnostics, scope="Gaussian draws from the declared covariance on actual native support. No atmosphere or original extraction noise is simulated; nuisance refitting is linear.")


def correlations(matrix):
    norm = np.sqrt(np.maximum(np.diag(matrix), 0.))
    return np.divide(matrix, norm[:, None] * norm[None], out=np.zeros_like(matrix), where=(norm[:, None] * norm[None]) > 0)


def row_summary(rows, supported, values, response, block_cov, effective, fixed_block_cov, fixed_effective, draws_fixed, draws_refit):
    result = []
    diagonal = np.maximum(np.diagonal(block_cov, axis1=1, axis2=2), 0.)
    fixed_diagonal = np.maximum(np.diagonal(fixed_block_cov, axis1=1, axis2=2), 0.)
    for i, row in enumerate(rows):
        output = dict(row, supported=bool(supported[i]))
        if supported[i]:
            value = values["residual"][i] * 1e6
            sigma = np.sqrt(diagonal[:, i]) * 1e6
            fixed_sigma = np.sqrt(fixed_diagonal[:, i]) * 1e6
            envelope = [float(value - 2 * sigma.max()), float(value + 2 * sigma.max())]
            fixed_envelope = [float(value - 2 * fixed_sigma.max()), float(value + 2 * fixed_sigma.max())]
            output.update(**{key + "_ppm": float(array[i] * 1e6) for key, array in values.items()},
                fixed_planet_sigma_ppm=float(np.sqrt(max(response["fixed_covariance"][i, i], 0.)) * 1e6),
                local_RV_amplitude_refit_sigma_ppm=float(np.sqrt(max(response["local_covariance"][i, i], 0.)) * 1e6),
                fixed_planet_time_block_sigma_range_ppm=[float(fixed_sigma.min()), float(fixed_sigma.max())],
                local_refit_time_block_sigma_range_ppm=[float(sigma.min()), float(sigma.max())],
                fixed_planet_empirical_effective_block_range=[float(fixed_effective[:, i].min()), float(fixed_effective[:, i].max())],
                local_refit_empirical_effective_block_range=[float(effective[:, i].min()), float(effective[:, i].max())],
                fixed_planet_all_block_offsets_two_sd_proxy_envelope_ppm=fixed_envelope,
                local_refit_all_block_offsets_two_sd_proxy_envelope_ppm=envelope,
                sign_excludes_zero_for_every_fixed_planet_two_sd_proxy=bool(fixed_envelope[0] > 0 or fixed_envelope[1] < 0),
                sign_excludes_zero_for_every_local_refit_two_sd_proxy=bool(envelope[0] > 0 or envelope[1] < 0),
                noise_draw_fixed_planet_residual_quantiles05_50_95_ppm=(np.quantile(draws_fixed[:, i], [.05, .5, .95]) * 1e6).tolist(),
                noise_draw_refit_planet_residual_quantiles05_50_95_ppm=(np.quantile(draws_refit[:, i], [.05, .5, .95]) * 1e6).tolist())
        result.append(output)
    return result


def phase_pairs(rows, supported, values, response, block_cov, fixed_block_cov):
    """All supported phase pairs in the predeclared primary product, not a search."""
    selected = [i for i, row in enumerate(rows) if row["grid"] == "fine" and row["product"] == "common_complete" and row["scope"] == "all" and supported[i]]
    result = []
    for position, left in enumerate(selected):
        for right in selected[position + 1:]:
            if rows[left]["band"] != rows[right]["band"]:
                continue
            delta = (values["residual"][right] - values["residual"][left]) * 1e6
            fixed = response["fixed_covariance"]
            variance = fixed[left, left] + fixed[right, right] - 2 * fixed[left, right]
            block_variance = block_cov[:, left, left] + block_cov[:, right, right] - 2 * block_cov[:, left, right]
            sigmas = np.sqrt(np.maximum(block_variance, 0.)) * 1e6
            fixed_block_variance = fixed_block_cov[:, left, left] + fixed_block_cov[:, right, right] - 2 * fixed_block_cov[:, left, right]
            fixed_sigmas = np.sqrt(np.maximum(fixed_block_variance, 0.)) * 1e6
            interval = [float(delta - 2 * sigmas.max()), float(delta + 2 * sigmas.max())]
            fixed_interval = [float(delta - 2 * fixed_sigmas.max()), float(delta + 2 * fixed_sigmas.max())]
            result.append(dict(earlier_row_id=rows[left]["row_id"], later_row_id=rows[right]["row_id"], band=rows[left]["band"],
                later_minus_earlier_residual_ppm=float(delta), fixed_planet_difference_sigma_ppm=float(np.sqrt(max(variance, 0.)) * 1e6),
                phase_correlation=float(correlations(fixed)[left, right]),
                fixed_planet_time_block_difference_sigma_range_ppm=[float(fixed_sigmas.min()), float(fixed_sigmas.max())],
                local_refit_time_block_difference_sigma_range_ppm=[float(sigmas.min()), float(sigmas.max())],
                fixed_planet_all_block_offsets_two_sd_proxy_envelope_ppm=fixed_interval,
                local_refit_all_block_offsets_two_sd_proxy_envelope_ppm=interval,
                sign_excludes_zero_for_every_fixed_planet_two_sd_proxy=bool(fixed_interval[0] > 0 or fixed_interval[1] < 0),
                sign_excludes_zero_for_every_local_refit_two_sd_proxy=bool(interval[0] > 0 or interval[1] < 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("/tmp/escape-halpha-phase-localization/maps.npz"))
    parser.add_argument("--rows", type=Path, default=Path("/tmp/escape-halpha-phase-localization/rows.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=Path("/tmp/escape-halpha-phase-uncertainty"))
    parser.add_argument("--posterior-draws", type=int, default=64)
    parser.add_argument("--injections", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repository = Path(__file__).resolve().parents[1]
    paths = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    model_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    sample_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    surface_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    bins_path = Path("/tmp/escape-gp-combined-20261125/combined_arrays.npz")
    sources = [Path(__file__), *[repository / "examples" / name for name in ("halpha_noise_audit.py", "gp_halpha.py", "gp_controls.py", "halpha_geometry_fit.py", "phase_halpha.py", "cloudy_halpha.py")], repository / "src/escape/time_gp.py"]
    input_paths = [*paths, model_path, sample_path, surface_path, bins_path, args.maps, args.rows, *sources]
    pins = {str(path): digest(path) for path in input_paths}
    metadata = json.loads(args.rows.read_text())
    selected = [i for i, row in enumerate(metadata["rows"]) if
                (row["grid"] == "fine" and row["product"] == "common_complete") or
                (row["grid"] == "fine" and row["product"] == "band_complete" and row["scope"] == "all") or
                (row["grid"] == "fine" and row["product"] == "band_complete" and row["band"] == "core" and row["phase_low"] >= .40) or
                (row["grid"] == "coarse" and row["product"] == "common_complete" and row["scope"] == "all")]
    if not selected:
        raise ValueError("No declared phase-band rows found")
    rows = [metadata["rows"][i] for i in selected]
    for row in rows:
        if row["grid"] == "fine" and [row["phase_low"], row["phase_high"]] != PHASE_EDGES[row["phase_index"]:row["phase_index"] + 2]:
            raise ValueError("Fine phase bins differ from the predeclared edges")
    model, surface, samples = (dict(np.load(path)) for path in (model_path, surface_path, sample_path))
    noise = {key: np.median(samples[key], axis=(0, 1)) for key in ("log_gp", "log_length", "log_white")}
    flat = {key: value.reshape((-1,) + value.shape[2:]) for key, value in samples.items()}
    indices = np.sort(np.random.default_rng(20261202).choice(len(flat["log_white"]), args.posterior_draws, replace=False))
    parameters = metadata["parameters"]
    archive = np.load(args.maps)
    reports, arrays, details = {}, {}, {}
    for selection, offset in (("primary", 0), ("alternate", 1)):
        groups = load_gp_data(paths, pixel_offset=offset)
        maps = []
        for k, group in enumerate(groups):
            for key in ("phase", "stellar_velocity", "pixel_index", "weight", "science", "eclipse"):
                if not np.array_equal(archive[f"{selection}__g{k}__{key}"], group[key]):
                    raise ValueError(f"Map input alignment mismatch: {selection}, group{k}, {key}")
            maps.append(archive[f"{selection}__g{k}__maps"][selected])
        supported = np.array([metadata["selections"][selection]["rows"][i]["supported"] for i in selected])
        problem = make_gp_problem(groups, model, surface)
        covs = [covariance(state, group, noise, k) for k, (state, group) in enumerate(zip(problem["states"], groups))]
        for k, cov in enumerate(covs):
            cov["length"] = np.exp(noise["log_length"][k])
        checks = map_validation(groups, maps, rows, supported)
        if any(checks[key] > 1e-10 for key in ("common_mode_leakage_max", "recovered_source_negative_max", "recovered_source_outside_declared_phase_band_max", "recovered_source_normalization_error_max", "unsupported_map_max")) or checks["unresolved_supported_columns"]:
            raise ValueError(f"Phase map validation failed: {checks}")
        print(json.dumps(dict(selection=selection, progress="maps_validated", rows=len(rows), checks=checks)), flush=True)
        response = responses(problem, covs, maps, parameters)
        values = evaluate(problem, covs, maps, parameters)
        checks["independent_map_values_max_abs_difference_ppm"] = max(abs(values[key][i] * 1e6 - metadata["selections"][selection]["values_ppm"][original][key])
            for i, original in enumerate(selected) if supported[i] for key in values)
        if checks["independent_map_values_max_abs_difference_ppm"] > 1e-7:
            raise ValueError("Independent phase-map values do not match")
        block_rows, block_cov, effective = blocks(problem, response["local"], response["innovations"])
        _, fixed_block_cov, fixed_effective = blocks(problem, response["fixed"], response["innovations"])
        print(json.dumps(dict(selection=selection, progress="responses_and_blocks_complete")), flush=True)
        fixed_draws, refit_draws, fits = [], [], []
        fit = fit_factory(problem)
        for index in indices:
            draw_noise = {key: flat[key][index] for key in noise}
            draw_cov = [covariance(state, group, draw_noise, k) for k, (state, group) in enumerate(zip(problem["states"], groups))]
            fixed_draws.append(evaluate(problem, draw_cov, maps, parameters)["residual"])
            fitted = fit(draw_cov, parameters)
            refit_draws.append(evaluate(problem, draw_cov, maps, fitted["parameters"])["residual"])
            fits.append(dict(draw_index=int(index), **fitted))
        fixed_draws, refit_draws = np.asarray(fixed_draws), np.asarray(refit_draws)
        injections = injection_check(problem, covs, maps, response, args.injections)
        legacy = None
        if selection == "primary":
            old_maps, _ = band_maps(groups, covs, np.load(bins_path)["edges"])
            old_response = responses(problem, covs, old_maps, parameters)
            old_record = band_record(problem, old_maps, covs, parameters, True)
            old_block = block_sensitivity(problem, old_maps, covs, parameters)
            legacy = dict(fixed_covariance_max_abs_difference_ppm2=float(np.max(abs(old_response["fixed_covariance"] * 1e12 - old_record["fixed_mean_fitted_gp_residual_covariance_ppm2"]))),
                local_covariance_max_abs_difference_ppm2=float(np.max(abs(old_response["local_covariance"] * 1e12 - old_block["model_family_covariance_after_local_RV_amplitude_refit_ppm2"]))))
        summary = row_summary(rows, supported, values, response, block_cov, effective, fixed_block_cov, fixed_effective, fixed_draws, refit_draws)
        details[selection] = dict(rows=summary, noise_draw_fits=fits)
        reports[selection] = dict(map_validation=checks, rows=[row for row in summary if row["scope"] == "all" or (row["product"] == "band_complete" and row["band"] == "core" and row["phase_low"] >= .40)],
            phase_pairs=phase_pairs(rows, supported, values, response, block_cov, fixed_block_cov), block_definitions=block_rows,
            nuisance_information=response["information"].tolist(), nuisance_derivative_leakage_max=response["nuisance_derivative_leakage_max"],
            legacy_three_band_correspondence=legacy, injections=injections,
            noise_draw_count=len(fits), noise_draw_optimizer_failure_count=sum(not row["success"] for row in fits))
        arrays.update({f"{selection}__{key}": array for key, array in dict(row_id=np.array([row["row_id"] for row in rows]), supported=supported,
            fixed_covariance_ppm2=response["fixed_covariance"] * 1e12, local_covariance_ppm2=response["local_covariance"] * 1e12,
            fixed_correlation=correlations(response["fixed_covariance"]), local_time_block_covariance_ppm2=block_cov * 1e12,
            fixed_time_block_covariance_ppm2=fixed_block_cov * 1e12,
            local_time_block_effective_contributors=effective, fixed_time_block_effective_contributors=fixed_effective, noise_fixed_residual_draws_ppm=fixed_draws * 1e6,
            noise_refit_residual_draws_ppm=refit_draws * 1e6, **{key + "_ppm": value * 1e6 for key, value in values.items()}).items()})
        print(json.dumps(dict(selection=selection, rows=len(rows), supported=int(supported.sum()), checks=checks, injections=injections)), flush=True)
    artifact = args.raw_output / "arrays.npz"
    np.savez_compressed(artifact, **arrays)
    details_path = args.raw_output / "details.json"
    details_path.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    result = dict(schema="escape.halpha_phase_uncertainty.v1", phase_edges=PHASE_EDGES, bands=BANDS,
        fixed_parameters=parameters, source_input_sha256=pins, unchanged_source_input_hashes=all(digest(path) == sha for path, sha in pins.items()),
        raw_artifact_sha256={str(path): digest(path) for path in (artifact, details_path)}, results=reports, elapsed_seconds=time.monotonic() - started,
        scope=__doc__, limits=["Rows retain the full original reference and GP; unsupported rows are zero maps with supported=false, never extrapolated predictions.",
            "Fixed-planet covariance and global local-linear RV/amplitude covariance are separate; neither marginalizes atmosphere physics or unknown extraction covariance.",
            "Local-refit covariance and block proxy intervals retain the original fixed-parameter center; particularly for alternate pixels this is not the mean from a full RV/amplitude refit.",
            "Noise posterior draws explore the existing grey/slope GP family. Refitting RV/amplitude is a separate sensitivity from the fixed-parameter phase comparison.",
            "Time-block multipliers use already fitted GP/common residuals, include possible phase-dependent physical errors, and assume independence between blocks. They do not calibrate raw extraction noise or GP-process uncertainty.",
            "Two-standard-deviation proxy envelopes and many phase/order comparisons are descriptive, not calibrated confidence levels, simultaneous intervals, or detection significances."])
    (args.output / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
