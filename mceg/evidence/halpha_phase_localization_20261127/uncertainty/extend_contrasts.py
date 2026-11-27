"""Derive within-phase and matched-night contrasts without rerunning all rows."""

from pathlib import Path
import json
import sys

import jax
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY / "examples"))

from gp_halpha import load_gp_data, make_gp_problem
from halpha_noise_audit import covariance, digest
from halpha_phase_uncertainty import blocks, evaluate, map_validation, responses


def summarize(coefficients, values, fixed, local, fixed_blocks, local_blocks, definitions, draws_fixed, draws_refit):
    c = np.asarray(coefficients)
    result = {key + "_ppm": float(c @ value) for key, value in values.items()}
    center = result["residual_ppm"]
    for name, cov, block in (("fixed_planet", fixed, fixed_blocks), ("local_refit", local, local_blocks)):
        sigma = np.sqrt(np.maximum(np.einsum("r,brs,s->b", c, block, c), 0.))
        interval = [float(center - 2 * sigma.max()), float(center + 2 * sigma.max())]
        result[name] = dict(conditional_sigma_ppm=float(np.sqrt(max(c @ cov @ c, 0.))),
            all_time_block_sigma_range_ppm=[float(sigma.min()), float(sigma.max())],
            by_block_size={str(size): [float(min(sigma[i] for i, row in enumerate(definitions) if row["block_exposures"] == size)),
                                     float(max(sigma[i] for i, row in enumerate(definitions) if row["block_exposures"] == size))] for size in (1, 2, 4, 8)},
            all_block_offsets_two_sd_proxy_envelope_ppm=interval,
            sign_excludes_zero_for_every_two_sd_proxy=bool(interval[0] > 0 or interval[1] < 0))
    for name, draws in (("fixed_planet", draws_fixed), ("refit_planet", draws_refit)):
        result["noise_draw_" + name + "_quantiles05_50_95_ppm"] = np.quantile(draws @ c, [.05, .5, .95]).tolist()
    return result


def main():
    output = Path(__file__).resolve().parent
    raw = Path("/tmp/escape-halpha-phase-uncertainty")
    map_path = Path("/tmp/escape-halpha-phase-localization/matched_nights_maps.npz")
    metadata_path = output.parent / "localization/match_nights.json"
    paths = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    model_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    surface_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    samples_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    inputs = [*paths, model_path, surface_path, samples_path, map_path, metadata_path, raw / "arrays.npz", raw / "details.json", output / "review.json",
              Path(__file__), REPOSITORY / "examples/halpha_phase_uncertainty.py", REPOSITORY / "examples/halpha_noise_audit.py", REPOSITORY / "examples/gp_halpha.py", REPOSITORY / "src/escape/time_gp.py"]
    pins = {str(path): digest(path) for path in inputs}
    full = np.load(raw / "arrays.npz")
    details = json.loads((raw / "details.json").read_text())
    review = json.loads((output / "review.json").read_text())
    metadata = json.loads(metadata_path.read_text())
    archive = np.load(map_path)
    samples = dict(np.load(samples_path))
    noise = {key: np.median(samples[key], axis=(0, 1)) for key in ("log_gp", "log_length", "log_white")}
    flat = {key: array.reshape((-1,) + array.shape[2:]) for key, array in samples.items()}
    result = {}
    for selection, offset in (("primary", 0), ("alternate", 1)):
        rows = details[selection]["rows"]
        definitions = review["results"][selection]["block_definitions"]
        values = {key: full[f"{selection}__{key}_ppm"] for key in ("observed", "planet", "gp", "residual")}
        extra = dict(fixed=full[f"{selection}__fixed_covariance_ppm2"], local=full[f"{selection}__local_covariance_ppm2"],
            fixed_blocks=full[f"{selection}__fixed_time_block_covariance_ppm2"], local_blocks=full[f"{selection}__local_time_block_covariance_ppm2"],
            definitions=definitions, draws_fixed=full[f"{selection}__noise_fixed_residual_draws_ppm"], draws_refit=full[f"{selection}__noise_refit_residual_draws_ppm"])
        within = []
        for i, row in enumerate(rows):
            if row["product"] != "common_complete" or row["scope"] != "all" or row["band"] != "blue":
                continue
            j = next(k for k, other in enumerate(rows) if other["grid"] == row["grid"] and other["phase_index"] == row["phase_index"] and other["product"] == row["product"] and other["scope"] == "all" and other["band"] == "core")
            supported = row["supported"] and rows[j]["supported"]
            entry = dict(grid=row["grid"], phase_low=row["phase_low"], phase_high=row["phase_high"], supported=supported,
                         blue_row_id=row["row_id"], core_row_id=rows[j]["row_id"])
            if supported:
                c = np.zeros(len(rows)); c[i], c[j] = 1., -1.
                entry.update(summarize(c, values, **extra))
            within.append(entry)

        groups = load_gp_data(paths, pixel_offset=offset)
        selected = [4, 7]  # The supplied matched core rows: night0, night1.
        matched_rows = [metadata["rows"][i] for i in selected]
        if [(row["band"], row["scope"]) for row in matched_rows] != [("core", "night0"), ("core", "night1")]:
            raise ValueError("Matched-night row interface changed")
        maps = []
        for k, group in enumerate(groups):
            for key in ("phase", "stellar_velocity", "pixel_index", "weight", "science", "eclipse"):
                if not np.array_equal(archive[f"{selection}__g{k}__{key}"], group[key]):
                    raise ValueError("Matched-night map alignment changed")
            maps.append(archive[f"{selection}__g{k}__maps"][selected])
        supported = np.array([metadata["selections"][selection]["rows"][i]["supported"] for i in selected])
        validation = map_validation(groups, maps, matched_rows, supported)
        if not supported.all() or validation["recovered_source_normalization_error_max"] > 1e-10 or validation["recovered_source_outside_declared_phase_band_max"] > 1e-10:
            raise ValueError("Matched-night source support validation failed")
        problem = make_gp_problem(groups, dict(np.load(model_path)), dict(np.load(surface_path)))
        covs = [covariance(state, group, noise, k) for k, (state, group) in enumerate(zip(problem["states"], groups))]
        parameters = metadata["parameters"]
        response = responses(problem, covs, maps, parameters)
        values = evaluate(problem, covs, maps, parameters)
        validation["independent_value_max_abs_difference_ppm"] = max(abs(values[key][k] * 1e6 - metadata["selections"][selection]["values_ppm"][i][key]) for k, i in enumerate(selected) for key in values)
        definitions, fixed_blocks, _ = blocks(problem, response["fixed"], response["innovations"])
        _, local_blocks, _ = blocks(problem, response["local"], response["innovations"])
        fixed_draws, refit_draws = [], []
        for fit in details[selection]["noise_draw_fits"]:
            draw = {key: flat[key][fit["draw_index"]] for key in noise}
            draw_cov = [covariance(state, group, draw, k) for k, (state, group) in enumerate(zip(problem["states"], groups))]
            fixed_draws.append(evaluate(problem, draw_cov, maps, parameters)["residual"] * 1e6)
            # Full-data nuisance fit already performed for this identical draw.
            refit_draws.append(evaluate(problem, draw_cov, maps, fit["parameters"])["residual"] * 1e6)
        matched = summarize([-1., 1.], {key: value * 1e6 for key, value in values.items()},
            response["fixed_covariance"] * 1e12, response["local_covariance"] * 1e12,
            fixed_blocks * 1e12, local_blocks * 1e12, definitions, np.array(fixed_draws), np.array(refit_draws))
        matched.update(map_validation=validation, rows=matched_rows, row_residuals_ppm=(values["residual"] * 1e6).tolist(),
            fixed_covariance_ppm2=(response["fixed_covariance"] * 1e12).tolist(), local_covariance_ppm2=(response["local_covariance"] * 1e12).tolist())
        result[selection] = dict(within_phase_blue_minus_core=within, matched_core_night1_minus_night0=matched)
    receipt = dict(scope=__doc__, input_source_sha256=pins, unchanged_input_source_hashes=all(digest(path) == sha for path, sha in pins.items()), results=result,
        limits=["Two-SD proxy envelopes are descriptive across block offsets; they are not calibrated or simultaneous confidence intervals.",
                "The two nights retain different original common references and conditional GP fits. Matched phase/band support does not by itself isolate atmospheric variability.",
                "Local-refit covariance retains the original fixed center, including for alternate pixels. Noise-draw refits reuse the identical full-data fits from the main audit."])
    (output / "contrasts.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({selection: result[selection]["matched_core_night1_minus_night0"] for selection in result}), flush=True)


if __name__ == "__main__":
    main()
