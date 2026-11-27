"""Test smooth stellar/calibration proxies against the native phase residuals.

Three low-dimensional families are fitted: a PHOENIX line-depth modulation at
the published 7.58695-hour pulsation period, a line-width drift, and broad
curvature drifting linearly with time. Orders share coefficients within each night. These are
conditional nuisance experiments, not identifications of stellar pulsations
or a reconstruction of the extraction/calibration pipeline.
"""

from pathlib import Path
import argparse
import json

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from cloudy_halpha import stellar_reference
from gp_halpha import load_gp_data, make_gp_problem, PERIOD_MINUTES
from halpha_noise_audit import centered, covariance, digest
from halpha_phase_uncertainty import apply_precision
from halpha_phase_controls import solve_scaled
from escape.time_gp import prepare_time_gp, time_gp_loglike
from escape.observation import C_KMS

PULSE_MINUTES = 7.58695 * 60
SCALES = dict(depth=.01, width=.01, stellar_shift=.1, calibration_shift=.1, curvature=.001)
FAMILIES = dict(stellar_depth_pulse=("depth", "pulse"), stellar_width_drift=("width", "drift"), curvature_drift=("curvature", "drift"))


def frame_velocity(group):
    orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
    return C_KMS * ((1 + group["stellar_velocity"][None] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)


def time_basis(group, kind):
    time = group["time_min"]
    if kind == "drift":
        return ((time - time.mean()) / 100.)[:, None]
    angle = 2 * np.pi * time / PULSE_MINUTES
    return np.stack([np.sin(angle), np.cos(angle)], axis=1)


def stellar_terms(group, model, surface, parameters):
    """Reproduce the existing two-stage interpolation and continuum mean."""
    velocity = model["model_velocity_kms"]
    sampled = np.stack([np.interp(group["phase"], model["phase"], column) for column in model["fine_intrinsic_phases"].T], axis=1)
    sampled[~group["science"]] = 0.
    shifted = np.stack([np.interp(velocity - parameters[0], velocity, row) for row in sampled]) * np.exp(parameters[1])
    normal = np.linspace(-600., 600., 2401)
    normal = normal[abs(normal) > 400.]
    orbital = 244.8 * np.sin(2 * np.pi * group["phase"])

    def planet(coordinate):
        coordinate = np.asarray(coordinate)
        if coordinate.ndim == 1:
            coordinate = coordinate[None]
        query = C_KMS * ((1 + coordinate / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)
        return np.stack([np.interp(x, velocity, row) for x, row in zip(query, shifted)])

    def star_at(coordinate):
        coordinate = np.asarray(coordinate)
        return stellar_reference(coordinate.ravel(), surface, 110.).reshape(coordinate.shape)

    coordinate = group["stellar_velocity"]
    star, star_normal = [stellar_reference(x, surface, 110.) for x in (coordinate, normal)]
    planet_native, planet_normal = planet(coordinate), planet(normal)
    total = star[None] + planet_native
    total_normal = star_normal[None] + planet_normal
    normalization = total_normal.mean(axis=1)
    continuum_fit = np.polyfit(normal, star_normal, 1)
    continuum = lambda x: np.polyval(continuum_fit, x)
    depth = star - continuum(coordinate)
    depth_normal = star_normal - continuum(normal)
    step, width_step = .1, .001

    def shifted_star(x):
        return (stellar_reference(x - step, surface, 110.) - stellar_reference(x + step, surface, 110.)) / (2 * step)

    def width_star(x):
        left, right = x / (1 + width_step), x / (1 - width_step)
        return continuum(x) * (stellar_reference(left, surface, 110.) / continuum(left)
            - stellar_reference(right, surface, 110.) / continuum(right)) / (2 * width_step)

    def normalized_derivative(value, normal_value):
        return value[None] / total - normal_value.mean() / normalization[:, None]

    def shifted_total(offset):
        return np.log((stellar_reference(coordinate - offset, surface, 110.)[None] + planet(coordinate - offset))
            / (stellar_reference(normal - offset, surface, 110.)[None] + planet(normal - offset)).mean(axis=1)[:, None])

    q = .5 * (3 * ((coordinate - 250) / 140)**2 - 1)
    q_normal = .5 * (3 * ((normal - 250) / 140)**2 - 1)
    templates = dict(depth=normalized_derivative(depth, depth_normal),
        width=normalized_derivative(width_star(coordinate), width_star(normal)),
        stellar_shift=normalized_derivative(shifted_star(coordinate), shifted_star(normal)),
        calibration_shift=(shifted_total(step) - shifted_total(-step)) / (2 * step),
        curvature=q[None] - (total_normal * q_normal[None]).mean(axis=1)[:, None] / normalization[:, None])
    templates = {key: value * SCALES[key] for key, value in templates.items()}

    def exact(kind, modulation):
        if kind == "depth":
            numerator = total + modulation[:, None] * depth[None]
            denominator = normalization + modulation * depth_normal.mean()
        elif kind == "width":
            if np.any(1 + modulation <= 0):
                return None
            query = coordinate[None] / (1 + modulation[:, None])
            query_normal = normal[None] / (1 + modulation[:, None])
            numerator = continuum(coordinate)[None] * star_at(query) / continuum(query) + planet_native
            denominator = (continuum(normal)[None] * star_at(query_normal) / continuum(query_normal) + planet_normal).mean(axis=1)
        elif kind in ("stellar_shift", "calibration_shift"):
            query, query_normal = coordinate[None] - modulation[:, None], normal[None] - modulation[:, None]
            numerator = star_at(query) + (planet(query) if kind == "calibration_shift" else planet_native)
            denominator = (star_at(query_normal) + (planet(query_normal) if kind == "calibration_shift" else planet_normal)).mean(axis=1)
        else:
            exponent = modulation[:, None] * q_normal[None]
            # Stable normalizer; a constant per exposure is not discarded silently.
            anchor = np.max(exponent, axis=1)
            log_norm = anchor + np.log(np.mean(total_normal * np.exp(exponent - anchor[:, None]), axis=1))
            return modulation[:, None] * q[None] - log_norm[:, None] + np.log(normalization)[:, None]
        if kind != "curvature":
            if np.any(numerator <= 0) or np.any(denominator <= 0):
                return None
            return np.log(numerator / denominator[:, None]) - np.log(total / normalization[:, None])

    return dict(templates=templates, exact=exact, log_model=np.log(total / normalization[:, None]),
                star=star, continuum=continuum(coordinate), normalizer=normalization)


def design(groups, terms, mode, kind):
    count = 1 if kind == "drift" else 2
    result = []
    for group, term in zip(groups, terms):
        value = np.zeros((*group["weight"].shape, 2 * count))
        start = group["night"] * count
        value[:, :, start:start + count] = term["templates"][mode][:, :, None] * time_basis(group, kind)[:, None]
        result.append(value)
    return result


def project_field(groups, states, covs, maps, fields):
    result = np.zeros((maps[0].shape[0], fields[0].shape[-1]))
    for group, state, cov, mapping, value in zip(groups, states, covs, maps, fields):
        product = np.einsum("ep,mp,epk->mek", group["weight"], state.basis,
            np.stack([centered(group, value[:, :, k]) for k in range(value.shape[-1])], axis=-1))
        coefficients = (cov["estimator"] @ product.reshape(len(cov["estimator"]), -1)).reshape(product.shape)
        gp = np.einsum("mek,mp->epk", coefficients, state.basis)
        result += mapping.reshape(len(mapping), -1) @ (value - gp).reshape(-1, value.shape[-1])
    return result


def prepare(groups, noise, mask=None):
    selected, states, covs = [], [], []
    for k, original in enumerate(groups):
        group = dict(original)
        if mask is not None:
            group["weight"] = original["weight"] * mask[k]
        x = (group["stellar_velocity"] - 250) / 140
        state = prepare_time_gp(group["weight"], np.array([np.ones_like(x), x]))
        group["weight"] = np.asarray(state.weight)
        selected.append(group); states.append(state); covs.append(covariance(state, group, noise, k))
    return selected, states, covs


def statistics(setup, residuals, basis):
    groups, states, covs = setup
    precision = [apply_precision(g, s, c, d) for g, s, c, d in zip(groups, states, covs, basis)]
    gram = sum(np.einsum("epk,epl->kl", d, p) for d, p in zip(basis, precision))
    rhs = sum(np.einsum("ep,epk->k", r, p) for r, p in zip(residuals, precision))
    q = sum(float(np.sum(r * apply_precision(g, s, c, r))) for g, s, c, r in zip(groups, states, covs, residuals))
    return gram, rhs, q


def gain(stats, beta):
    gram, rhs, _ = stats
    return float(2 * beta @ rhs - beta @ gram @ beta)


def phase_values(values, maps):
    supported = np.any(np.stack([np.any(mapping != 0, axis=(1, 2)) for mapping in maps]), axis=0)
    return [float(value) if keep else None for value, keep in zip(values, supported)]


def masks(groups, kind):
    output = []
    for group in groups:
        if kind == "outside_phase":
            keep = ~((group["phase"] >= .375) & (group["phase"] < .4))
            output.append(np.broadcast_to(keep[:, None], group["weight"].shape))
        elif kind == "outside_planet":
            output.append((abs(frame_velocity(group)) >= 60.) | group["eclipse"][:, None])
        elif kind == "science_only":
            output.append(np.broadcast_to(group["science"][:, None], group["weight"].shape))
        else:
            output.append(np.full_like(group["weight"], group["order"] == int(kind[-1]), dtype=bool))
    return output


def shape_screen(groups, terms, states, covs, maps, residuals, target, target_cov):
    """Match three target band values only, then expose the full-data cost."""
    records = []
    lower = la.cholesky(target_cov, lower=True)
    for mode in SCALES:
        for kind in ("drift", "pulse"):
            basis = design(groups, terms, mode, kind)
            response = project_field(groups, states, covs, maps, basis)
            columns = 1 if kind == "drift" else 2  # Target has night0 support only.
            a = la.solve_triangular(lower, response[9:12, :columns], lower=True)
            y = la.solve_triangular(lower, target, lower=True)
            local, _, rank, singular = np.linalg.lstsq(a, y, rcond=1e-10)
            beta = np.r_[local, np.zeros(columns)]
            stats = statistics((groups, states, covs), residuals, basis)
            response_value = response @ beta
            modulation = time_basis(groups[0], kind) @ local * SCALES[mode]
            record = dict(mode=mode, time_model=kind, target_only_coefficients=beta.tolist(), rank=int(rank),
                singular_values=singular.tolist(), maximum_night0_physical_modulation=float(np.max(abs(modulation))),
                unit="fractional line-depth/width" if mode in ("depth", "width") else "km/s" if "shift" in mode else "log-curvature coefficient",
                target_response_ppm=(response_value[9:12] * 1e6).tolist(), target_remaining_residual_ppm=((target - response_value[9:12]) * 1e6).tolist(),
                full_data_delta_Q_improvement=gain(stats, beta),
                phase_responses_ppm=phase_values(response_value * 1e6, maps),
                scope="Three-band flexibility screen only; target-selected amplitudes are not inferred physical parameters.")
            exact_fields = []
            for group, term in zip(groups, terms):
                start = group["night"] * columns
                modulation = time_basis(group, kind) @ beta[start:start + columns] * SCALES[mode]
                exact_fields.append(term["exact"](mode, modulation))
            record["exact_positive_flux"] = all(value is not None and np.isfinite(value).all() for value in exact_fields)
            if record["exact_positive_flux"]:
                exact_response = project_field(groups, states, covs, maps, [value[:, :, None] for value in exact_fields])[:, 0]
                q_exact = sum(float(np.sum((r - d) * apply_precision(g, s, c, r - d))) for g, s, c, r, d in zip(groups, states, covs, residuals, exact_fields))
                record["exact_target_response_ppm"] = (exact_response[9:12] * 1e6).tolist()
                record["exact_full_data_delta_Q_improvement"] = stats[2] - q_exact
                record["linearization_max_native_logflux_error_ppm"] = max(float(np.max(abs(value - np.einsum("epk,k->ep", d, beta))) * 1e6) for value, d in zip(exact_fields, basis))
            records.append(record)
    return records


def fit_families(contexts, noise):
    primary, alternate = contexts["primary"], contexts["alternate"]
    results, plot = {}, {}
    for family, (mode, kind) in FAMILIES.items():
        basis = {name: design(item["groups"], item["terms"], mode, kind) for name, item in contexts.items()}
        stats = {name: statistics((item["groups"], item["states"], item["covs"]), item["residuals"], basis[name]) for name, item in contexts.items()}
        responses = {name: project_field(item["groups"], item["states"], item["covs"], item["maps"], basis[name]) for name, item in contexts.items()}
        trials = {}
        for training in ("full", "outside_phase", "outside_planet", "science_only", "order0", "order1"):
            train_stats = stats["primary"] if training == "full" else statistics(prepare(primary["groups"], noise, masks(primary["groups"], training)), primary["residuals"], basis["primary"])
            beta, beta_cov, rank = solve_scaled(train_stats[0], train_stats[1])
            row = dict(coefficients=beta.tolist(), conditional_coefficient_sigma=np.sqrt(np.maximum(np.diag(beta_cov), 0.)).tolist(), rank=rank,
                training_delta_Q_improvement=gain(train_stats, beta), full_primary_delta_Q_improvement=gain(stats["primary"], beta),
                alternate_pixel_delta_Q_improvement=gain(stats["alternate"], beta),
                heldout_conditional_log_score_gain=None if training == "full" else .5 * (gain(stats["primary"], beta) - gain(train_stats, beta)),
                primary_phase_response_ppm=phase_values(responses["primary"] @ beta * 1e6, primary["maps"]),
                alternate_phase_response_ppm=phase_values(responses["alternate"] @ beta * 1e6, alternate["maps"]))
            count = 1 if kind == "drift" else 2
            modulations = [time_basis(group, kind) @ beta[group["night"] * count:(group["night"] + 1) * count] * SCALES[mode] for group in primary["groups"]]
            row["maximum_modulation_by_night"] = [float(np.max(abs(modulations[i]))) for i in (0, 2)]
            row["within_declared_small_perturbation_range"] = bool(max(row["maximum_modulation_by_night"]) <= (.1 if mode in ("depth", "width") else .01))
            row["small_perturbation_range_scope"] = "Diagnostic linear-response guard: 10% line-depth/width change or 0.01 log-curvature coefficient; not an observational calibration prior."
            for selection, item in contexts.items():
                row[selection + "_target_remaining_residual_ppm"] = ((item["baseline_bands"][9:12] - responses[selection][9:12] @ beta) * 1e6).tolist()
            if training == "full":
                exact_fields = [term["exact"](mode, modulation) for term, modulation in zip(primary["terms"], modulations)]
                linear = [np.einsum("epk,k->ep", d, beta) for d in basis["primary"]]
                direct_linear_gain = 0.
                for k, (group, state, residual, field) in enumerate(zip(primary["groups"], primary["states"], primary["residuals"], linear)):
                    arguments = (state, group["time_min"], np.exp(noise["log_gp"][k]), np.exp(noise["log_length"][k]), np.exp(noise["log_white"][k]))
                    direct_linear_gain += 2 * float(time_gp_loglike(residual - field, *arguments) - time_gp_loglike(residual, *arguments))
                row["direct_production_delta_Q_improvement"] = direct_linear_gain
                row["direct_production_delta_Q_absolute_difference"] = abs(direct_linear_gain - row["full_primary_delta_Q_improvement"])
                row["exact_positive_flux"] = all(field is not None and np.isfinite(field).all() for field in exact_fields)
                if row["exact_positive_flux"]:
                    row["linearization_max_native_logflux_error_ppm"] = max(float(np.max(abs(a - b)) * 1e6) for a, b in zip(exact_fields, linear))
                    projected = project_field(primary["groups"], primary["states"], primary["covs"], primary["maps"], [value[:, :, None] for value in exact_fields])[:, 0]
                    row["exact_target_remaining_residual_ppm"] = ((primary["baseline_bands"][9:12] - projected[9:12]) * 1e6).tolist()
                    row["exact_vs_linear_target_max_absolute_difference_ppm"] = float(np.max(abs(np.array(row["exact_target_remaining_residual_ppm"]) - row["primary_target_remaining_residual_ppm"])))
                    new_residual = [r - field for r, field in zip(primary["residuals"], exact_fields)]
                    _, _, q_exact = statistics((primary["groups"], primary["states"], primary["covs"]), new_residual, basis["primary"])
                    row["exact_full_primary_delta_Q_improvement"] = stats["primary"][2] - q_exact
                    direct_bands = project_field(primary["groups"], primary["states"], primary["covs"], primary["maps"], [value[:, :, None] for value in new_residual])[:, 0]
                    row["independent_exact_residual_reprojection_max_absolute_difference_ppm"] = float(np.max(abs(direct_bands - primary["baseline_bands"] + projected)) * 1e6)
                plot[family] = {selection: contexts[selection]["baseline_bands"] * 1e6 - responses[selection] @ beta * 1e6 for selection in contexts}
            trials[training] = row
        results[family] = dict(mode=mode, time_model=kind, parameter_count=len(basis["primary"][0][0, 0]), trials=trials,
            interpretation="Delta Q is descriptive at fixed GP and planet parameters. Held-out gains condition on archived noise fitted using the full data; alternate pixels are correlated sensitivity data.")
    return results, plot


def plot_results(output, contexts, fits, curves):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.), layout="constrained")
    group, term = contexts["primary"]["groups"][0], contexts["primary"]["terms"][0]
    exposure = int(np.argmin(abs(group["phase"] - .384)))
    for key in ("depth", "width", "calibration_shift", "curvature"):
        row = term["templates"][key][exposure]
        # Remove a grey level only for this shape display.
        axes[0, 0].plot(group["stellar_velocity"], (row - row.mean()) * 1e6, label=key.replace("_", " "))
    axes[0, 0].set(xlabel="Stellar velocity (km/s)", ylabel="Unit proxy shape minus grey (ppm)", title="PHOENIX and broad continuum templates")
    axes[0, 0].legend(fontsize=8)
    x = np.arange(3)
    for selection, style in (("primary", "o-"), ("alternate", "s--")):
        axes[0, 1].plot(x, contexts[selection]["baseline_bands"][9:12] * 1e6, style, color="k", label=selection + " baseline")
        for family, value in curves.items():
            axes[0, 1].plot(x, value[selection][9:12], style, label=selection + " " + family.replace("_", " "))
    axes[0, 1].axhline(0, color="0.5", lw=.7)
    axes[0, 1].set(xticks=x, xticklabels=["Blue", "Core", "Red"], ylabel="Remaining residual (ppm)", title="0.376–0.393: full-data nuisance fits")
    axes[0, 1].legend(fontsize=6)
    phase = np.array([.3125, .3375, .3625, .3875, .4125, .43])
    for selection, style in (("primary", "o-"), ("alternate", "s--")):
        baseline = np.array(phase_values(contexts[selection]["baseline_bands"] * 1e6, contexts[selection]["maps"]), dtype=float).reshape(6, 3)
        axes[1, 0].plot(phase, baseline[:, 0] - baseline[:, 1], style, color="k", label=selection + " baseline")
        for family, value in curves.items():
            row = np.array(phase_values(value[selection], contexts[selection]["maps"]), dtype=float).reshape(6, 3)
            axes[1, 0].plot(phase, row[:, 0] - row[:, 1], style, label=selection + " " + family.replace("_", " "))
    axes[1, 0].axhline(0, color="0.5", lw=.7)
    axes[1, 0].axvspan(.4, .435, color="0.9")
    axes[1, 0].set_xlim(.3, .435)
    axes[1, 0].set(xlabel="Phase bin center (complete support only)", ylabel="Blue − core residual (ppm)", title="Collateral response at earlier phases")
    for j, (family, value) in enumerate(fits.items()):
        labels = ["outside_phase", "outside_planet", "science_only", "order0", "order1"]
        gains = [value["trials"][key]["heldout_conditional_log_score_gain"] for key in labels]
        axes[1, 1].bar(np.arange(5) + (j - (len(fits) - 1) / 2) * .25, gains, .25, label=family.replace("_", " "))
    axes[1, 1].axhline(0, color="0.5", lw=.7)
    axes[1, 1].set_yscale("symlog", linthresh=3.)
    axes[1, 1].set(xticks=np.arange(5), xticklabels=["phase", "planet stripe", "eclipse", "order 1", "order 0"], ylabel="Conditional log-score gain (symlog)", title="Score the named held-out data")
    axes[1, 1].legend(fontsize=7)
    fig.suptitle("Smooth stellar/calibration proxies: conditional flexibility and prediction checks", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output / ("systematics." + extension), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    model_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    stellar_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    noise_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    map_path = Path("/tmp/escape-halpha-phase-localization/maps.npz")
    row_path = Path("/tmp/escape-halpha-phase-localization/rows.json")
    cov_path = Path("/tmp/escape-halpha-phase-uncertainty/arrays.npz")
    sources = [Path(__file__), *[root / "examples" / key for key in ("gp_halpha.py", "cloudy_halpha.py", "phase_halpha.py", "halpha_noise_audit.py", "halpha_phase_uncertainty.py", "halpha_phase_controls.py")], root / "src/escape/time_gp.py"]
    pins = {str(path): digest(path) for path in [*paths, model_path, stellar_path, noise_path, map_path, row_path, cov_path, *sources]}
    model, surface = dict(np.load(model_path)), dict(np.load(stellar_path))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in dict(np.load(noise_path)).items()}
    metadata, maps = json.loads(row_path.read_text()), np.load(map_path)
    parameters = metadata["parameters"]
    contexts = {}
    for selection, offset in (("primary", 0), ("alternate", 1)):
        groups = load_gp_data(paths, pixel_offset=offset)
        problem = make_gp_problem(groups, model, surface)
        groups, states, covs = prepare(groups, noise)
        native = [np.asarray(value) for value in problem["native_log_models"](jnp.asarray(parameters))]
        terms = [stellar_terms(group, model, surface, parameters) for group in groups]
        parity = max(float(np.max(abs(value - term["log_model"]))) for value, term in zip(native, terms))
        if parity > 1e-12:
            raise ValueError(f"Native stellar/planet interpolation differs: {parity}")
        selected_maps = [maps[f"{selection}__g{i}__maps"][:18] for i in range(4)]
        residuals = [group["log_data"] - value for group, value in zip(groups, native)]
        bands = project_field(groups, states, covs, selected_maps, [r[:, :, None] for r in residuals])[:, 0]
        expected = metadata["selections"][selection]["values_ppm"][:18]
        error = max(abs(bands[i] * 1e6 - row["residual"]) for i, row in enumerate(expected) if row["residual"] is not None)
        if error > 1e-7:
            raise ValueError("Phase-band response does not match the fixed baseline")
        contexts[selection] = dict(groups=groups, states=states, covs=covs, terms=terms, maps=selected_maps,
            residuals=residuals, baseline_bands=bands, native_parity=parity, target_parity_ppm=error)
    target_cov = np.load(cov_path)["primary__fixed_covariance_ppm2"][9:12, 9:12] / 1e12
    item = contexts["primary"]
    screen = shape_screen(item["groups"], item["terms"], item["states"], item["covs"], item["maps"], item["residuals"], item["baseline_bands"][9:12], target_cov)
    fits, curves = fit_families(contexts, noise)
    plot_results(args.output, contexts, fits, curves)
    output = dict(scope=__doc__, pulsation_period_minutes=PULSE_MINUTES, representative_template_scales=SCALES,
        fixed_parameters=parameters, source_input_sha256=pins, unchanged_source_input_hashes=all(digest(path) == sha for path, sha in pins.items()),
        baseline={name: dict(target_residual_ppm=(item["baseline_bands"][9:12] * 1e6).tolist(), native_interpolation_parity=item["native_parity"], baseline_reconstruction_max_abs_logflux_ppm=item["native_parity"] * 1e6, phase_map_parity_ppm=item["target_parity_ppm"]) for name, item in contexts.items()},
        analytic_target_only_screen=screen, families=fits,
        references=[dict(url="https://authors.library.caltech.edu/records/6ee28-5vn96", claim="Wong et al. (2020) measured a photometric pulsation period of 7.58695 +/- 0.00091 hours; this fixes a proxy timescale, not its H-alpha amplitude or 2024 phase."),
                    dict(url="https://arxiv.org/abs/2004.13733", claim="Wyttenbach et al. (2020) reported spectroscopic stellar pulsations with period 7.54 +/- 0.12 hours in different observations.")],
        limitations=["The saved PHOENIX photosphere and uniform rotation110 are unchanged baseline choices; depth/width morphs are not time-dependent stellar-atmosphere calculations.",
            "The depth/width/shift target-only screen deliberately uses target information; its full-data collateral cost is reported. It is not evidence for the screened cause.",
            "Stellar-depth pulse and broad-curvature drift were the initial low-dimensional fits. Width drift was added because the target-only screen gave it the most compatible target sign pattern; this is exploratory model selection. Curvature is a throughput/continuum proxy, not a measured calibration error.",
            "Period is fixed from other epochs. Independent nightly sine/cosine coefficients allow unknown phases because absolute timing/telemetry is unavailable.",
            "Archived noise hyperparameters used all data; holdout scores are conditional diagnostics. Adjacent pixel interleaves are correlated, and cross-order independence is not established.",
            "Q gains do not include nuisance-prior/complexity penalties and are not Bayes factors or calibrated detections."])
    output["figure_sha256"] = {path.name: digest(path) for path in args.output.glob("systematics.*")}
    (args.output / "review.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(baseline=output["baseline"], families=fits)), flush=True)


if __name__ == "__main__":
    main()
