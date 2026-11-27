"""Audit H-alpha shape stability using the archived data and fixed GP settings.

Sample-selection changes are descriptive sensitivities, not likelihood ratios.
The observed profiles contain no physical template. Conditional GP profiles and
residuals retain the archived baseline atmosphere and fixed RV/amplitude.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import jax
import numpy as np
import scipy.linalg as la

from gp_halpha import load_gp_data, make_gp_problem
from phase_halpha import REST_ANGSTROM
from escape.observation import C_KMS
from escape.time_gp import time_gp_posterior

jax.config.update("jax_enable_x64", True)
WINDOWS = {"blue": (-40, -16), "core": (-8, 8), "red": (16, 40)}
PARAMETERS = np.array([2.8601527471258423, .12403235289413468])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_group(group, noise):
    """The existing projected grey/slope GP, evaluated with NumPy/SciPy."""
    weight, data, model = (np.asarray(group[k]) for k in ("weight", "log_data", "model"))
    basis = np.array([np.ones(weight.shape[1]), (group["stellar_velocity"] - 250) / 140])
    total = weight.sum(axis=0)
    common = weight / np.where(total > 0, total, 1)
    centered = lambda x: np.where(weight > 0, x - (common * x).sum(axis=0), 0)
    modes, exposures = basis.shape[0], weight.shape[0]
    design = (basis[:, None, :] * weight[None]).reshape(modes * exposures, -1)
    diagonal = np.einsum("ep,mp,np->mne", weight, basis, basis)
    gram = np.einsum("mne,ef->menf", diagonal, np.eye(exposures)).reshape(modes * exposures, -1)
    gram -= (design / np.where(total > 0, total, 1)) @ design.T
    gram = (gram + gram.T) / 2
    k = group["night"] * 2 + group["order"]
    scale = np.exp(noise["log_white"][k])
    distance = np.sqrt(3) * abs(group["time_min"][:, None] - group["time_min"][None]) / np.exp(noise["log_length"][k])
    correlation = (1 + distance) * np.exp(-distance) + 1e-10 * np.eye(exposures)
    factor = la.block_diag(*[a * la.cholesky(correlation, lower=True) for a in np.exp(noise["log_gp"][k])])
    system = np.eye(len(factor)) + factor.T @ gram @ factor / scale**2
    residuals = [centered(data - model), centered(data)]
    products = np.stack([np.einsum("ep,ep,mp->me", weight, x, basis).ravel() for x in residuals], axis=1)
    coefficients = factor @ la.solve(system, factor.T @ products / scale**2, assume_a="pos")
    gp, zero_gp = [(coefficients[:, i].reshape(modes, exposures).T @ basis) for i in range(2)]
    q = np.sum(weight * residuals[0]**2) / scale**2 - products[:, 0] @ coefficients[:, 0] / scale**2
    return dict(observed=data, planet=model, gp=gp, zero_planet_gp=zero_gp,
                residual=data - model - gp, centered_observed=centered(data),
                centered_gp=centered(gp), centered_residual=centered(data - model - gp),
                Q=float(q), white_scale=float(scale), common=common)


def project(group, fields, edges, reference="common"):
    orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
    velocity = C_KMS * ((1 + group["stellar_velocity"][None] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)
    index = np.searchsorted(edges, velocity, side="right") - 1
    weight = group["weight"]
    coefficient = np.array([weight * group["science"][:, None] * (index == b) for b in range(len(edges) - 1)])
    if reference == "common":
        ref = fields["common"]
    else:
        keep = group["eclipse"] & (group["phase"] >= reference[0]) & (group["phase"] < reference[1])
        ref_weight = weight * keep[:, None]
        total = ref_weight.sum(axis=0)
        coefficient *= (total > 0)[None, None]
        ref = ref_weight / np.where(total > 0, total, 1)
    projected = coefficient - ref[None] * coefficient.sum(axis=1)[:, None]
    variance = fields["white_scale"]**2 * np.sum(projected**2 / np.where(weight > 0, weight, np.inf), axis=(1, 2))
    out = {key: np.einsum("bep,ep->b", projected, fields[key]) for key in
           ("observed", "planet", "gp", "zero_planet_gp", "residual")}
    out.update(weight=coefficient.sum(axis=(1, 2)), variance=variance)
    by_exposure = coefficient.sum(axis=2)
    return out, by_exposure, velocity


def combine(sums, edges):
    total = {key: sum(row[key] for row in sums) for key in sums[0]}
    weight = total["weight"]
    norm = np.where(weight > 0, weight, 1)
    result = {key: total[key] / norm * 1e6 for key in ("observed", "planet", "gp", "zero_planet_gp", "residual")}
    result.update(velocity=(edges[:-1] + edges[1:]) / 2, weight=weight,
                  white_error=np.sqrt(total["variance"]) / norm * 1e6)
    return result


def bands(row):
    output = {}
    for name, (lo, hi) in WINDOWS.items():
        keep = (row["velocity"] >= lo) & (row["velocity"] < hi) & (row["weight"] > 0) & (row["white_error"] > 0)
        w = 1 / row["white_error"][keep]**2
        output[name] = {key: float(np.average(row[key][keep], weights=w)) if keep.any() else None
                        for key in ("observed", "planet", "gp", "zero_planet_gp", "residual")}
        output[name].update(bin_count=int(keep.sum()), white_error_diagnostic=float(1 / np.sqrt(w.sum())) if keep.any() else None)
    output["blue_minus_core"] = {key: output["blue"][key] - output["core"][key]
        if output["blue"][key] is not None and output["core"][key] is not None else None
        for key in ("observed", "planet", "gp", "zero_planet_gp", "residual")}
    return output


def evaluate(groups, noise, edges, reference="common", details=False):
    sums, fields, support = [], [], {}
    for group in groups:
        field = solve_group(group, noise)
        row, exposures, velocity = project(group, field, edges, reference)
        sums.append(row)
        fields.append(field)
        if details:
            entries = []
            for b, v in enumerate((edges[:-1] + edges[1:]) / 2):
                w = exposures[b]
                active = w > 0
                entries.append(dict(velocity_kms=float(v), exposure_count=int(active.sum()),
                    native_sample_count=int(np.sum((group["weight"] > 0) & group["science"][:, None] & (velocity >= edges[b]) & (velocity < edges[b + 1]))),
                    exposure_ids=group["exposure_id"][active].tolist(), phases=group["phase"][active].tolist(),
                    exposure_weight_fraction=(w[active] / w.sum()).tolist() if active.any() else [],
                    effective_exposures=float(w.sum()**2 / (w @ w)) if active.any() else 0))
            support[group["label"]] = entries
    selections = {"all": list(range(len(groups)))}
    selections.update({f"night{night}": [i for i, g in enumerate(groups) if g["night"] == night] for night in (0, 1)})
    selections.update({group["label"]: [i] for i, group in enumerate(groups)})
    profiles = {key: combine([sums[i] for i in ids], edges) for key, ids in selections.items() if ids}
    summary = dict(bands_ppm={key: bands(row) for key, row in profiles.items()},
                   group_Q={group["label"]: field["Q"] for group, field in zip(groups, fields)},
                   science_exposures={group["label"]: int(group["science"].sum()) for group in groups},
                   eclipse_exposures={group["label"]: int(group["eclipse"].sum()) for group in groups})
    if details:
        summary["bin_support"] = support
    return summary, profiles, fields


def select(groups, phase_range=None, drop=None, lower_velocity=110., high_rms_mask=False):
    output = []
    for original in groups:
        keep = np.ones(len(original["phase"]), dtype=bool)
        if phase_range:
            keep &= ~original["science"] | ((original["phase"] >= phase_range[0]) & (original["phase"] < phase_range[1]))
        if drop and original["night"] == drop[0]:
            keep &= ~np.isin(original["exposure_id"], drop[1])
        if not np.any(keep & original["science"]):
            continue
        pixels = original["stellar_velocity"] >= lower_velocity
        group = dict(original)
        for key in ("phase", "science", "eclipse", "time_min", "exposure_id"):
            group[key] = original[key][keep]
        for key in ("weight", "log_data", "model"):
            group[key] = original[key][keep][:, pixels].copy()
        for key in ("stellar_velocity", "pixel_index"):
            group[key] = original[key][pixels]
        if high_rms_mask:
            center = group["log_data"] - np.sum(group["weight"] * group["log_data"], axis=0) / np.maximum(group["weight"].sum(axis=0), 1)
            rms = np.sum(group["weight"] * center**2, axis=0) / np.maximum((group["weight"] > 0).sum(axis=0), 1)
            group["weight"][:, rms > np.quantile(rms, .99)] = 0
        group["weight"][:, (group["weight"] > 0).sum(axis=0) < 2] = 0
        output.append(group)
    return output


def load_groups(paths, archive, surface, stride=2, offset=0):
    groups = load_gp_data(paths, pixel_stride=stride, pixel_offset=offset)
    problem = make_gp_problem(groups, archive, surface)
    models = problem["native_log_models"](PARAMETERS)
    for group, model in zip(groups, models):
        group["model"] = np.asarray(model)
        with h5py.File(paths[group["night"]]) as source:
            original_phase = source["phase"][:]
        group["exposure_id"] = np.searchsorted(original_phase, group["phase"])
    return groups, problem


def plot_results(output, profiles, groups, fields, edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
    for col, selection in enumerate(["all", "night0", "night1"]):
        for label, color in [("primary", "#0072B2"), ("alternate", "#D55E00"), ("all_pixels", ".3")]:
            row = profiles[label][selection]
            keep = (row["velocity"] >= -60) & (row["velocity"] <= 60) & (row["weight"] > 0)
            axes[0, col].errorbar(row["velocity"][keep], row["observed"][keep], row["white_error"][keep], fmt=".-", lw=.8, ms=3, color=color, label=label.replace("_", " "))
            axes[1, col].plot(row["velocity"][keep], row["residual"][keep], ".-", lw=.8, color=color)
        axes[0, col].set_title({"all": "Both nights", "night0": "August 8: no eclipse exposures", "night1": "August 20: late science + eclipse"}[selection])
        axes[1, col].set_xlabel("Planet-frame velocity (km/s)")
        for ax in axes[:, col]:
            ax.axhline(0, color=".6", lw=.5)
            ax.axvspan(-40, -16, color="#0072B2", alpha=.07)
            ax.axvspan(-8, 8, color=".5", alpha=.08)
    axes[0, 0].set_ylabel("Observed contrast (ppm)")
    axes[1, 0].set_ylabel("Baseline + GP residual (ppm)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Interleaved pixels: observed common-reference profile; white display errors only")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"pixel_night_comparison.{extension}", dpi=180)
    plt.close(fig)

    exposure_arrays = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", layout="constrained")
    for group, field in zip(groups, fields):
        night = group["night"]
        _, weights, velocity = project(group, field, edges)
        for name, (lo, hi) in WINDOWS.items():
            centers = (edges[:-1] + edges[1:]) / 2
            w = weights[(centers >= lo) & (centers < hi)].sum(axis=0)
            if w.sum():
                axes[night, 0].plot(group["phase"], w / w.sum(), ".-", lw=.6,
                    label=f"{name}, order {group['order']}")
        for name in ["observed", "gp", "residual"]:
            arr = {"observed": field["centered_observed"], "gp": field["centered_gp"], "residual": field["centered_residual"]}[name]
            values, blue_values, core_values = [], [], []
            for row in range(len(group["phase"])):
                means = []
                for lo, hi in [WINDOWS["blue"], WINDOWS["core"]]:
                    w = group["weight"][row] * (velocity[row] >= lo) * (velocity[row] < hi)
                    means.append(np.sum(w * arr[row]) / w.sum() * 1e6 if w.sum() else np.nan)
                values.append(means[0] - means[1])
                blue_values.append(means[0])
                core_values.append(means[1])
            science = group["science"]
            for key, value in [(name + "__blue_minus_core", values), (name + "__blue", blue_values), (name + "__core", core_values)]:
                exposure_arrays[group["label"] + "__" + key] = np.asarray(value)
            exposure_arrays[group["label"] + "__phase"] = group["phase"]
            exposure_arrays[group["label"] + "__science"] = science
            axes[night, 1].plot(group["phase"][science], np.asarray(values)[science], ".-", lw=.6, ms=2,
                alpha=.7, label=f"{name}, order {group['order']}")
        axes[night, 0].set_ylabel(f"Night {night}: band weight fraction")
        axes[night, 1].set_ylabel("Per-exposure blue − core (ppm)")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    axes[0, 1].legend(fontsize=7, ncol=2, frameon=False)
    axes[1, 0].set_xlabel("Orbital phase")
    axes[1, 1].set_xlabel("Orbital phase; partial band coverage retained")
    fig.suptitle("Different velocity bands use different exposure weights and phase coverage")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"phase_support.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(11, 10), layout="constrained")
    for row, (group, field) in enumerate(zip(groups, fields)):
        for col, frame in enumerate(["stellar", "planet"]):
            if frame == "stellar":
                coordinate = np.broadcast_to(group["stellar_velocity"], group["weight"].shape)
                grid = np.arange(110, 398, 8.)
            else:
                orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
                coordinate = C_KMS * ((1 + group["stellar_velocity"][None] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)
                grid = np.arange(-64, 72, 8.)
            data = np.full((len(group["phase"]), len(grid) - 1), np.nan)
            for b, (lo, hi) in enumerate(zip(grid[:-1], grid[1:])):
                w = group["weight"] * (coordinate >= lo) * (coordinate < hi)
                data[:, b] = np.divide(np.sum(w * field["centered_observed"], axis=1), w.sum(axis=1), out=np.full(len(w), np.nan), where=w.sum(axis=1) > 0) * 1e6
            display_grid = REST_ANGSTROM * (1 + grid / C_KMS) if frame == "stellar" else grid
            # Separate rectangles avoid painting across the excluded ingress gap.
            for e, phase in enumerate(group["phase"]):
                axes[row, col].pcolormesh(display_grid, [phase - .00075, phase + .00075], data[e:e + 1], vmin=-3000, vmax=3000, cmap="RdBu_r", shading="flat")
            if frame == "stellar":
                track = 244.8 * np.sin(2 * np.pi * group["phase"])
                for center in [-28., 0., 28.]:
                    axes[row, col].plot(REST_ANGSTROM * (1 + (track + center) / C_KMS), group["phase"], color="k", lw=.6, alpha=.5)
                axes[row, col].set_xlim(display_grid[0], display_grid[-1])
            axes[row, col].set_title(group["label"] + ": " + frame + " frame")
            axes[row, col].set_ylabel("Orbital phase")
            axes[row, col].set_xlabel("Archived wavelength (Angstrom)" if frame == "stellar" else "Velocity (km/s)")
    fig.suptitle("Observed time-centered native ratios, no planet or GP subtraction; color ±3000 ppm")
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"observed_frame_maps.{extension}", dpi=180)
    plt.close(fig)
    return exposure_arrays


def sensitivity_summary(items):
    summary = {}
    for group in ["all", "night0", "night1"]:
        summary[group] = {}
        for band in [*WINDOWS, "blue_minus_core"]:
            summary[group][band] = {}
            for field in ["observed", "residual"]:
                available = [item for item in items if group in item["bands_ppm"] and item["bands_ppm"][group][band][field] is not None]
                extrema = {}
                for name, function in [("minimum", min), ("maximum", max)]:
                    item = function(available, key=lambda x: x["bands_ppm"][group][band][field])
                    extrema[name] = dict(value_ppm=item["bands_ppm"][group][band][field],
                                         omitted_night=item["night"], omitted_exposure_ids=item["omitted_exposure_ids"])
                summary[group][band][field] = extrema
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    paths = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    archive_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    surface_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    noise_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    bins_path = Path("/tmp/escape-gp-combined-20261125/combined_arrays.npz")
    archive, surface = dict(np.load(archive_path)), dict(np.load(surface_path))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in np.load(noise_path).items()}
    edges = np.load(bins_path)["edges"]
    groups, original_problem = load_groups(paths, archive, surface)
    primary, profiles, fields = evaluate(groups, noise, edges, details=True)
    comparisons, all_profiles = {"primary": primary}, {"primary": profiles}
    parity = []
    for group, state, field in zip(groups, original_problem["states"], fields):
        k = group["night"] * 2 + group["order"]
        posterior = time_gp_posterior(group["log_data"] - group["model"], state, group["time_min"],
            np.exp(noise["log_gp"][k]), np.exp(noise["log_length"][k]), np.exp(noise["log_white"][k]))
        gp = np.asarray(posterior.mean).T @ np.asarray(state.basis)
        parity.append(float(np.max(abs(gp - field["gp"]))))
    assert max(parity) < 1e-10
    assert abs(sum(primary["group_Q"].values()) - 38326.64894003179) < 1e-6
    print(json.dumps({"primary": primary["bands_ppm"], "GP_parity": parity}), flush=True)
    for label, stride, offset in [("alternate", 2, 1), ("all_pixels", 1, 0), *[(f"stride4_offset{i}", 4, i) for i in range(4)]]:
        selected, _ = load_groups(paths, archive, surface, stride, offset)
        comparisons[label], all_profiles[label], _ = evaluate(selected, noise, edges, details=label in ["alternate", "all_pixels"])
        print(json.dumps({"selection": label, "all": comparisons[label]["bands_ppm"]["all"]}), flush=True)
    for label, phase_range in [("early_before_0p40", (.30, .40)), ("late_after_0p40", (.40, .435)),
                                ("overlapping_nights", (.4032, .435)), ("blue_supported_phase", (.30, .4141))]:
        selected = select(groups, phase_range=phase_range)
        comparisons[label], all_profiles[label], _ = evaluate(selected, noise, edges)
        # Hold every datum, common reference and GP conditional solution fixed;
        # only change which science exposures enter the display coefficients.
        display_groups = [dict(g, science=g["science"] & (g["phase"] >= phase_range[0]) & (g["phase"] < phase_range[1])) for g in groups]
        display_label = label + "_display_only"
        comparisons[display_label], all_profiles[display_label], display_fields = evaluate(display_groups, noise, edges)
        assert all(np.array_equal(a["gp"], b["gp"]) for a, b in zip(fields, display_fields))
        comparisons[display_label]["selection_scope"] = "Only displayed science exposures change. Full-data common reference, conditional GP, and physical native model remain fixed."
        comparisons[label]["selection_scope"] = "Retained science exposures change. Rebuild the common reference and GP conditional mean with unchanged GP hyperparameters and unchanged RV/amplitude."
    for lower in [120., 130.]:
        for suffix, phase_range in [("full", None), ("early", (.30, .40))]:
            label = f"stellar_min_{int(lower)}_{suffix}"
            comparisons[label], all_profiles[label], _ = evaluate(select(groups, phase_range=phase_range, lower_velocity=lower), noise, edges)
    for label, ref in [("eclipse_reference", (.47, .50)), ("early_eclipse_reference", (.47, .48)), ("late_eclipse_reference", (.48, .50))]:
        comparisons[label], all_profiles[label], _ = evaluate([g for g in groups if g["night"] == 1], noise, edges, reference=ref)
    for label, bins in [("bins_shifted_4", edges + 4), ("bins_width_4", np.arange(edges[0], edges[-1] + .01, 4)),
                        ("bins_width_12", np.arange(edges[0], edges[-1] + 12, 12))]:
        comparisons[label], all_profiles[label], _ = evaluate(groups, noise, bins)
    comparisons["exclude_top_1pct_pixel_rms"], all_profiles["exclude_top_1pct_pixel_rms"], _ = evaluate(select(groups, high_rms_mask=True), noise, edges)
    # Extending to the paper's science boundary adds ingress exposures. Inspect
    # observed common-reference values only: the saved uneclipsed template is
    # not an occultation model for those additional exposures.
    extended = load_gp_data(paths, science_max=.45)
    extended_sums, extended_support = [], {}
    for group in extended:
        weight = group["weight"]
        k = group["night"] * 2 + group["order"]
        field = {key: np.zeros_like(weight) for key in ["planet", "gp", "zero_planet_gp", "residual"]}
        field.update(observed=group["log_data"], white_scale=np.exp(noise["log_white"][k]),
                     common=weight / np.maximum(weight.sum(axis=0), 1))
        row, exposure_weight, velocity = project(group, field, edges)
        extended_sums.append(row)
        extra = group["science"] & (group["phase"] >= .435)
        extended_support[group["label"]] = dict(extra_science_exposures=int(extra.sum()),
            extra_native_samples_by_band={name: int(np.sum(extra[:, None] & (weight > 0) & (velocity >= lo) & (velocity < hi))) for name, (lo, hi) in WINDOWS.items()})
    extended_rows = {"all": combine(extended_sums, edges), **{f"night{n}": combine(extended_sums[n * 2:n * 2 + 2], edges) for n in [0, 1]}}
    extended_observed = dict(scope="Observed common-reference values only, science phase <0.45. Added ingress exposures change the reference; no planet prediction, GP solve or Q is claimed for this selection.",
        extra_support=extended_support, bands_ppm={selection: {name: {"observed": entry["observed"]} for name, entry in bands(row).items()} for selection, row in extended_rows.items()})
    leave_one, leave_blocks = [], []
    for night in (0, 1):
        original = next(g for g in groups if g["night"] == night)
        ids = original["exposure_id"][original["science"]]
        for omitted in ids:
            summary, _, _ = evaluate(select(groups, drop=(night, [omitted])), noise, edges)
            leave_one.append(dict(night=night, omitted_exposure_ids=[int(omitted)],
                                  phase=float(original["phase"][original["exposure_id"] == omitted][0]), bands_ppm=summary["bands_ppm"]))
        for block in np.array_split(ids, 5):
            summary, _, _ = evaluate(select(groups, drop=(night, block)), noise, edges)
            leave_blocks.append(dict(night=night, omitted_exposure_ids=block.tolist(), bands_ppm=summary["bands_ppm"]))
    np.savez_compressed(args.raw_output / "profiles.npz", **{f"{label}__{selection}__{key}": array
        for label, selections in all_profiles.items() for selection, row in selections.items() for key, array in row.items()})
    exposure_arrays = plot_results(args.output, all_profiles, groups, fields, edges)
    np.savez_compressed(args.raw_output / "exposure_bands.npz", **exposure_arrays)
    inputs = [Path(__file__), *[Path(__file__).with_name(name) for name in ["gp_halpha.py", "phase_halpha.py", "cloudy_halpha.py"]],
              *[Path(__file__).resolve().parents[1] / "src/escape" / name for name in ["time_gp.py", "observation.py", "stellar_radiation.py"]],
              *paths, archive_path, surface_path, noise_path, bins_path]
    details = dict(bin_support={}, leave_one_science_exposure=leave_one, leave_fifth_science_blocks=leave_blocks)
    for label, case in comparisons.items():
        if "bin_support" not in case:
            continue
        details["bin_support"][label] = case.pop("bin_support")
        case["bin_support_summary"] = {}
        for group, entries in details["bin_support"][label].items():
            selected = []
            for entry in entries:
                if not -44 <= entry["velocity_kms"] <= 44:
                    continue
                selected.append({key: value for key, value in entry.items() if key not in ["exposure_ids", "phases", "exposure_weight_fraction"]})
                selected[-1]["phase_min"] = min(entry["phases"]) if entry["phases"] else None
                selected[-1]["phase_max"] = max(entry["phases"]) if entry["phases"] else None
            case["bin_support_summary"][group] = selected
    details_path = args.raw_output / "selection_details.json"
    details_path.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    night_weights = {}
    for name, (lo, hi) in WINDOWS.items():
        row = all_profiles["primary"]["all"]
        keep = (row["velocity"] >= lo) & (row["velocity"] < hi) & (row["weight"] > 0)
        w = 1 / row["white_error"][keep]**2
        night_weights[name] = {night: float(np.average(all_profiles["primary"][night]["weight"][keep] / row["weight"][keep], weights=w)) for night in ["night0", "night1"]}
    report = dict(scope="Empirical shape consistency using archived native data, fixed baseline RV/amplitude and fixed GP hyperparameters. Changed samples and display bins are sensitivity analyses, not comparable absolute likelihoods.",
        limitations=["White display errors are not GP-correlated significances.", "The observed profile subtracts no physical template or GP, but its display weights retain baseline-conditioned fitted white scales. Common-reference subtraction can self-subtract a planet signal.",
                     "Eclipse-only reference is available on August 20 only; August 8 has no eclipse exposures.",
                     "Band averages are supported by different phases and wavelengths; apparent shape can combine phase evolution with spectral structure.",
                     "The top-1-percent pixel-RMS exclusion is exploratory and data-dependent; it is not a replacement data reduction."],
        fixed_parameters=dict(rv_kms=float(PARAMETERS[0]), amplitude=float(np.exp(PARAMETERS[1]))), windows_kms=WINDOWS,
        selected_native_support={group["label"]: dict(exposures=len(group["phase"]), pixels=len(group["stellar_velocity"]),
            valid_measurements=int((group["weight"] > 0).sum()), invalid_or_unsupported_measurements=int((group["weight"] == 0).sum()),
            science_exposures=int(group["science"].sum()), eclipse_exposures=int(group["eclipse"].sum()),
            phase_min=float(group["phase"].min()), phase_max=float(group["phase"].max())) for group in groups},
        GP_numpy_parity_max_abs=parity, primary_Q=sum(primary["group_Q"].values()), noise_medians={k:v.tolist() for k,v in noise.items()},
        cases=comparisons, primary_band_night_weight_fraction=night_weights,
        original_science_boundary_observed_only=extended_observed,
        leave_one_science_exposure=dict(number_of_trials=len(leave_one), extrema=sensitivity_summary(leave_one)),
        leave_fifth_science_blocks=dict(number_of_trials=len(leave_blocks), extrema=sensitivity_summary(leave_blocks)),
        inputs={str(p): digest(p) for p in inputs}, raw_artifacts={str(p): digest(p) for p in [args.raw_output / "profiles.npz", args.raw_output / "exposure_bands.npz", details_path]},
        figures={str(p): digest(p) for p in args.output.glob("*.p*")})
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output / "review.json", flush=True)


if __name__ == "__main__":
    main()
