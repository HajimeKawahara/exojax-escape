"""Localize archived H-alpha residuals with predeclared phase/support rules.

The original full-data reference, baseline RV/amplitude, and conditional GP are
fixed. Complete-band averages require observed native support; missing late
blue bands remain unsupported. Maps are linear in native log-flux arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from halpha_data_consistency import load_groups, solve_group, PARAMETERS, WINDOWS
from escape.observation import C_KMS

PHASE_EDGES = {"fine": [.30, .325, .35, .375, .40, .425, .435],
               "coarse": [.30, .35, .40, .435]}
PRODUCTS = ["common_complete", "band_complete", "legacy_display"]
SCOPES = ["all", "night0", "night1", "night0_order0", "night0_order1", "night1_order0", "night1_order1"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_labels():
    rows = []
    for grid, edges in PHASE_EDGES.items():
        for product in PRODUCTS:
            for scope in SCOPES if grid == "fine" else ["all"]:
                for phase_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                    for band in WINDOWS:
                        rows.append(dict(row_id=len(rows), grid=grid, product=product, scope=scope,
                            phase_index=phase_index, phase_low=left, phase_high=right, band=band))
    return rows


def in_scope(group, scope):
    return scope == "all" or scope == f"night{group['night']}" or scope == group["label"]


def support_and_quadrature(groups):
    """Native 4-km/s subbin means and conservative complete-band criteria.

    Each band edge must lie between retained valid native pixel centers, and
    every four-km/s subbin must contain a positive-weight measurement. This is
    deliberately stricter than assuming a pixel response beyond a mask edge.
    Complete support is required in both simultaneously observed orders.
    """
    support, quadrature, coordinates = [], [], []
    for group in groups:
        orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
        velocity = C_KMS * ((1 + group["stellar_velocity"][None] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)
        weight = group["weight"]
        valid = weight > 0
        lower = np.min(np.where(valid, velocity, np.inf), axis=1)
        upper = np.max(np.where(valid, velocity, -np.inf), axis=1)
        bands, means = {}, {}
        for band, (left, right) in WINDOWS.items():
            edges = np.arange(left, right + .1, 4.)
            raw = np.array([weight * (velocity >= lo) * (velocity < hi) for lo, hi in zip(edges[:-1], edges[1:])])
            normalization = raw.sum(axis=2)
            mean = np.mean(np.divide(raw, normalization[:, :, None], out=np.zeros_like(raw), where=normalization[:, :, None] > 0), axis=0)
            complete = (lower <= left) & (upper >= right) & np.all(normalization > 0, axis=0) & group["science"]
            bands[band] = dict(complete_native=complete, subbin_measurement_counts=np.sum(raw > 0, axis=2),
                               subbin_edges=edges, any_native=np.any(normalization > 0, axis=0) & group["science"])
            means[band] = mean
        support.append(bands)
        quadrature.append(means)
        coordinates.append(velocity)
    for night in [0, 1]:
        indices = [i for i, group in enumerate(groups) if group["night"] == night]
        reference_phase = groups[indices[0]]["phase"]
        for i in indices:
            np.testing.assert_array_equal(groups[i]["phase"], reference_phase)
        for band in WINDOWS:
            complete = np.logical_and.reduce([support[i][band]["complete_native"] for i in indices])
            for i in indices:
                support[i][band]["complete_both_orders"] = complete
        common = np.logical_and.reduce([support[indices[0]][band]["complete_both_orders"] for band in WINDOWS])
        for i in indices:
            support[i]["common_complete"] = common
    return support, quadrature, coordinates


def project_reference(group, coefficient):
    weight = group["weight"]
    reference = weight / np.where(weight.sum(axis=0) > 0, weight.sum(axis=0), 1)
    return coefficient - reference * coefficient.sum(axis=0)[None]


def legacy_coefficients(groups, fields, coordinates, row, display_edges):
    """Exactly the previous display, restricted only in displayed phase/scope."""
    centers = (display_edges[:-1] + display_edges[1:]) / 2
    left, right = WINDOWS[row["band"]]
    bins = np.flatnonzero((centers >= left) & (centers < right))
    raw, total = [], np.zeros(len(bins))
    for group, velocity in zip(groups, coordinates):
        phase = group["phase"]
        selected = group["science"] & (phase >= row["phase_low"]) & (phase < row["phase_high"]) & in_scope(group, row["scope"])
        value = np.array([group["weight"] * selected[:, None] * (velocity >= display_edges[b]) * (velocity < display_edges[b + 1]) for b in bins])
        raw.append(value)
        total += value.sum(axis=(1, 2))
    normalized = [np.divide(value, total[:, None, None], out=np.zeros_like(value), where=total[:, None, None] > 0) for value in raw]
    variance = np.zeros(len(bins))
    for group, field, values in zip(groups, fields, normalized):
        for b, value in enumerate(values):
            projected = project_reference(group, value)
            variance[b] += field["white_scale"]**2 * np.sum(np.divide(projected**2, group["weight"], out=np.zeros_like(projected), where=group["weight"] > 0))
    white_weight = np.divide(1., variance, out=np.zeros_like(variance), where=(variance > 0) & (total > 0))
    if white_weight.sum():
        white_weight /= white_weight.sum()
    return [np.einsum("b,bep->ep", white_weight, value) for value in normalized]


def construct_maps(groups, fields, labels, display_edges):
    support, quadrature, coordinates = support_and_quadrature(groups)
    maps = [np.zeros((len(labels), *group["weight"].shape)) for group in groups]
    raw_maps = [np.zeros_like(value) for value in maps]
    metadata = []
    for row in labels:
        band = row["band"]
        if row["product"] == "legacy_display":
            coefficients = legacy_coefficients(groups, fields, coordinates, row, display_edges)
        else:
            chosen = []
            for group, available in zip(groups, support):
                complete = available["common_complete"] if row["product"] == "common_complete" else available[band]["complete_both_orders"]
                chosen.append(complete & (group["phase"] >= row["phase_low"]) & (group["phase"] < row["phase_high"]) & in_scope(group, row["scope"]))
            count = sum(int(keep.sum()) for keep in chosen)
            coefficients = [means[band] * keep[:, None] / max(count, 1) for means, keep in zip(quadrature, chosen)]
        group_counts, group_ids, group_phase, fractions, native_counts, subbin_support = [], [], [], [], [], []
        phases, exposure_weights = [], []
        for i, (group, available, coefficient) in enumerate(zip(groups, support, coefficients)):
            r = row["row_id"]
            maps[i][r] = project_reference(group, coefficient)
            raw_maps[i][r] = coefficient
            ew = coefficient.sum(axis=1)
            keep = ew > 0
            group_counts.append(int(keep.sum()))
            group_ids.append(group["exposure_id"][keep].tolist())
            group_phase.append(group["phase"][keep].tolist())
            fractions.append(float(coefficient.sum()))
            native_counts.append(int((coefficient > 0).sum()))
            subbin_support.append(available[band]["subbin_measurement_counts"][:, keep].sum(axis=1).tolist())
            phases.extend(group["phase"][keep])
            exposure_weights.extend(ew[keep])
        supported = sum(group_counts) > 0
        total = sum(fractions)
        assert abs(total - float(supported)) < 1e-12
        metadata.append(dict(supported=supported, complete_fixed_band=row["product"] != "legacy_display",
            group_exposure_counts=group_counts, group_exposure_ids=group_ids, group_phases=group_phase,
            group_coefficient_fraction=fractions, group_native_measurement_counts=native_counts,
            group_subbin_measurement_counts=subbin_support,
            phase_min=float(min(phases)) if supported else None, phase_max=float(max(phases)) if supported else None,
            phase_weighted_mean=float(np.average(phases, weights=exposure_weights)) if supported else None,
            group_exposure_weighted_effective_count=float(1 / np.sum(np.square(exposure_weights))) if supported else 0.))
    return maps, raw_maps, metadata, support


def values_from_maps(groups, fields, maps, metadata):
    result = []
    values = {name: sum(np.einsum("rep,ep->r", mapping, field[name]) for mapping, field in zip(maps, fields)) * 1e6
              for name in ["observed", "planet", "gp", "residual"]}
    variance = sum(field["white_scale"]**2 * np.einsum("rep,rep,ep->r", mapping, mapping,
                   np.divide(1., group["weight"], out=np.zeros_like(group["weight"]), where=group["weight"] > 0))
                   for mapping, field, group in zip(maps, fields, groups)) * 1e12
    for r, meta in enumerate(metadata):
        result.append({**{name: float(value[r]) if meta["supported"] else None for name, value in values.items()},
                       "white_error_diagnostic": float(np.sqrt(variance[r])) if meta["supported"] else None})
    return result


def plot(output, labels, selections):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    colors = {"blue": "#0072B2", "core": ".3", "red": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, layout="constrained")
    for column, product in enumerate(PRODUCTS):
        for selection, style in [("primary", "-"), ("alternate", "--")]:
            for band, color in colors.items():
                ids = [r["row_id"] for r in labels if r["grid"] == "fine" and r["scope"] == "all" and r["product"] == product and r["band"] == band]
                x = [(labels[i]["phase_low"] + labels[i]["phase_high"]) / 2 for i in ids]
                for row, field in enumerate(["observed", "residual"]):
                    y = [selections[selection]["values_ppm"][i][field] for i in ids]
                    axes[row, column].plot(x, [np.nan if v is None else v for v in y], style, color=color, marker="o" if selection == "primary" else "x", label=f"{band}: {selection}")
                    axes[row, column].axhline(0, color=".6", lw=.6)
        axes[0, column].set_title({"common_complete": "Same exposures, complete B/C/R", "band_complete": "Complete band, band-specific exposures", "legacy_display": "Original display: partial support allowed"}[product], fontsize=9)
        axes[1, column].set_xlabel("Declared phase interval midpoint")
    axes[0, 0].set_ylabel("Observed common-reference contrast (ppm)")
    axes[1, 0].set_ylabel("Data − fixed baseline − conditional GP (ppm)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Phase localization: missing complete bands are not filled or extrapolated", fontsize=11)
    for ext in ["pdf", "png"]:
        fig.savefig(output / f"phase_residuals.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout="constrained")
    for row, selection in enumerate(["primary", "alternate"]):
        for night in [0, 1]:
            for product, style in [("common_complete", "--"), ("band_complete", "-")]:
                for band, color in colors.items():
                    ids = [r["row_id"] for r in labels if r["grid"] == "fine" and r["scope"] == f"night{night}" and r["product"] == product and r["band"] == band]
                    x = [(labels[i]["phase_low"] + labels[i]["phase_high"]) / 2 for i in ids]
                    # Both orders have the same eligible physical exposures.
                    count = [selections[selection]["rows"][i]["group_exposure_counts"][night * 2] for i in ids]
                    axes[row, night].plot(x, count, style, marker="o", color=color, label=f"{band}, {product.replace('_', ' ')}")
            axes[row, night].set_title(f"{selection}: August {'8' if night == 0 else '20'}")
            axes[row, night].set_ylabel("Eligible physical exposures")
            axes[row, night].set_ylim(bottom=-.5)
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)
    axes[1, 0].set_xlabel("Declared phase interval midpoint")
    axes[1, 1].set_xlabel("Declared phase interval midpoint")
    fig.suptitle("Complete native support: conservative band edges and both-order requirement")
    for ext in ["pdf", "png"]:
        fig.savefig(output / f"phase_support.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    archive_path = Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz")
    surface_path = Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz")
    noise_path = Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz")
    display_path = Path("/tmp/escape-gp-combined-20261125/combined_arrays.npz")
    paths = [Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")]
    archive, surface = dict(np.load(archive_path)), dict(np.load(surface_path))
    noise = {key: np.median(value, axis=(0, 1)) for key, value in np.load(noise_path).items()}
    display_edges = np.load(display_path)["edges"]
    labels, selections, arrays = row_labels(), {}, {}
    for selection, offset in [("primary", 0), ("alternate", 1)]:
        groups, _ = load_groups(paths, archive, surface, offset=offset)
        fields = [solve_group(group, noise) for group in groups]
        maps, raw_maps, metadata, support = construct_maps(groups, fields, labels, display_edges)
        values = values_from_maps(groups, fields, maps, metadata)
        common_leak = max(float(np.max(abs(mapping.sum(axis=1)))) for mapping in maps)
        assert common_leak < 1e-12
        for r, meta in enumerate(metadata):
            if not meta["supported"]:
                assert all(np.count_nonzero(mapping[r]) == 0 for mapping in maps)
        q = sum(field["Q"] for field in fields)
        if selection == "primary":
            assert abs(q - 38326.64894003179) < 1e-6
        selections[selection] = dict(rows=metadata, values_ppm=values, full_data_Q_at_fixed_parameters=q,
                                     maximum_common_mode_leakage=common_leak,
            support_by_group={group["label"]: {band: dict(complete_native_exposure_ids=group["exposure_id"][available[band]["complete_native"]].tolist(),
                         complete_both_orders_exposure_ids=group["exposure_id"][available[band]["complete_both_orders"]].tolist(),
                         any_native_exposure_ids=group["exposure_id"][available[band]["any_native"]].tolist()) for band in WINDOWS}
                         for group, available in zip(groups, support)})
        for i, (group, mapping, raw, field) in enumerate(zip(groups, maps, raw_maps, fields)):
            prefix = f"{selection}__g{i}__"
            arrays[prefix + "maps"] = mapping
            arrays[prefix + "raw_maps"] = raw
            for key in ["phase", "stellar_velocity", "pixel_index", "weight", "science", "eclipse", "exposure_id", "time_min", "log_data", "model"]:
                arrays[prefix + key] = group[key]
            arrays[prefix + "gp"] = field["gp"]
        main_ids = [r["row_id"] for r in labels if r["grid"] == "fine" and r["product"] == "common_complete" and r["scope"] == "all"]
        print(json.dumps(dict(selection=selection, primary_rows=[dict(**labels[i], **metadata[i], **values[i]) for i in main_ids])), flush=True)
    np.savez_compressed(args.raw_output / "maps.npz", **arrays)
    repository = Path(__file__).resolve().parents[1]
    inputs = [Path(__file__), *[repository / "examples" / name for name in ["halpha_data_consistency.py", "gp_halpha.py", "phase_halpha.py", "cloudy_halpha.py"]],
              *[repository / "src/escape" / name for name in ["time_gp.py", "observation.py", "stellar_radiation.py"]],
              archive_path, surface_path, noise_path, display_path, *paths]
    report = dict(schema_version=1, rows=labels, selections=selections, parameters=PARAMETERS.tolist(),
        phase_edges=PHASE_EDGES, windows_kms=WINDOWS, subbin_width_kms=4.,
        scope="Linear phase/band summaries of the original full-data native contrasts at fixed baseline RV/amplitude, reference and conditional GP. No atmosphere solve, GP retraining, or missing-band extrapolation.",
        map_convention="maps apply the unchanged weighted common-reference projection. They do not subtract GP or profile nuisance parameters. raw_maps are pre-reference positive normalized coefficients. Values are ppm; maps act on dimensionless native log flux.",
        weighting="Within each 4-km/s subbin: archive inverse-variance weights. Subbins: equal weights. Complete-band exposures: equal weights; simultaneous orders: equal weights. Both orders must meet support. Legacy display retains its original bin-error weighting.",
        completeness="Every subbin has positive-weight measurements and both band endpoints lie within the range of valid native pixel centers in both orders. common_complete also requires this for all three bands on each selected exposure.",
        limitations=["Observed summaries subtract no physical template, but archive weights and white display scales are not independent noise calibration.",
                     "Different phase rows share a full-data reference and are correlated; white errors are descriptive only.",
                     "band_complete rows can use different exposures in different bands. Only common_complete compares all bands on the same exposures.",
                     "A supported legacy display value may cover only a fraction of the named band; it is not a complete-band measurement.",
                     "Native subbin averages are a discrete quadrature, not an assumed continuous spectrum beyond the observed pixels."],
        inputs={str(path): sha256(path) for path in inputs}, raw_maps=dict(path=str(args.raw_output / "maps.npz"), sha256=sha256(args.raw_output / "maps.npz")))
    rows_path = args.raw_output / "rows.json"
    rows_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # The shared map interface is sealed before optional figure generation.
    print(json.dumps(dict(maps=str(args.raw_output / "maps.npz"), rows=str(rows_path), source_sha256=sha256(__file__))), flush=True)
    plot(args.output, labels, selections)
    report["figures"] = {str(path): sha256(path) for path in args.output.glob("*.p*")}
    report["raw_rows"] = dict(path=str(rows_path), sha256=sha256(rows_path))
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
