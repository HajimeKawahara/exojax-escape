"""Combine the existing GP posterior across all science phases and both nights.

Use predictions.npz produced by gp_halpha.py; no new fit or background
reconstruction is performed. Each night/order retains its own common-spectrum
projection, including the eclipse exposures used by the original likelihood.
All supported science pixels are combined in fixed planet-frame velocity bins.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from escape.observation import C_KMS
from gp_halpha import normal_mixture_quantiles


def planet_velocity(predictions, group):
    """Use the same fixed orbital coordinates as the original spectral display."""
    prefix = f"group{group}_"
    orbital = 244.8 * np.sin(2 * np.pi * predictions[prefix + "phase"])
    stellar = predictions[prefix + "stellar_velocity"]
    return C_KMS * ((1 + stellar[None] / C_KMS) / (1 + orbital[:, None] / C_KMS) - 1)


def combined_spectrum(predictions, edges):
    """Sum per-draw projected means and variances before taking quantiles.

    For science-bin weights A, the projected coefficients are
    P_ep = A_ep - w_ep sum_f(A_fp) / sum_f(w_fp).
    The denominator retains every likelihood exposure, not just science rows.
    GP groups are conditionally independent, but shared planet/hyperparameter
    draws must remain aligned across groups when forming the mixture.
    """
    bins = len(edges) - 1
    draws = len(predictions["group0_planet"])
    weight_sum = np.zeros(bins)
    observed = np.zeros(bins)
    planet = np.zeros((draws, bins))
    total = np.zeros_like(planet)
    gp_variance = np.zeros_like(planet)
    white_variance = np.zeros_like(planet)
    night_weight = np.zeros((2, bins))
    presence = {}
    for group in range(4):
        prefix = f"group{group}_"
        weight = predictions[prefix + "weight"]
        science = predictions[prefix + "science"]
        index = (
            np.searchsorted(edges, planet_velocity(predictions, group), side="right")
            - 1
        )
        coefficient = np.array(
            [weight * science[:, None] * (index == b) for b in range(bins)]
        )
        local_weight = coefficient.sum(axis=(1, 2))
        weight_sum += local_weight
        night_weight[group // 2] += local_weight
        included = np.any(coefficient > 0, axis=2)
        if group // 2 in presence:
            presence[group // 2] |= included
        else:
            presence[group // 2] = included
        common = weight / np.where(weight.sum(axis=0) > 0, weight.sum(axis=0), 1)
        projected = coefficient - common[None] * coefficient.sum(axis=1)[:, None]
        observed += np.einsum("bep,ep->b", projected, predictions[prefix + "log_data"])
        local_planet = np.einsum(
            "bep,dep->db", projected, predictions[prefix + "planet"]
        )
        planet += local_planet
        stellar = predictions[prefix + "stellar_velocity"]
        basis = np.array([np.ones_like(stellar), (stellar - 250) / 140])
        design = np.einsum("bep,mp->bme", projected, basis).reshape(bins, -1)
        total += (
            local_planet + predictions[prefix + "gp_mean"].reshape(draws, -1) @ design.T
        )
        gp_variance += np.einsum(
            "bi,dij,bj->db",
            design,
            predictions[prefix + "gp_covariance"],
            design,
            optimize=True,
        )
        variance = (projected**2 / np.where(weight > 0, weight, np.inf)).sum(
            axis=(1, 2)
        )
        white_variance += (
            predictions["white_scale"][:, group, None] ** 2 * variance[None]
        )
    keep = weight_sum > 0
    normalization = np.where(keep, weight_sum, 1)
    planet /= normalization
    total /= normalization
    gp_variance /= normalization**2
    white_variance /= normalization**2
    return dict(
        velocity=((edges[:-1] + edges[1:]) / 2)[keep],
        observed=(observed / normalization)[keep],
        error=np.sqrt(white_variance.mean(axis=0))[keep],
        planet=np.quantile(planet, [0.05, 0.5, 0.95], axis=0)[:, keep],
        total=normal_mixture_quantiles(total, gp_variance)[:, keep],
        weight=weight_sum[keep],
        night_weight=night_weight[:, keep],
        night_exposures=np.array([presence[n].sum(axis=1) for n in range(2)])[:, keep],
    )


def plot(arrays, output):
    """Show the complete supported interval, with changing exposure coverage."""
    import matplotlib.pyplot as plt

    fig, (ax, coverage) = plt.subplots(
        2,
        1,
        figsize=(11, 6.7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1]},
        layout="constrained",
    )
    velocity = arrays["velocity"]
    ax.errorbar(
        velocity,
        arrays["observed"] * 1e6,
        arrays["error"] * 1e6,
        fmt=".",
        ms=5,
        color="0.4",
        alpha=0.8,
        label="Combined data (white 1σ)",
    )
    band = arrays["total"] * 1e6
    ax.fill_between(velocity, band[0], band[2], color="C0", alpha=0.22)
    ax.plot(velocity, band[1], color="C0", label="Planet + GP (pointwise 90% interval)")
    ax.plot(
        velocity, arrays["planet"][1] * 1e6, color="C1", label="Planet contribution"
    )
    if "white_planet" in arrays:
        ax.plot(
            velocity,
            arrays["white_planet"][1] * 1e6,
            "--",
            color="C3",
            label="White-only fit",
        )
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set(
        ylabel="Projected log-ratio (ppm)",
        title="All science phases + both nights: same NUTS posterior",
    )
    ax.legend(fontsize=9, loc="upper right")
    for night, label, color in [
        (0, "August 8 (83 exposures)", "C2"),
        (1, "August 20 (21 exposures)", "C4"),
    ]:
        coverage.step(
            velocity,
            arrays["night_exposures"][night],
            where="mid",
            label=label,
            color=color,
        )
    coverage.set(
        ylabel="Contributing\nexposures",
        xlabel="Planet-frame velocity (km/s)",
        ylim=(0, 88),
    )
    coverage.legend(fontsize=9, loc="upper right")
    ax.set_xlim(arrays["edges"][0], arrays["edges"][-1])
    fig.supxlabel(
        "Phase 0.30–0.435; original even pixels; inverse-variance weights. "
        "Each night/order keeps its own common-reference removal.\n"
        "Bands describe the fitted latent signal, not new noisy observations. "
        "This is not an absolute planet emission spectrum.",
        fontsize=9,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(args):
    predictions = dict(np.load(args.predictions))
    velocity = np.concatenate(
        [
            planet_velocity(predictions, group)[
                predictions[f"group{group}_science"][:, None]
                & (predictions[f"group{group}_weight"] > 0)
            ]
            for group in range(4)
        ]
    )
    if not np.isfinite(args.bin_width) or args.bin_width <= 0:
        raise ValueError("Use a positive finite velocity-bin width.")
    # Anchor at the coverage edge, avoiding an almost empty partial first bin.
    start = np.nextafter(velocity.min(), -np.inf)
    count = int(np.floor((velocity.max() - start) / args.bin_width)) + 1
    edges = start + args.bin_width * np.arange(count + 1)
    arrays = combined_spectrum(predictions, edges)
    arrays["edges"] = edges
    if args.white is not None:
        white = dict(np.load(args.white))
        for group in range(4):
            for key in [
                "phase",
                "science",
                "stellar_velocity",
                "pixel_index",
                "weight",
                "log_data",
            ]:
                np.testing.assert_array_equal(
                    predictions[f"group{group}_{key}"], white[f"group{group}_{key}"]
                )
        comparison = combined_spectrum(white, edges)
        np.testing.assert_array_equal(arrays["velocity"], comparison["velocity"])
        arrays["white_planet"] = comparison["planet"]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "combined_arrays.npz", **arrays)
    plot(arrays, args.output / "combined_gp_spectrum.png")
    paths = [args.predictions, Path(__file__)] + (
        [] if args.white is None else [args.white]
    )
    review = dict(
        scope="Display-only combination of the same fixed-atmosphere conditional posterior; no new NUTS run.",
        science_exposures=[
            int(predictions[f"group{2*n}_science"].sum()) for n in range(2)
        ],
        posterior_draws=len(predictions["group0_planet"]),
        native_science_velocity_range_kms=[
            float(velocity.min()),
            float(velocity.max()),
        ],
        bin_width_kms=args.bin_width,
        bins=len(arrays["velocity"]),
        sources={str(p): sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=[
            "The full supported fitting window is shown, not the full original HDF5 orders.",
            "Common-reference removal can create negative shoulders; no absolute planet spectrum is recovered.",
            "Phase/night contributions vary with wavelength; this is not an unweighted phase average.",
            "Shared posterior draws are combined before quantiles; conditional GP covariance is retained.",
            "Only fitted white-noise marginal errors are displayed; full spectral/order covariance remains unresolved.",
        ],
    )
    (args.output / "review.json").write_text(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--white", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=8.0)
    main(parser.parse_args())
