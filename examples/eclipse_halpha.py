"""Test an eclipse reference without assuming it is a clean calibration.

The input is one archived night (phase, wave, flux, err); flux already stores
F / nightly_master - 1. Re-referencing is an explicitly declared transformation
whose original common master cancels. The original extraction/master covariance
is unavailable: all reported errors condition on independent supplied errors.
No reference is selected by fit quality, and no correction is adopted here.

Example::

    python examples/eclipse_halpha.py --data night2.h5 --output results/eclipse
    python examples/eclipse_halpha.py --plot-only --output results/eclipse
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_KMS = 299792.458
REST_ANGSTROM = 1e8 / 15233.146


def remaster(wave, phase, residual, error, science, reference, wavelength):
    """Return a coadd and its shared-reference covariance, including weights.

    Science and reference exposures must be disjoint. If m_p is the new
    reference at native pixel p, the coadd weight is m_p**2 / sigma_ep**2.
    Consequently dR_j/dm_p = [AF_jp - 2 A_jp m_p (1+R_j)] / D_j, where
    A=sum(1/sigma**2), AF=sum((1+r)/sigma**2), and D=sum(m**2/sigma**2).
    This includes the reference dependence of both flux and coadd weights.
    """
    if np.any(science & reference) or not np.any(reference):
        raise ValueError("Science and nonempty reference sets must be disjoint.")
    good = np.isfinite(residual) & (residual > -1) & np.isfinite(error) & (error > 0)
    ivar = np.divide(1, error**2, out=np.zeros_like(error), where=good)
    flux = np.where(good, 1 + residual, 0)
    sums = ivar[reference].sum(axis=0)
    m = np.divide((ivar[reference] * flux[reference]).sum(axis=0), sums,
                  out=np.zeros_like(sums), where=sums > 0)
    vm = np.divide(1, sums, out=np.zeros_like(sums), where=sums > 0)
    edges = np.r_[wavelength[0] - np.diff(wavelength)[0] / 2,
                  (wavelength[:-1] + wavelength[1:]) / 2,
                  wavelength[-1] + np.diff(wavelength)[-1] / 2]
    count = len(wavelength)
    a = np.zeros((count, *wave.shape))
    af = np.zeros_like(a)
    original = np.zeros(count)
    original_weight = np.zeros(count)
    for e in np.flatnonzero(science):
        shift = 244.8 * np.sin(2 * np.pi * phase[e])
        for order in range(len(wave)):
            vs = C_KMS * (wave[order] / REST_ANGSTROM - 1)
            index = np.searchsorted(edges, wave[order] / (1 + shift / C_KMS)) - 1
            keep = ((index >= 0) & (index < count) & (abs(vs) >= 110)
                    & good[e, order] & (m[order] > 0))
            p = np.flatnonzero(keep)
            j = index[p]
            np.add.at(a[:, order], (j, p), ivar[e, order, p])
            np.add.at(af[:, order], (j, p), ivar[e, order, p] * flux[e, order, p])
            original += np.bincount(j, weights=ivar[e, order, p] * residual[e, order, p],
                                   minlength=count)
            original_weight += np.bincount(j, weights=ivar[e, order, p], minlength=count)
    denominator = (a * m[None]**2).sum(axis=(1, 2))
    valid = denominator > 0
    safe = np.where(valid, denominator, 1)
    result = (af * m[None]).sum(axis=(1, 2)) / safe - 1
    derivative = (af - 2 * a * m[None] * (1 + result[:, None, None])) / safe[:, None, None]
    flat = derivative.reshape(count, -1)
    covariance = (flat * vm.ravel()) @ flat.T
    result[~valid] = np.nan
    original = np.divide(original, original_weight, out=np.full(count, np.nan),
                         where=original_weight > 0)
    variance = np.where(valid, 1 / safe, np.nan)
    return dict(value=result, original=original, weight=denominator,
                science_variance=variance, reference_covariance=covariance,
                derivative=derivative, reference=m, a=a, af=af)


def time_control(phase, residual, error, wavelength, low, high):
    """Profile native-pixel intercepts; test one common linear temporal mode.

    Work with log(1+r), so an arbitrary common wavelength-dependent master
    is an additive pixel intercept. Subtracting its weighted time mean removes
    that intercept. Covariance below retains the shared fitted-mean term.
    The error transformation is first order in the supplied residual errors.
    """
    velocity = C_KMS * (wavelength / REST_ANGSTROM - 1)
    take = (velocity >= low) & (velocity < high)
    r, s = residual[:, take], error[:, take]
    good = np.isfinite(r) & (r > -1) & np.isfinite(s) & (s > 0)
    y = np.log(np.where(good, 1 + r, 1))
    sigma = np.divide(s, 1 + r, out=np.ones_like(s), where=good)
    w = np.where(good, 1 / sigma**2, 0)
    total = w.sum(axis=0)
    safe = np.where(total > 0, total, 1)
    mu = (w * y).sum(axis=0) / safe
    x = phase - .48
    xm = (w * x[:, None]).sum(axis=0) / safe
    dx, dy = x[:, None] - xm, y - mu
    den = np.sum(w * dx**2)
    slope = np.sum(w * dx * dy) / den
    coefficients = w / w.sum(axis=1)[:, None]
    centered = (coefficients * dy).sum(axis=1)
    mode = (coefficients * dx).sum(axis=1)
    covariance = np.diag((coefficients**2 * sigma**2).sum(axis=1))
    covariance -= (coefficients / safe[None]) @ coefficients.T
    # Three disjoint, ordered time blocks. Intercepts and slope are trained
    # only on the other blocks, then scored on the held-out exposures.
    block_results = []
    for held in np.array_split(np.arange(len(phase)), 3):
        train = np.ones(len(phase), bool)
        train[held] = False
        wt = w[train]
        st = wt.sum(axis=0)
        st = np.where(st > 0, st, 1)
        ym = (wt * y[train]).sum(axis=0) / st
        xt = (wt * x[train, None]).sum(axis=0) / st
        dxt = x[train, None] - xt
        b = np.sum(wt * dxt * (y[train] - ym)) / np.sum(wt * dxt**2)
        null = y[held] - ym
        changed = null - b * (x[held, None] - xt)
        block_results.append(dict(first_phase=float(phase[held[0]]),
            last_phase=float(phase[held[-1]]), training_slope_per_phase=float(b),
            heldout_weighted_sse_constant=float(np.sum(w[held] * null**2)),
            heldout_weighted_sse_linear=float(np.sum(w[held] * changed**2))))
    return dict(centered=centered, covariance=covariance, mode=mode,
                slope=slope, slope_error=den**-.5,
                sse_constant=float(np.sum(w * dy**2)),
                sse_linear=float(np.sum(w * (dy - slope * dx)**2)),
                samples=int(good.sum()), temporal_holdouts=block_results,
                # Sufficient statistics permit genuine order holdout below.
                trend_denominator=den)


def run(data: Path, output: Path) -> None:
    """Evaluate prespecified reference windows and stellar-frame controls."""
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(data) as h:
        phase, wave, residual, error = [h[k][:] for k in ("phase", "wave", "flux", "err")]
        metadata = dict(root_attributes=list(h.attrs),
                        dataset_attributes={key: list(h[key].attrs) for key in h})
    if len(wave) != 2 or residual.shape != (len(phase), *wave.shape):
        raise ValueError("This example expects two orders on a fixed nightly grid.")
    if np.any(np.diff(phase) <= 0) or not (.4 < phase.min() < .41 and .49 < phase.max() < .5):
        raise ValueError("Use the archived 2024-08-20 sequence in chronological order.")
    wavelength = np.loadtxt(Path(__file__).resolve().parents[1] / "data" /
        "zenodo_20707255" / "KELT9b_w6565_avg_p0.30-0.45.dat")[:, 0]
    velocity = C_KMS * (wavelength / REST_ANGSTROM - 1)
    separation = 3.15 * np.sqrt(np.sin(2 * np.pi * phase)**2
        + np.cos(np.deg2rad(86.79))**2 * np.cos(2 * np.pi * phase)**2)
    science = (phase >= .40320277222508594) & (phase <= .4351290585057086)
    windows = {f"late_{cut:.3f}": phase >= cut for cut in (.465, .47, .475, .48)}
    windows["early_disjoint"] = (phase >= .47) & (phase < .48)
    arrays = dict(velocity_kms=velocity, phase_all=phase,
                  eclipse_margin_rstar=1 - separation - .24)
    references = []
    for name, reference in windows.items():
        if not np.all(separation[reference] + .24 < 1):
            raise ValueError("Reference is not fully occulted in the declared geometry.")
        result = remaster(wave, phase, residual, error, science, reference, wavelength)
        arrays[f"coadd_{name}"] = result["value"]
        arrays[f"error_{name}"] = np.sqrt(result["science_variance"]
                                          + np.diag(result["reference_covariance"]))
        arrays[f"reference_covariance_{name}"] = result["reference_covariance"]
        arrays["archived"] = result["original"]
        record = dict(name=name, count=int(reference.sum()),
            minimum_occultation_margin_rstar=float(np.min(1 - separation[reference] - .24)),
            regions={})
        for label, low, high in (("red_peak", 15, 45), ("red_wing", 50, 92)):
            take = (velocity >= low) & (velocity <= high) & (result["weight"] > 0)
            w = result["weight"][take] / result["weight"][take].sum()
            cov = result["reference_covariance"][np.ix_(take, take)]
            record["regions"][label] = dict(mean_ppm=float(w @ result["value"][take] * 1e6),
                reference_error_ppm=float(np.sqrt(w @ cov @ w) * 1e6))
        references.append(record)
    # Eclipse-only controls are independent of the science exposures.
    eclipse = phase >= .47
    arrays["phase_eclipse"] = phase[eclipse]
    bands = ((110, 210), (210, 300), (300, 390))
    controls = []
    for low, high in bands:
        paired = []
        for order in (0, 1):
            c = time_control(phase[eclipse], residual[eclipse, order],
                             error[eclipse, order], wave[order], low, high)
            key = f"order{order}_{low}_{high}"
            for field in ("centered", "covariance", "mode"):
                arrays[f"{key}_{field}"] = c.pop(field)
            c["slope"] = float(c["slope"])
            c["slope_error"] = float(c["slope_error"])
            c["trend_denominator"] = float(c["trend_denominator"])
            c.update(order=order, stellar_velocity_band_kms=[low, high])
            paired.append(c)
        for order in (0, 1):
            test, train = paired[order], paired[1 - order]
            # The held-out order retains its own nuisance pixel intercepts.
            # Only the temporal slope transfers across independent orders.
            test["heldout_order_sse_predicted"] = (test["sse_linear"]
                + (train["slope"] - test["slope"])**2 * test["trend_denominator"])
        controls.extend(paired)
    threshold_controls = []
    for cutoff in (.465, .47, .475, .48):
        selected = phase >= cutoff
        for order in (0, 1):
            c = time_control(phase[selected], residual[selected, order],
                             error[selected, order], wave[order], 110, 210)
            threshold_controls.append(dict(cutoff=cutoff, order=order,
                count=int(selected.sum()), slope_per_phase=float(c["slope"]),
                conditional_slope_error=float(c["slope_error"])))
    summary = dict(scope=__doc__.split("Example::")[0].strip(),
        geometry=dict(a_over_rstar=3.15, inclination_degrees=86.79,
                      rp_over_rstar=.08, outer_radius_rp=3, orbit="circular"),
        science_exposures=int(science.sum()), eclipse_exposures=int(eclipse.sum()),
        input_metadata=metadata, reference_windows=references, time_controls=controls,
        threshold_time_controls=threshold_controls,
        control_status=dict(common_master_cancellation="verified algebraically and by synthetic tests",
            reference_window_sensitivity="measured; no window selected for best fit",
            calibrated_stellar_variability_significance="unresolved: original noise covariance unknown",
            planet_free_reference="conditional: unverified timing metadata and no emission beyond 3 Rp",
            unique_temporal_background="unresolved: threshold/order/time-block tests do not establish a common linear trend",
            physical_phase_variability="unresolved: this control cannot attribute the night difference to the planet"),
        interpretation="A conditional eclipse/background control, not an adopted correction or calibrated variability significance.",
        limitations=[
            "Archive phase convention and adopted circular geometry determine eclipse membership; timestamps/ephemeris are absent from this HDF5.",
            "No H-alpha emission outside 3 Rp is assumed. A tail can violate this control.",
            "Nested reference windows and the same science spectra are correlated; they are not independent measurements.",
            "All errors assume independent supplied residual errors before this transformation; unknown original-master, extraction, time and order correlations remain.",
            "Log errors are first-order propagated; raw independent Gaussian noise need not remain Gaussian after division or logarithms.",
            "A held-out order is a replication check, not proof of independence: both orders share stellar, telluric and reduction effects.",
            "Temporal blocks are held out chronologically, not randomly; these 15 exposures cannot identify a unique stellar/reduction mechanism.",
            "Early-night data have no equivalent eclipse reference; changing the late-night reference alone cannot validate physical phase variability."
        ], sources={"data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "observations": "https://zenodo.org/records/20707255",
            "nightly_master_method": "https://arxiv.org/html/2606.23899v1",
            "geometry": "https://doi.org/10.1051/0004-6361/201936396"})
    np.savez_compressed(output / "eclipse_derived.npz", **arrays)
    (output / "eclipse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(output)


def plot(output: Path) -> None:
    """Plot only compact derived arrays; raw spectra are not required."""
    with np.load(output / "eclipse_derived.npz") as f:
        a = dict(f)
    summary = json.loads((output / "eclipse_summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(a["phase_all"], a["eclipse_margin_rstar"], "o", ms=3)
    ax.axhline(0, color="k", lw=.8)
    ax.axvspan(.47, .48, alpha=.15, color="tab:blue", label="6 early reference exposures")
    ax.axvspan(.48, .5, alpha=.15, color="tab:orange", label="9 late reference exposures")
    ax.set(xlabel="Supplied orbital phase", ylabel="Full-occultation margin / stellar radius",
           title="Conditional eclipse membership (outer radius 3 Rp)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for order, marker, color in ((0, "o", "tab:blue"), (1, "s", "tab:orange")):
        key = f"order{order}_110_210"
        control = next(c for c in summary["time_controls"] if c["order"] == order
                       and c["stellar_velocity_band_kms"][0] == 110)
        ax.errorbar(a["phase_eclipse"], a[f"{key}_centered"] * 1e6,
            np.sqrt(np.diag(a[f"{key}_covariance"])) * 1e6,
            marker=marker, ms=3, ls="none", color=color, label=f"Order {order + 1}")
        ax.plot(a["phase_eclipse"], control["slope"] * a[f"{key}_mode"] * 1e6, color=color)
    ax.axhline(0, color="k", lw=.8)
    ax.set(xlabel="Supplied orbital phase", ylabel="Centered log flux (ppm)",
           title="Stellar frame: 110–210 km/s; conditional errors")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(a["velocity_kms"], a["archived"] * 1e6, color="k", lw=1, label="Archived residual")
    for name, label, color in (("early_disjoint", "Early reference: 0.47–0.48", "tab:blue"),
                               ("late_0.480", "Late reference: 0.48–0.50", "tab:orange")):
        ax.plot(a["velocity_kms"], a[f"coadd_{name}"] * 1e6, color=color, label=label)
        ax.fill_between(a["velocity_kms"], (a[f"coadd_{name}"] - a[f"error_{name}"]) * 1e6,
                        (a[f"coadd_{name}"] + a[f"error_{name}"]) * 1e6, color=color, alpha=.15)
    ax.set(xlabel="Planet-frame velocity (km/s)", ylabel="Same 21 science exposures (ppm)",
           title="Disjoint references; shared-reference noise retained", xlim=(-50, 100))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for label, marker, color in (("red_peak", "o", "tab:red"), ("red_wing", "s", "tab:purple")):
        records = summary["reference_windows"][:4]
        ax.errorbar([float(c["name"].split("_")[1]) for c in records],
            [c["regions"][label]["mean_ppm"] for c in records],
            [c["regions"][label]["reference_error_ppm"] for c in records],
            marker=marker, color=color, label=label.replace("_", " "))
    ax.set(xlabel="Lower phase limit of reference window", ylabel="Remastered signal (ppm)",
           title="Nested-reference sensitivity (reference-only errors)")
    ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    fig.suptitle("Eclipse residual control: no background correction adopted", fontsize=13)
    fig.savefig(output / "eclipse_controls.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
    elif args.data is None:
        parser.error("--data is required unless --plot-only is used")
    else:
        run(args.data, args.output)
