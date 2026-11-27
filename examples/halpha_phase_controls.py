"""Conditional native phase-band corrections and observation-response controls.

The rectangular additive log-flux basis is a residual diagnostic, not an
atmosphere or an intrinsic emission profile. A partly observed band does not
measure its unobserved portion. No new atmosphere is solved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gp_halpha import load_gp_data, make_gp_problem
from halpha_data_consistency import PARAMETERS
from halpha_noise_audit import covariance

jax.config.update("jax_enable_x64", True)
C = 299792.458
PHASE_EDGES = np.array([.30, .325, .35, .375, .40, .425, .435])
BANDS = dict(blue=(-40., -16.), core=(-8., 8.), red=(16., 40.))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common(group, values):
    weight = group["weight"]
    expanded = weight[..., None]
    total = weight.sum(axis=0)[None, :, None]
    mean = np.sum(expanded * values, axis=0, keepdims=True) / np.where(total > 0, total, 1.)
    return np.where(expanded > 0, values - mean, 0.)


def gp_mean(group, state, cov, values):
    basis = np.asarray(state.basis)
    product = np.einsum("mp,ep,epk->mek", basis, group["weight"], common(group, values))
    coefficients = cov["estimator"] @ product.reshape(-1, values.shape[-1])
    return np.einsum("mek,mp->epk", coefficients.reshape(*product.shape), basis)


def precision(group, state, cov, values):
    """Apply the existing common-reference, GP-marginalized precision."""
    return group["weight"][..., None] * common(
        group, values - gp_mean(group, state, cov, values)) / cov["scale"]**2


def solve_scaled(gram, rhs, relative_cutoff=1e-10):
    """Use a unit-diagonal Gram matrix so RV/log-amplitude units do not set rank."""
    scale = np.sqrt(np.maximum(np.diag(gram), np.finfo(float).tiny))
    normalized = (gram / scale[:, None]) / scale[None, :]
    normalized = (normalized + normalized.T) / 2
    values, vectors = np.linalg.eigh(normalized)
    keep = values > relative_cutoff * max(values[-1], np.finfo(float).tiny)
    inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    inverse = (inverse / scale[:, None]) / scale[None, :]
    answer = inverse @ rhs
    return answer, inverse, dict(rank=int(keep.sum()), dimension=len(gram),
        normalized_eigenvalues=values.tolist(), relative_rank_cutoff=relative_cutoff,
        retained_condition_number=float(values[-1] / values[keep][0]) if keep.any() else None)


def read_maps(directory, product):
    metadata = json.loads((directory / "rows.json").read_text())
    indices = [i for i, row in enumerate(metadata["rows"])
               if row["grid"] == "fine" and row["product"] == product and row["scope"] == "all"]
    if len(indices) != 18:
        raise ValueError("Expected six phase bins by three bands in the all-night map.")
    support = np.array([metadata["selections"]["primary"]["rows"][i]["supported"] for i in indices])
    with np.load(directory / "maps.npz") as saved:
        maps = [saved[f"primary__g{i}__maps"][indices] for i in range(4)]
        raw = [saved[f"primary__g{i}__raw_maps"][indices] for i in range(4)]
    return maps, support, [metadata["rows"][i] for i in indices], raw


def matrix_summary(matrix, supported, active):
    selected = matrix[np.ix_(supported, active)]
    singular = np.linalg.svd(selected, compute_uv=False)
    cutoff = 1e-10 * singular[0] if len(singular) else 0.
    rank = int(np.sum(singular > cutoff))
    denominator = np.sum(abs(matrix), axis=0)
    diagonal = np.abs(np.diag(matrix))
    return dict(supported_rows=int(supported.sum()), active_columns=int(active.sum()), rank=rank,
        native_active_nullity=int(active.sum()) - rank, singular_values=singular.tolist(),
        off_diagonal_absolute_response_fraction=np.divide(denominator - diagonal, denominator,
            out=np.zeros_like(denominator), where=denominator > 0).tolist())


def run(args):
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), args.maps / "maps.npz", args.maps / "rows.json", args.baseline,
        args.stellar, args.noise, *args.observations,
        *[root / "examples" / name for name in ("gp_halpha.py", "phase_halpha.py", "cloudy_halpha.py",
                                               "halpha_noise_audit.py", "halpha_data_consistency.py")],
        root / "src/escape/time_gp.py"]
    pins = {str(path.resolve()): digest(path) for path in paths}
    groups = load_gp_data(args.observations)
    problem = make_gp_problem(groups, dict(np.load(args.baseline)), dict(np.load(args.stellar)))
    states = problem["states"]
    noise = {key: np.median(value, axis=(0, 1)) for key, value in dict(np.load(args.noise)).items()}
    covs = [covariance(state, group, noise, i) for i, (state, group) in enumerate(zip(states, groups))]
    models = [np.asarray(value) for value in problem["native_log_models"](jnp.asarray(PARAMETERS))]
    jacobians = [np.asarray(value) for value in jax.jacfwd(problem["native_log_models"])(jnp.asarray(PARAMETERS))]
    with np.load(args.maps / "maps.npz") as saved_maps:
        for i, (group, model) in enumerate(zip(groups, models)):
            for key in ("phase", "stellar_velocity", "pixel_index", "weight", "science", "eclipse", "log_data"):
                np.testing.assert_array_equal(saved_maps[f"primary__g{i}__{key}"], group[key])
            np.testing.assert_allclose(saved_maps[f"primary__g{i}__model"], model, rtol=0, atol=1e-12)
    bases, residuals, coverage = [], [], []
    labels = [dict(phase_index=i, phase_range=PHASE_EDGES[i:i+2].tolist(), band=band)
              for i in range(6) for band in BANDS]
    for group, model in zip(groups, models):
        orbital = 244.8 * np.sin(2 * np.pi * group["phase"])
        velocity = C * ((1 + group["stellar_velocity"][None] / C) / (1 + orbital[:, None] / C) - 1)
        basis = np.stack([group["science"][:, None]
            * ((group["phase"] >= PHASE_EDGES[i]) & (group["phase"] < PHASE_EDGES[i + 1]))[:, None]
            * ((velocity >= left) & (velocity < right)) * (group["weight"] > 0)
            for i in range(6) for left, right in BANDS.values()], axis=-1).astype(float)
        bases.append(basis)
        residuals.append((group["log_data"] - model)[..., None])
        coverage.append(dict(label=group["label"],
            native_samples_by_column=basis.sum(axis=(0, 1)).astype(int).tolist(),
            exposures_with_any_support_by_column=(basis.sum(axis=1) > 0).sum(axis=0).tolist()))
    p_basis = [precision(g, s, c, d) for g, s, c, d in zip(groups, states, covs, bases)]
    p_jac = [precision(g, s, c, j) for g, s, c, j in zip(groups, states, covs, jacobians)]
    p_data = [precision(g, s, c, r) for g, s, c, r in zip(groups, states, covs, residuals)]
    outside_data = [common(g, r) * (d.sum(axis=-1) == 0)[..., None]
                    for g, r, d in zip(groups, residuals, bases)]
    p_outside = [precision(g, s, c, r) for g, s, c, r in zip(groups, states, covs, outside_data)]
    gram = sum(np.einsum("epk,epl->kl", d, p) for d, p in zip(bases, p_basis))
    rhs = sum(np.einsum("epk,epq->kq", d, p)[:, 0] for d, p in zip(bases, p_data))
    rhs_outside = sum(np.einsum("epk,epq->kq", d, p)[:, 0] for d, p in zip(bases, p_outside))
    weighted_norm = sum(np.einsum("ep,epk->k", g["weight"], d*d) for g, d in zip(groups, bases))
    active = weighted_norm > 0
    a = np.flatnonzero(active)
    beta, beta_cov, fixed = solve_scaled(gram[np.ix_(a, a)], rhs[a])
    nuisance_gram = sum(np.einsum("epk,epl->kl", j, p) for j, p in zip(jacobians, p_jac))
    nuisance_rhs = sum(np.einsum("epk,epq->kq", j, p)[:, 0] for j, p in zip(jacobians, p_data))
    nuisance_rhs_outside = sum(np.einsum("epk,epq->kq", j, p)[:, 0] for j, p in zip(jacobians, p_outside))
    cross = sum(np.einsum("epk,epl->kl", d, p) for d, p in zip(bases, p_jac))
    _, nuisance_inverse, nuisance_rank = solve_scaled(nuisance_gram, nuisance_rhs)
    profiled_gram = gram - cross @ nuisance_inverse @ cross.T
    profiled_rhs = rhs - cross @ nuisance_inverse @ nuisance_rhs
    beta_profiled, profiled_cov, profiled = solve_scaled(profiled_gram[np.ix_(a, a)], profiled_rhs[a])
    nuisance_shift = nuisance_inverse @ (nuisance_rhs - cross[a].T @ beta_profiled)
    baseline_q = float(sum(np.sum(r * p) for r, p in zip(residuals, p_data)))
    fixed.update(correction_ppm=(beta * 1e6).tolist(),
        outside_basis_residual_contribution_ppm=(beta_cov @ rhs_outside[a] * 1e6).tolist(),
        conditional_standard_error_ppm=(np.sqrt(np.maximum(np.diag(beta_cov), 0.)) * 1e6).tolist(),
        Q_after_local_corrections=baseline_q - float(rhs[a] @ beta))
    profiled.update(correction_ppm=(beta_profiled * 1e6).tolist(),
        outside_basis_residual_contribution_ppm=(profiled_cov @ (
            rhs_outside - cross @ nuisance_inverse @ nuisance_rhs_outside)[a] * 1e6).tolist(),
        conditional_standard_error_ppm=(np.sqrt(np.maximum(np.diag(profiled_cov), 0.)) * 1e6).tolist(),
        local_RV_log_amplitude_shift=nuisance_shift.tolist())

    # Exact additive-log injections test this finite-dimensional diagnostic,
    # without using its response as a guarantee about physical identifiability.
    truth = 200e-6 * np.sin(np.arange(len(a)) + .3)
    recovered = beta_cov @ (gram[np.ix_(a, a)] @ truth)
    injected_error = float(np.max(abs(recovered - truth)))
    constant_leak = max(float(np.max(abs(precision(g, s, c,
        np.broadcast_to(np.sin(g["stellar_velocity"] / 37)[None, :, None], (*g["weight"].shape, 1))))))
        for g, s, c in zip(groups, states, covs))

    arrays = dict(native_gram=gram, native_rhs=rhs, active_columns=active,
        fixed_correction=beta, fixed_covariance=beta_cov, profiled_correction=beta_profiled,
        profiled_covariance=profiled_cov, nuisance_gram=nuisance_gram)
    responses = {}
    direct_identity_error, common_leakage = 0., 0.
    for product in ("common_complete", "band_complete"):
        maps, supported, rows, raw_maps = read_maps(args.maps, product)
        if [(row["phase_index"], row["band"]) for row in rows] != [
                (label["phase_index"], label["band"]) for label in labels]:
            raise ValueError("Map row order does not match the declared phase-band basis.")
        if any(m.shape[1:] != g["weight"].shape for m, g in zip(maps, groups)):
            raise ValueError("Saved maps and current native groups have different support.")
        direct = sum(np.einsum("bep,epk->bk", m, d) for m, d in zip(raw_maps, bases))
        direct_identity_error = max(direct_identity_error, float(np.max(abs(direct - np.diag(supported.astype(float))))))
        common_leakage = max(common_leakage, max(float(np.max(abs(m.sum(axis=1)))) for m in maps))
        common_response = sum(np.einsum("bep,epk->bk", m, d) for m, d in zip(maps, bases))
        gp_basis = [d - gp_mean(g, s, c, d) for g, s, c, d in zip(groups, states, covs, bases)]
        gp_jac = [j - gp_mean(g, s, c, j) for g, s, c, j in zip(groups, states, covs, jacobians)]
        gp_response = sum(np.einsum("bep,epk->bk", m, d) for m, d in zip(maps, gp_basis))
        band_jac = sum(np.einsum("bep,epk->bk", m, j) for m, j in zip(maps, gp_jac))
        profiled_response = gp_response - band_jac @ nuisance_inverse @ cross.T
        variants = dict(common=common_response, gp=gp_response, gp_local_nuisance=profiled_response)
        responses[product] = dict(supported=supported.tolist(), rows=rows,
            stages={name: matrix_summary(value, supported, active) for name, value in variants.items()},
            response={name: value.tolist() for name, value in variants.items()})
        arrays.update({product + "__" + name: value for name, value in variants.items()})
        arrays[product + "__supported"] = supported

    # A planet-pattern fixed in stellar wavelength is invisible without an
    # eclipse anchor. The same pattern, zero in eclipse, is not a common mode.
    gauge = []
    for night in (0, 1):
        white_norm, marginalized_norm = 0., 0.
        for group, state, cov in zip(groups, states, covs):
            if group["night"] != night:
                continue
            pattern = (group["science"][:, None] * np.exp(-.5 * ((group["stellar_velocity"] - 170) / 30)**2)[None])[..., None]
            white_norm += np.sum(group["weight"][..., None] * pattern**2) / cov["scale"]**2
            marginalized_norm += np.sum(pattern * precision(group, state, cov, pattern))
        gauge.append(dict(night=night, precision_norm_squared=float(marginalized_norm),
            fraction_of_unprojected_white_norm=float(marginalized_norm / white_norm)))
    full_band = np.array(responses["band_complete"]["supported"])
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "arrays.npz"
    np.savez_compressed(raw_path, **arrays)
    checks = dict(input_unchanged=all(digest(path) == value for path, value in pins.items()),
        baseline_Q_reproduced=abs(baseline_q - 38326.64894003179) < 1e-6,
        finite_corrections=bool(np.isfinite(beta).all() and np.isfinite(beta_profiled).all()),
        direct_map_injection_identity=direct_identity_error < 1e-12,
        common_reference_map_annihilated=common_leakage < 1e-12,
        full_rank_native_injection_recovery=(injected_error < 1e-10 if fixed["rank"] == len(a) else True))
    review = dict(scope=__doc__.strip(), source_and_input_sha256=pins, checks=checks, passed=all(checks.values()),
        phase_edges=PHASE_EDGES.tolist(), bands_kms=BANDS, baseline_parameters=PARAMETERS.tolist(),
        labels=labels, active_column_indices=a.tolist(),
        unobserved_columns=np.flatnonzero(~active).tolist(),
        columns_with_only_partial_band_support=np.flatnonzero(active & ~full_band).tolist(),
        native_coverage=coverage, baseline_Q=baseline_q, fixed_nuisance=fixed, local_nuisance_profile=profiled,
        nuisance_rank=nuisance_rank, responses=responses, stationary_stellar_pattern_gauge=gauge,
        finite_dimensional_injection_maximum_error=injected_error,
        native_inputs_match_saved_maps=True, direct_map_injection_identity_maximum_error=direct_identity_error,
        common_reference_map_leakage_maximum_abs=common_leakage,
        constant_reference_precision_action_maximum_abs=constant_leak,
        raw_arrays=dict(path=str(raw_path), sha256=digest(raw_path)),
        limitations=["The fitted coefficients add log flux only on observed native samples; they are not intrinsic line emissivities or physical atmosphere parameters.",
            "Rectangular coefficients assume a constant correction across each observed phase-band patch. Partial support cannot measure the missing part of that band.",
            "Outside these patches the mean stays at the baseline. Correlated residuals outside them can affect the GLS corrections through common referencing and GP precision.",
            "For the outside-basis contribution, first remove the full weighted common spectrum to fix the arbitrary reference, then zero residuals inside the selected native patches. This exact linear decomposition is reference-invariant, but not a separately estimated contaminant.",
            "Displayed phase bands mix under shared referencing, conditional GP fitting, and global RV/amplitude removal. Display-response rank is distinct from native-basis rank.",
            "The time-constant stellar-wavelength gauge remains in a night without eclipse. An eclipse anchor breaks the demonstrated science-only pattern gauge only under the zero-planet-in-eclipse assumption.",
            "Reported conditional standard errors use the fixed archived GP family; they are not calibrated significances. Local nuisance profiling is linear, not a new nonlinear fit.",
            "Noiseless recovery establishes only this chosen finite-dimensional model; it cannot prove unique physical phase localization."])
    args.review.mkdir(parents=True, exist_ok=True)
    (args.review / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    plot(responses["band_complete"], args.review / "phase_response.png")
    print(json.dumps(dict(checks=checks, fixed_rank=fixed["rank"], profiled_rank=profiled["rank"],
        active_columns=a.tolist(), fixed_correction_ppm=fixed["correction_ppm"], gauge=gauge)), flush=True)
    if not review["passed"]:
        raise RuntimeError("A phase control failed; inspect the receipt.")


def plot(response, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    supported = np.array(response["supported"])
    labels = [f"{i+1}{b}" for i in range(6) for b in "BCR"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    values = [np.asarray(value) for value in response["response"].values()]
    limit = max(np.max(abs(value)) for value in values)
    color = plt.get_cmap("coolwarm").copy()
    color.set_bad("0.8")
    for ax, (name, value) in zip(axes, response["response"].items()):
        value = np.where(supported[:, None], value, np.nan)
        im = ax.imshow(value, cmap=color, vmin=-limit, vmax=limit)
        ax.set(title=name, xlabel="Injected native phase-band correction",
            xticks=np.arange(18), xticklabels=labels, yticks=np.arange(18), yticklabels=labels)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("Displayed fully supported phase-band mean")
    fig.suptitle("Phase bins 1–6; B/C/R bands. Grey rows have no full-band support.")
    fig.subplots_adjust(bottom=.2, top=.85, right=.87, wspace=.15)
    color_axis = fig.add_axes([.90, .2, .015, .65])
    fig.colorbar(im, cax=color_axis, label="Output / injected log flux")
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("/tmp/escape-halpha-phase-localization"))
    parser.add_argument("--observations", nargs=2, type=Path, default=[Path(f"/tmp/escape-observation-processing/KELT9b_2024-08-{day}_line_w6565.h5") for day in ("08", "20")])
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/escape-e8-mdot-sphere/fit_arrays.npz"))
    parser.add_argument("--stellar", type=Path, default=Path("/tmp/escape-stellar-sed-data/phoenix10000_surface_flux.npz"))
    parser.add_argument("--noise", type=Path, default=Path("/tmp/escape-gp-nuts-20261125/gp/samples.npz"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/escape-halpha-phase-controls"))
    parser.add_argument("--review", type=Path, default=Path("mceg/evidence/halpha_phase_localization_20261127/controls"))
    run(parser.parse_args())
