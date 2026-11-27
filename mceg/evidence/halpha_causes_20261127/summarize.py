"""Rebuild the compact comparison table from the three executed receipts."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def run():
    paths = [HERE / family / "review.json" for family in ("sources", "kinematics", "systematics")]
    source, motion, systematic = [json.loads(p.read_text()) for p in paths]
    comparisons = []
    source_cases = [
        ("Baseline", "baseline"),
        ("Upper source x0.5", "upper_source_half"),
        ("Inner x1.5, upper x0.5", "increased_source_contrast"),
        ("Upper opacity x2.014, fixed emission", "upper_lower_level_accumulation"),
        ("Free model 1, original shape", "free_model1_original"),
        ("Free model 1, refitted shape", "free_model1_refitted_shape")]
    motion_cases = [
        ("Fitted smooth longitude map", "all_data_longitude_fit"),
        ("Localized source patch x4", "patch_+30_3"),
        ("Upper LOS offset +5 km/s", "upper_LOS_+5"),
        ("Upper radial speed +13 km/s", "upper_radial_add_+13")]
    for label, name in source_cases + motion_cases:
        selections = {}
        for selection in ("primary", "alternate"):
            if (label, name) in source_cases:
                r = source["selections"][selection]["global_RV_amplitude_profiled"][name]["response"]
                residual, gain = r["target_ppm"]["residual"], -r["delta_Q"]
            else:
                r = motion["cases"][name]["selections"][selection]["profiled"]
                residual = [r["target"]["all"][band]["residual_ppm"] for band in ("blue", "core", "red")]
                gain = r["Q_improvement"]
            selections[selection] = dict(target_BCR_residual_ppm=residual, Q_improvement=gain)
        comparisons.append(dict(label=label, case=name, selections=selections))
    selected_systematics = {}
    for name, value in systematic["families"].items():
        r = value["trials"]["full"]
        selected_systematics[name] = dict(
            exact_primary_target_BCR_residual_ppm=r["exact_target_remaining_residual_ppm"],
            exact_primary_Q_improvement=r["exact_full_primary_delta_Q_improvement"],
            alternate_linear_Q_improvement=r["alternate_pixel_delta_Q_improvement"],
            maximum_modulation_by_night=r["maximum_modulation_by_night"])
    summary = dict(
        scope="Selected descriptive comparisons; no calibrated cause identification.",
        profile_comparison="Each fixed shape profiles the same two global RV/amplitude parameters; each interleave has its own profiled baseline. Longitude-map shape was selected on primary data.",
        source_and_input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [Path(__file__)]},
        comparisons=comparisons,
        systematics_fixed_original_RV_amplitude=selected_systematics,
        limitations=[
            "Complementary interleaves are correlated checks, not independent detections.",
            "Source/opacity changes do not solve atmospheric equilibrium; opacity-bracket refinement has a narrower scope than the fully refined source-only responses.",
            "A smaller selected residual can coexist with worse global Q; neither metric is a calibrated physical validation."])
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    tex = [r"\textbf{Executed comparison.} Positive $\Delta Q=Q_{\rm base}-Q_{\rm case}$ denotes a better full-native-data score. Each row below refits the same two global RV/amplitude parameters; the alternative pixels also use their own refitted baseline. Noise hyperparameters are fixed. Values are rounded ppm, and the original fixed-baseline alternative residuals in the introduction therefore differ slightly from this table.",
           r"\begin{center}\small\begin{tabular}{lrrr}\toprule",
           r"Shape & Primary B/C/R & Alternate B/C/R & $\Delta Q$, primary/alternate\\\midrule"]
    def triple(values):
        return "/".join(f"{x:+.0f}" for x in values)
    for row in comparisons:
        a, b = row["selections"]["primary"], row["selections"]["alternate"]
        label = row["label"].replace(" x", r" $\times$")
        tex.append(f"{label} & {triple(a['target_BCR_residual_ppm'])} & {triple(b['target_BCR_residual_ppm'])} & {a['Q_improvement']:+.2f}/{b['Q_improvement']:+.2f}" + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{center}",
        r"\textbf{Source and opacity.} Reducing upper emission improves the trough, but appreciable blue residual remains. Doubling upper opacity at fixed source gives only a small response; increasing the lower-state reservoir while keeping emission approximately fixed behaves much like lowering the source. A stronger inner source also adds red emission. Matching only the selected blue/core means with two source coefficients requires large changes and creates substantial collateral errors; it is not an admissible all-data solution.",
        r"\textbf{Kinematics and longitude.} A smooth positive longitude map improves primary $Q$ by less than one. A localized bright region can increase blue emission but barely deepens the core and worsens the global comparison. A uniform upper LOS offset is not a radial wind: the actual ray-projected radial acceleration moves the blue/core residual in the wrong direction. The LOS offset's global preference also changes with pixel selection.",
        r"\textbf{Stellar/calibration controls.} With the original planetary RV/amplitude held fixed, the full-data stellar-width drift changes the first night's width by at most 0.64\%; its exact target residuals are approximately $(+551,-548,-185)$ ppm and its primary $\Delta Q$ is 22.18. This removes only about 6.5\% of the blue-minus-core discrepancy. Depth pulsation and broad-curvature drift remove less than 1\%. A target-selected 8.45\% width drift can mimic much of the local shape, but its exact full-data $Q$ worsens by about 1471. It is not a viable correction supported by these data.",
        r"\textbf{Model 1 is informative, but not a physical identification.} Its original shape reduces both target residuals much more than uniform source scaling. Reoptimizing its shape further improves global $Q$ on primary pixels while leaving a larger local core error. The alternative-pixel global gain is small or absent. Thus a good local shape, a better global score and a physically realizable atmosphere are distinct requirements."]
    (ROOT / "doc/halpha_causes_results.tex").write_text("\n\n".join(tex) + "\n")


if __name__ == "__main__":
    run()
