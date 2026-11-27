"""Generate report tables from the qualified physical and observer receipts."""
import json
import math
from pathlib import Path


def main():
    campaign = Path(__file__).resolve().parent
    root = campaign.parents[2]
    numerical = json.loads((campaign / "numerical_summary.json").read_text())
    structure = json.loads((campaign / "review/structure.json").read_text())
    observations = json.loads((campaign / "observer/comparison.json").read_text())
    accepted = {case["factor"] for case in numerical["accepted"]}
    selected = {float(row["case"][1:]) for row in observations["rows"]
                if row["selection"] == "primary" and row["nuisance_mode"] == "profiled"
                and row["case"] != "D1.2"} | {max(accepted)}
    lines = [r"\textbf{Coupled density response.}",
        r"At $r=1.9375R_p$ (cell 5), ratios are relative to the $D=1$ spherical root.",
        r"$N_2$ is the radial column above $1.725R_p$; $r_{\tau=1}$ uses the static nominal line center.",
        r"\begin{center}\begin{tabular}{rrrrrr}", r"\toprule",
        r"$D$ & $n_H/n_{H,1}$ & $n_2/n_{2,1}$ & $S/S_1$ & $N_2$ [cm$^{-2}$] & $r_{\tau=1}/R_p$ \\",
        r"\midrule"]
    for case in structure["candidates"]:
        if not any(math.isclose(case["factor"], factor, rel_tol=1e-12) for factor in selected):
            continue
        ratio = case["ratio_to_D1"]
        column = case["n2_column_above_radius_cm2"]["1.725"]
        lines.append(f"{case['factor']:g} & {ratio['n_h_cm3'][5]:.3f} & "
            f"{ratio['n2_cm3'][5]:.3f} & {ratio['nominal_center_total_source_per_wavenumber'][5]:.3f} & "
            f"{column:.3e} & {case['static_center_tau_one_radius_rp']:.3f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{center}",
        r"\textbf{Matched native observations.}",
        r"Primary pixels; the same two global nuisance parameters are profiled for each shape.",
        r"Residuals are data minus the sum of planet and conditional-GP predictions, in ppm.",
        r"Positive $\Delta Q$ means a smaller full-data score.",
        r"\begin{center}\begin{tabular}{rrrrrrr}\toprule",
        r"$D$ & Blue & Core & Red & Blue$-$core & $\Delta Q$ vs PR13 & $\Delta Q$ vs $D=1$ \\",
        r"\midrule"]
    for row in observations["rows"]:
        if row["selection"] != "primary" or row["nuisance_mode"] != "profiled":
            continue
        if float(row["case"][1:]) not in selected:
            continue
        assert row["status"] in ("previous_accepted_D1_control", "independent_CPU_verified")
        b, c, r = row["residual_BCR_ppm"]
        lines.append(f"{row['case'][1:]} & {b:+.1f} & {c:+.1f} & {r:+.1f} & {b-c:.1f} & "
                     f"{row['Q_gain_vs_PR13']:+.2f} & {row['Q_gain_vs_D1']:+.2f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{center}",
        r"The PR13 primary reference has B/C/R residuals $+590.5/-585.5/-215.8$ ppm.",
        r"The complementary pixels, fixed-nuisance controls, per-order results, all-phase collateral changes",
        r"and optimizer-bound distances are retained in \texttt{observer/comparison.json}."]
    (root / "doc/halpha_dense_outer_tables.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
