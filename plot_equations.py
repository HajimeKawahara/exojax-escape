"""Render the simplified forward-model equations for the PR description."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Write a self-contained equation figure without requiring LaTeX."""
    output = Path(__file__).resolve().parent / "figures" / "equations.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 9.0), facecolor="white")
    fig.text(0.05, 0.955, "Prescribed multilayer hydrogen model", fontsize=20,
             weight="bold", color="#17324d")
    fig.text(0.05, 0.916,
             r"$x=r/R_{\rm p}$, $1\leq x\leq3$; layers ordered from top to bottom",
             fontsize=13, color="#444444")

    rows = [
        (
            0.850, "1. Excited-hydrogen column",
            0.795,
            r"$n_2(r)=\frac{N_2\,q(r)}{\sum_k q(r_k)\,\Delta r_k},"
            r"\qquad q(r)=\exp[-(x-1)/0.35]$",
        ),
        (
            0.685, "2. Level-population ratio",
            0.635,
            r"$\frac{n_3}{n_2}=\frac{g_3}{g_2}"
            r"\exp\!\left[-\frac{hc\widetilde{\nu}_0}{k_{\rm B}T(r)}\right]"
            r"b(r),\qquad \frac{g_3}{g_2}=\frac{18}{8}$",
        ),
        (
            0.450, "3. Accelerating radial outflow",
            0.400,
            r"$v_r(r)=v_\infty\left[1-\exp[-(x-1)/0.8]\right]$",
        ),
        (
            0.300, "4. Continuum-subtracted planet / star contrast",
            0.205,
            r"$C_{\widetilde{\nu}}="
            r"\left(\frac{R_{\rm p}}{R_\star}\right)^2"
            r"\frac{\sum_{p}"
            r"\mu_p\,\Delta\Omega_p\,[I_{\widetilde{\nu},p}-B_{\widetilde{\nu}}(T_{\rm b})]}"
            r"{\pi B_{\widetilde{\nu}}(T_\star)}$",
        ),
    ]
    for title_y, title, equation_y, equation in rows:
        fig.text(0.05, title_y, title, fontsize=14, weight="bold",
                 color="#17324d")
        fig.text(0.07, equation_y, equation, fontsize=18, color="#111111")

    fig.text(0.07, 0.744,
             r"$N_2$: column [cm$^{-2}$]; $n_2$: density [cm$^{-3}$]; "
             r"$\Delta r_k$: layer thickness [cm]", fontsize=12)
    fig.text(0.07, 0.560,
             r"$b(r)\equiv b_3/b_2=b_{\min}+"
             r"\frac{1-b_{\min}}{1+\exp[(x-1.45)/0.08]}$", fontsize=17)
    fig.text(0.07, 0.515,
             r"$T(r)=T_{\rm b}+(T_{\rm top}-T_{\rm b})"
             r"\{1-\exp[-(x-1)/0.12]\},\quad T_{\rm b}=4500$ K", fontsize=12)
    fig.text(0.07, 0.351,
             "Layer velocities shift the line profiles before radiative transfer.",
             fontsize=12)
    fig.text(0.07, 0.130,
             r"ExoJAX solves for $I$ using $\Delta\tau_k=\alpha_k\Delta r_k$ "
             r"and $S_k=j_k/\alpha_k$.", fontsize=12)
    fig.text(0.07, 0.085,
             r"Sum over visible dayside patches; weights are $\mu_p\Delta\Omega_p$.",
             fontsize=12)
    fig.text(0.05, 0.030,
             "Blackbody boundary and star; Gaussian instrument response follows disk integration.",
             fontsize=11, color="#444444")
    fig.savefig(output, dpi=120, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
