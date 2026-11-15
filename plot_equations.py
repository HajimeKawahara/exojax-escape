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
    fig = plt.figure(figsize=(10.8, 11.5), facecolor="white")
    fig.text(0.05, 0.965, "Hydrogen model with an isothermal Parker wind", fontsize=20,
             weight="bold", color="#17324d")
    fig.text(0.05, 0.933,
             r"$x=r/R_{\rm p}$, $1\leq x\leq3$; layers ordered from top to bottom",
             fontsize=13, color="#444444")

    rows = [
        (
            0.890, "1. Prescribed excited-hydrogen column",
            0.845,
            r"$n_2(r)=\frac{N_2\,q(r)}{\sum_k q(r_k)\,\Delta r_k},"
            r"\qquad q(r)=\exp[-(x-1)/0.35]$",
        ),
        (
            0.755, "2. Prescribed level-population ratio",
            0.710,
            r"$\frac{n_3}{n_2}=\frac{g_3}{g_2}"
            r"\exp\!\left[-\frac{hc\widetilde{\nu}_0}{k_{\rm B}T(r)}\right]"
            r"b(r),\qquad \frac{g_3}{g_2}=\frac{18}{8}$",
        ),
        (
            0.565, "3. Isothermal transonic Parker wind",
            0.520,
            r"$u^2-\ln(u^2)=4\ln(r/r_{\rm s})+4r_{\rm s}/r-3,"
            r"\qquad u=v_r/c_{\rm s}$",
        ),
        (
            0.315, "4. Continuum-subtracted planet / star contrast",
            0.245,
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

    fig.text(0.07, 0.803,
             r"$N_2$: column [cm$^{-2}$]; $n_2$: density [cm$^{-3}$]; "
             r"$\Delta r_k$: layer thickness [cm]", fontsize=12)
    fig.text(0.07, 0.655,
             r"$b(r)\equiv b_3/b_2=b_{\min}+"
             r"\frac{1-b_{\min}}{1+\exp[(x-1.45)/0.08]}$", fontsize=17)
    fig.text(0.07, 0.615,
             r"$T(r)=T_{\rm b}+(T_{\rm top}-T_{\rm b})"
             r"\{1-\exp[-(x-1)/0.12]\},\quad T_{\rm b}=4500$ K", fontsize=12)
    fig.text(0.07, 0.468,
             r"$c_{\rm s}^2=\frac{k_{\rm B}T_{\rm wind}}{\bar{\mu}m_{\rm H}},"
             r"\qquad r_{\rm s}=\frac{GM_{\rm p}}{2c_{\rm s}^2}$", fontsize=17)
    fig.text(0.07, 0.428,
             r"Choose $u<1$ below $r_{\rm s}$ and $u>1$ above; $u(r_{\rm s})=1$.",
             fontsize=12)
    fig.text(0.07, 0.396,
             r"$T_{\rm wind}$ is an effective constant; $\bar{\mu}$ is constant; line $T(r)$ remains prescribed.",
             fontsize=12)
    fig.text(0.07, 0.365,
             "Planetary gravity only; no tidal term or arbitrary velocity rescaling.",
             fontsize=12)
    fig.text(0.07, 0.180,
             r"ExoJAX solves for $I$ using $\Delta\tau_k=\alpha_k\Delta r_k$ "
             r"and $S_k=j_k/\alpha_k$.", fontsize=12)
    fig.text(0.07, 0.145,
             "Layer velocities shift the line profiles before radiative transfer.",
             fontsize=12)
    fig.text(0.07, 0.110,
             r"Sum over visible dayside patches; weights are $\mu_p\Delta\Omega_p$.",
             fontsize=12)
    fig.text(0.05, 0.055,
             "Blackbody boundary and star; Gaussian instrument response follows disk integration.",
             fontsize=11, color="#444444")
    fig.savefig(output, dpi=120, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
