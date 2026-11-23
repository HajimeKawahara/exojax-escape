"""Explain the coarse-density error without claiming a thermal solution."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

root = Path('/tmp/escape-flow-gradient-audit')
out = Path('/tmp/escape-solver-continuation-publication')
diagnosis = json.loads((root/'diagnosis.json').read_text())
refinement = json.loads((root/'resolution_1025.json').read_text())
profile = np.load('/tmp/escape-five-cell-root-evidence/profile.npz')
radius = profile['radius_cm']/profile['base_radius_cm']
coarse = -np.asarray(diagnosis['legacy_expansion'])
same_wind = -np.asarray(refinement[0]['expansion'])
reference = -np.asarray(refinement[-1]['expansion'])
grid = np.array([row['ngrid'] for row in refinement[:-1]])
error = 100*np.array([row['expansion_maximum_relative_to_1025']
                      for row in refinement[:-1]])
np.savez_compressed(out/'flow_plot_data.npz', radius_rp=radius,
                    legacy_cooling=coarse, same_wind_cooling=same_wind,
                    reference_cooling=reference, wind_grid=grid,
                    expansion_error_percent=error)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3),
                         gridspec_kw={'width_ratios': [1.25, 1]})
ax = axes[0]
ax.axvspan(1., 1.13333333333333, color='0.92', label='Prescribed dense cells')
ax.semilogy(radius, coarse, 's--', color='#b44a3a',
            label='Density differences, 65-point wind')
ax.semilogy(radius, same_wind, 'o-', color='#2673a8',
            label='Wind derivative, same 65 points')
ax.semilogy(radius, reference, '.', color='black', markersize=6,
            label='Wind derivative, 1025 points')
ax.annotate(r'$\sim 4.2\times10^3$ excess', xy=(radius[-1], coarse[-1]),
            xytext=(1.65, 6e-5),
            arrowprops={'arrowstyle': '->', 'color': '#b44a3a'},
            color='#913c30')
ax.set(xlabel=r'Radius $r/R_p$', ylabel=r'Expansion cooling $-Q_{\rm exp}$ [erg cm$^{-3}$ s$^{-1}$]',
       xlim=(1., 2.24), ylim=(8e-8, 2e-3), title='Differentiate the same wind')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=.16)
ax = axes[1]
ax.plot(grid, error, 'o-', color='#2673a8')
ax.axhline(1., color='0.4', linestyle='--', label='1% numerical criterion')
ax.scatter([129], [error[1]], s=85, facecolors='none', edgecolors='#cc7a12', zorder=3)
ax.annotate('129 points: 0.622%', xy=(129, error[1]), xytext=(180, .79),
            arrowprops={'arrowstyle': '->', 'color': '#996016'}, color='#996016')
ax.set(xscale='log', xlabel='Wind collocation points',
       ylabel='Maximum cooling difference from 1025 points [%]',
       title='Refine energy terms as well as velocity', ylim=(0, 1.3))
ax.set_xticks(grid, [str(n) for n in grid])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=.16)
fig.suptitle('Fixed temperature and composition profile: a numerical energy repair', y=.985)
fig.text(.5, .015, 'Five atmospheric cells; this is not a thermal-equilibrium solution. '
         'The final solved atmosphere needs its own refinement check.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .055, 1, .93))
fig.savefig(out/'flow_control.png', dpi=180)
