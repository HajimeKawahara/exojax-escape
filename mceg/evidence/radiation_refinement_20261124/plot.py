"""Plot the paired radiation-grid comparison from safe derived arrays."""
from pathlib import Path
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent
a = np.load(root/'plot_arrays.npz')
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]}, constrained_layout=True)
axes[0].errorbar(a['observed_velocity'], a['observed']*1e6,
    yerr=a['supplied_error']*1e6, fmt='.', color='.5', alpha=.45,
    ms=3, elinewidth=.6, label='Keck/KPF observation')
axes[0].plot(a['velocity'], a['coarse']*1e6, color='#4477aa', label='Radiation orders 2 / 2 / 4')
axes[0].plot(a['velocity'], a['fine']*1e6, color='#cc6677', label='Radiation orders 4 / 4 / 8')
axes[0].set(ylabel='Processed contrast [ppm]',
    title='Radiation refinement changes the coupled thermal spectrum')
axes[0].legend(fontsize=9)
axes[1].plot(a['observed_velocity'], a['difference_over_error'], color='#cc6677')
axes[1].axhspan(-.1, .1, color='#228833', alpha=.12, label='Declared numerical budget: 0.1')
axes[1].axhline(0, color='.5', lw=.6)
axes[1].set(xlabel='Velocity from NIST H-alpha [km/s]',
    ylabel='Change / supplied error', xlim=(-105, 105), ylim=(-.65, .65))
axes[1].legend(fontsize=8, loc='lower left')
for axis in axes:
    axis.grid(alpha=.15)
fig.supxlabel('Orders: continuum / angle / cell-line quadrature. Both five-cell thermal roots are solved.\n'
    'Same refined observer and approximate processing; no fit or radial-convergence claim.\n'
    'Data: Yapeng Zhang (2026), doi:10.5281/zenodo.20707255, CC BY 4.0.', fontsize=8)
fig.savefig(root/'radiation_refinement.png', dpi=180)
fig.savefig(root/'radiation_refinement.pdf')
