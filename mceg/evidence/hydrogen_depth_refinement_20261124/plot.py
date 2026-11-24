"""Plot only the derived five-cell line-quadrature comparison in this bundle."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
data = np.load(root/'curves.npz')
meta = json.loads((root/'records.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
colors = ['#377eb8', '#e67e22', '#26855d']
for order, (name, label, color) in enumerate(zip(meta['cases'], meta['labels'], colors)):
    axes[0, 0].plot(data['radius_rp'], data[name+'_temperature_k'], 'o--' if order==2 else 'o-', color=color, markerfacecolor='none' if order==2 else color, label=label)
    axes[0, 1].semilogy(data['radius_rp'], data[name+'_n3']/data[name+'_n2'], 'o--' if order==2 else 'o-', color=color, markerfacecolor='none' if order==2 else color)
    axes[1, 0].plot(data['velocity_kms'], data[name+'_processed']*1e6, color=color, linestyle='--' if order==2 else '-', label=label)
    if name != meta['cases'][0]:
        delta = np.interp(data['observed_velocity_kms'], data['velocity_kms'],
                          data[name+'_processed']-data[meta['cases'][order-1]+'_processed'])
        axes[1, 1].plot(data['observed_velocity_kms'], delta/data['supplied_error'], color=color,
                        label='Depth 16 minus Gauss 8' if order==1 else 'Depth 32 minus Depth 16')
axes[0, 0].set(title='Solved temperature', xlabel=r'Radius / $R_p$', ylabel='Temperature [K]')
axes[0, 1].set(title='Solved hydrogen excitation', xlabel=r'Radius / $R_p$', ylabel=r'$n_3/n_2$')
for ax in axes[0]:
    ax.axvspan(1, meta['lower_boundary_edge_rp'], color='.6', alpha=.14)
    ax.set_xlim(1, 3)
axes[0, 0].legend(fontsize=8)
axes[1, 0].errorbar(data['observed_velocity_kms'], data['observed']*1e6,
    yerr=data['supplied_error']*1e6, fmt='.', color='.4', alpha=.48, ms=3,
    elinewidth=.6, label='Keck/KPF observation')
axes[1, 0].set(title='Same refined observer and processing', xlabel='Velocity from NIST H-alpha [km/s]',
    ylabel='Residual contrast [ppm]', xlim=(-105, 105))
axes[1, 0].legend(fontsize=8)
axes[1, 1].axhspan(-.1, .1, color='.6', alpha=.15, label='Declared numerical budget')
axes[1, 1].axhline(0, color='.5', lw=.7)
axes[1, 1].set(title='Original failure and subsequent refinement', xlabel='Observed velocity [km/s]',
    ylabel='Prediction difference / supplied uncertainty', xlim=(-105, 105))
axes[1, 1].legend(fontsize=8, loc='lower left', bbox_to_anchor=(.02,.14))
axes[1, 1].text(.03,.05,'Depth 16 to 32: maximum '+format(meta['refinement_comparison']['maximum_difference_over_supplied_error'],'.3g')+' supplied uncertainties', transform=axes[1, 1].transAxes, fontsize=8, color=colors[2])
for ax in axes.flat:
    ax.grid(alpha=.15)
fig.suptitle('Re-solving temperature, radiation, composition and wind: H line-depth integration only')
fig.supxlabel('Five solved cells; shaded lower boundary prescribed. Same atom, pressure boundary and ordinary-metal quadrature.\n'
    'This is a numerical control, not a fit or the later shared-IR atmosphere. Data: Zhang (2026), Zenodo 20707255, CC BY 4.0.', fontsize=8)
fig.savefig(root/'hydrogen_depth_refinement.png', dpi=170)
