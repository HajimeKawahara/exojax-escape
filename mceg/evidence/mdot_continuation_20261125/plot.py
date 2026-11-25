"""Render compact derived evidence; no stellar input or atmosphere evaluation."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})

def convergence():
    packet=json.loads((D/'root_summary.json').read_text())
    rows=packet['trajectory']
    fig,ax=plt.subplots(figsize=(8.2,4.5),layout='constrained')
    calls=[row['full_f_calls']for row in rows]
    ax.semilogy(calls,[row['infinity']for row in rows],'o-',color='#643a92',label='Maximum complete residual')
    ax.semilogy(calls,[row['l2']for row in rows],'s--',color='#56869b',alpha=.85,label='L2 residual (update merit)')
    ax.axhline(packet['unchanged_root_criterion'],color='black',ls=':',label='Unchanged maximum-residual criterion')
    ax.set(xlabel='New complete-residual evaluations in the final refinement window',ylabel='Dimensionless residual',
           title='The +5.127% mass-loss atmosphere now converges')
    ax.text(.02,.04,'Fresh repeat: identical residual\nConservation error < 3.61 × 10⁻¹⁶',transform=ax.transAxes,
            fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    ax.legend(loc='upper right',fontsize=9);ax.grid(alpha=.2)
    fig.savefig(D/'convergence.png',dpi=170);fig.savefig(D/'convergence.pdf');plt.close(fig)

def spectrum():
    a=dict(np.load(D/'curves.npz'));s=json.loads((D/'fit_summary.json').read_text())
    fig,(ax,res)=plt.subplots(2,1,figsize=(9,6.6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2.5,1]})
    v=a['observed_velocity_kms'];error=a['supplied_error'];obs=a['observed']
    ax.errorbar(v,obs*1e6,yerr=error*1e6,fmt='.',color='#686868',alpha=.65,ms=4,lw=.7,label='Zenodo observations')
    for key,color,label in [('baseline','#2679af',f"Previous atmosphere, χ² = {s['baseline_chi2']:.2f}"),
                             ('best','#a04677',f"Mass loss +5.127%, χ² = {s['best_chi2']:.2f}")]:
        ax.plot(a['output_velocity_kms'],a[key+'_processed']*1e6,color=color,lw=1.8,label=label)
        res.plot(v,(a[key+'_prediction']-obs)/error,'.-',ms=3,lw=.8,color=color)
    ax.set(ylabel='Processed contrast (ppm)',title='Own coupled atmospheres, same spherical observer and processing')
    ax.legend(fontsize=9,loc='upper left');ax.grid(alpha=.18)
    res.axhline(0,color='black',lw=.7);res.axhspan(-1,1,color='grey',alpha=.1)
    res.set(xlabel='Planet-frame velocity (km/s)',ylabel='(Model − data) / error',xlim=(v.min()-3,v.max()+3))
    res.grid(alpha=.18)
    fig.get_layout_engine().set(rect=(0,.035,1,.965))
    fig.text(.5,.01,'Zonal speed fixed near 20 km/s; only the new velocity offset is refitted. No amplitude or baseline adjustment.',ha='center',fontsize=9)
    fig.savefig(D/'spectrum.png',dpi=170);fig.savefig(D/'spectrum.pdf');plt.close(fig)

if __name__=='__main__':
    convergence()
    if (D/'curves.npz').exists():spectrum()
