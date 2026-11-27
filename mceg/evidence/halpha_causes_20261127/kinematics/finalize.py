"""Keep a compact review and plot the fixed-nuisance finite responses."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[3]
sys.path.insert(0,str(ROOT/'examples'))
from halpha_geometry_fit import sha256

path=OUTPUT/'review.json'
review=json.loads(path.read_text())
if 'all_cases_compact' not in review:
    full=Path('/tmp/escape-halpha-cause-kinematics/full_review.json')
    full.write_text(path.read_text())
    keep=['baseline','east_west_+1','substellar_+2','patch_+30_3','patch_+30_9','patch_+45_9',
          'upper_LOS_+2','upper_LOS_+5','v_zero','v_double','upper_radial_add_+5.5','upper_radial_add_+13','all_data_longitude_fit']
    review['all_cases_compact']={name:{selection:{mode:dict(Q=result['Q'],Q_improvement=result['Q_improvement'],
        parameters=result['parameters'],target_residual_BCR_ppm=[result['target']['all'][b]['residual_ppm'] for b in ['blue','core','red']],
        target_model_plus_GP_change_BCR_ppm=[result['target']['all'][b]['change_in_planet_plus_GP_ppm'] for b in ['blue','core','red']],
        nuisance_success=result['nuisance_success']) for mode,result in selected.items()} for selection,selected in value['selections'].items()}
        for name,value in review['cases'].items()}
    review['cases']={name:review['cases'][name] for name in keep}
    review['full_review']=dict(path=str(full),sha256=sha256(full))
review['compact_recipe']=dict(path=str(Path(__file__)),sha256=sha256(Path(__file__)))
path.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')

fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
styles={
    'patch_+30_9':('Longitude patch (10:1)', '#bc8d1b','s'),
    'upper_LOS_+5':('Upper LOS +5 km/s','#2574b8','^'),
    'upper_radial_add_+13':('Upper radial +13 km/s','#c54247','v'),
    'all_data_longitude_fit':('All-data longitude fit','#37875c','o')}
ax=axes[0]
for name,(label,color,marker) in styles.items():
    for selection,alpha,face in [('primary',1,color),('alternate',.7,'none')]:
        result=review['cases'][name]['selections'][selection]['fixed']
        x,y=[result['target']['all'][b]['change_in_planet_plus_GP_ppm'] for b in ['blue','core']]
        ax.scatter(x,y,color=color,facecolor=face,marker=marker,s=75,alpha=alpha,label=label if selection=='primary' else None)
        if selection=='primary': ax.plot([0,x],[0,y],color=color,alpha=.4,lw=1)
for selection,marker in [('primary','*'),('alternate','x')]:
    target=review['cases']['baseline']['selections'][selection]['fixed']['target']['all']
    ax.scatter(target['blue']['residual_ppm'],target['core']['residual_ppm'],color='black',marker=marker,s=120,
        label='Required change (primary / alternate)' if selection=='primary' else None)
ax.axhline(0,color='.7',lw=.7);ax.axvline(0,color='.7',lw=.7)
ax.set(xlabel='Change in blue prediction (ppm)',ylabel='Change in core prediction (ppm)',
    title='Target phase 0.376–0.393; fixed original RV/A')
ax.text(.02,.03,'Filled: primary pixels; open: alternate\nPrediction includes the conditional GP.',transform=ax.transAxes,fontsize=8)
ax.legend(fontsize=8,loc='upper right',framealpha=.85)
ax=axes[1]
phase=np.array([.3125,.3375,.3625,.3875,.4125,.43])
for name,(label,color,marker) in styles.items():
    data=np.array(review['cases'][name]['selections']['primary']['fixed']['phase']['common_complete']['model_plus_GP_change_BCR_ppm'],dtype=float)
    ax.plot(phase,data[:,0],color=color,marker=marker,label=label)
    ax.plot(phase,data[:,1],color=color,marker=marker,ls='--',alpha=.85)
ax.axhline(0,color='.7',lw=.7)
ax.axvspan(.375,.40,color='.5',alpha=.09)
ax.set(xlabel='Orbital phase bin',ylabel='Change in prediction (ppm)',
    title='Changes also occur outside the target',xlim=(.30,.438))
ax.text(.03,.19,'Solid: blue; dashed: core\nBlank late phases lack complete common B/C/R support.',transform=ax.transAxes,fontsize=8,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
for ext in ['pdf','png']:
    fig.savefig(OUTPUT/f'kinematic_directions.{ext}',dpi=180)
(OUTPUT/'figure_receipt.json').write_text(json.dumps(dict(
    inputs={str(p):sha256(p) for p in [Path(__file__),path]},
    outputs={name:sha256(OUTPUT/name) for name in ['kinematic_directions.pdf','kinematic_directions.png']},
    scope='Finite responses at fixed original RV/amplitude; no uncertainty or causal-identification claim. Both interleaves on the left; primary on the right. Unsupported phase values are blank.'),indent=2)+'\n')
