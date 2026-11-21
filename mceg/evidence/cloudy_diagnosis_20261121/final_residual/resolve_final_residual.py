from pathlib import Path
import json
import numpy as np
from cloudy_rates_diagnosis import solve_frozen,RATE_KEYS
root=Path('/tmp/cloudy-causal-rates');data=json.loads((root/'final_baseline_diagnosis.json').read_text());ours={k:np.array(v)for k,v in data['repaired_rates'].items()};cloud={k:np.array(v)for k,v in data['cloudy_rates'].items()};cases={}
for name in ['ground_bound_free','excited_bound_free','resolved_n_change_collisions','collapsed_n_change_collisions']:
 r={k:v.copy()for k,v in ours.items()}
 if name.endswith('bound_free'):
  sl=slice(0,1) if name.startswith('ground')else slice(1,None)
  for k in RATE_KEYS[2:4]:r[k][sl]=cloud[k][sl]
 else:
  mask=np.ones_like(r[RATE_KEYS[1]],dtype=bool);mask[:6,:6]=False
  if name.startswith('resolved'):mask=~mask
  mask[1:3,1:3]=False;mask[3:6,3:6]=False
  r[RATE_KEYS[1]][mask]=cloud[RATE_KEYS[1]][mask]
 cases[name]=r
result={mode:{k:solve_frozen(v,1e12,ne)for k,v in cases.items()}for mode,ne in [('fixed_ne',data['cloudy_operator']['header'][4]),('frozen_coefficients_charge_closure',None)]}
(root/'final_baseline_residual_blocks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for mode,vals in result.items():
 base=data['results'][mode]['legacy']['n3_cm3'];ref=data['results'][mode]['cloudy_full_operator']['n3_cm3'];print(mode)
 for k,v in vals.items():print(k,'remaining_percent',100*(v['n3_cm3']/ref-1),'change_percentage_points',100*(v['n3_cm3']-base)/ref)
