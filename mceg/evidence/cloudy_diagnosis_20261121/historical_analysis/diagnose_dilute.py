from pathlib import Path
import json,hashlib,sys
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
import numpy as np
from cloudy_rates_diagnosis import parse_operator,rate_blocks,solve_frozen,interventions,RATE_KEYS
from escape.hydrogen_atom import make_hydrogen_atom,atom_rate_coefficients
from escape.hydrogen_rates import EV_CGS,PLANCK_CGS,planck_occupation
from escape.hydrogen_mixing import proton_l_mixing_psm20
root=Path('/tmp/cloudy-causal-rates');raw=root/'rates_T9000_n10/rates_T9000_n10.out';operator=parse_operator(raw);cr=rate_blocks(operator);ne=operator['header'][4]
atom=make_hydrogen_atom(5)
edges=np.unique(np.r_[np.array(atom.ionization_energy_erg)/PLANCK_CGS,100*EV_CGS/PLANCK_CGS]);x,w=np.polynomial.legendre.leggauss(48);wid=np.diff(np.log(edges));nu=edges[:-1,None]*np.exp(wid[:,None]*(x+1)/2);weights=j.asarray((nu*wid[:,None]*w/2).ravel());nu=j.asarray(nu.ravel())
r=atom_rate_coefficients(atom,9000.,planck_occupation(atom.line_frequency_hz,10000.,.025),planck_occupation(nu,10000.,.025),nu,weights,electron_cm3=ne)
ours={k:np.array(v)for k,v in zip(RATE_KEYS,r)}
cases=interventions(ours,cr)
reverse={k:v.copy()for k,v in cr.items()};reverse[RATE_KEYS[1]][1:3,1:3]=ours[RATE_KEYS[1]][1:3,1:3];cases['cloudy_restore_legacy_n2']=reverse
results={mode:{k:solve_frozen(v,1e10,n)for k,v in cases.items()}for mode,n in [('fixed_ne',ne),('frozen_coefficients_charge_closure',None)]}
q=proton_l_mixing_psm20(9000.,ne,2,1,0,1/6.26489344e8)
result=dict(raw_operator_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),cloudy_operator=operator,repaired_rates={k:v.tolist()for k,v in ours.items()},cloudy_rates={k:v.tolist()for k,v in cr.items()},results=results,continuous_psm20_n2_q=float(q))
(root/'dilute_diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('ne',ne,'PSM n2',q,'cloudy n2',cr[RATE_KEYS[1]][2,1],flush=True)
for mode,vals in results.items():
 print(mode)
 for k,v in vals.items():print(k,v['n2_cm3'],v['n3_cm3'])
 a=vals['legacy']['n2_cm3'];b=vals['cloudy_full_operator']['n2_cm3'];c=vals['n2_mixing']['n2_cm3'];d=vals['cloudy_restore_legacy_n2']['n2_cm3'];print('explained',(a-c)/(a-b),'reverse',(d-b)/(a-b),flush=True)
