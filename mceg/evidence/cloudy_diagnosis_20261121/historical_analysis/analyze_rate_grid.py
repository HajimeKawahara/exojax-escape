"""Extract saved Cloudy rates and compare the continuous PSM20 adaptation."""
from pathlib import Path
import argparse,hashlib,json,sys
import jax
jax.config.update('jax_enable_x64',True)
from escape.hydrogen_mixing import proton_l_mixing_psm20
from psm_prototype import psm


def analyze(root):
    rows=[]
    for path in sorted(root.glob('rates_T*_n*/*.out')):
        header=None;bb=[]
        for line in path.read_text().splitlines():
            f=line.split()
            if not f:continue
            if f[0]=='MCEG_BEGIN':header=list(map(float,f[1:]));bb=[]
            elif f[0]=='MCEG_BB':bb.append(list(map(float,f[1:])))
        temperature,ne=header[3:5]
        for upper,lower,ell in [(4,3,1),(5,4,2)]:
            readout=next(row for row in bb if row[:2]==[upper,lower])
            tau=1/sum(row[2] for row in bb if row[0]==upper)
            q_cloudy=readout[8]
            exact,combined,uc=psm(temperature,ne,ell,ell-1,tau)
            q_runtime=float(proton_l_mixing_psm20(temperature,ne,3,ell,ell-1,tau))
            rows.append(dict(run=path.parent.name,T_K=temperature,ne_cm3=ne,upper=upper,lower=lower,lifetime_s=tau,cloudy_q_cm3_s=q_cloudy,cloudy_convention_q_cm3_s=exact,continuous_cloudy_mass_q_cm3_s=combined,runtime_q_cm3_s=q_runtime,Uc=uc,cloudy_convention_relative_error=exact/q_cloudy-1,runtime_relative_difference=q_runtime/q_cloudy-1,raw_output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    rows=analyze(args.root)
    result={'source':'Cloudy C23.01 CS_l_mixing_PS64_expI plus read-only operator readout', 'interpretation':'The diagnostic Cloudy-convention function retains its thermal-cutoff shortcut. The runtime continuously integrates both cutoffs with physical H-1/proton reduced mass. Differences from the shortcut are intentional, not errors fitted to Cloudy.', 'rows':rows, 'max_cloudy_convention_relative_error':max(abs(row['cloudy_convention_relative_error']) for row in rows),'max_runtime_relative_difference':max(abs(row['runtime_relative_difference']) for row in rows)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('rows',len(rows));print('max convention error',result['max_cloudy_convention_relative_error']);print('max intentional runtime difference',result['max_runtime_relative_difference'])
    for row in rows: print(row['run'],row['upper'],row['ne_cm3'],row['Uc'],row['cloudy_convention_relative_error'],row['runtime_relative_difference'])
