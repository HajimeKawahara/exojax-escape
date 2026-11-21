from pathlib import Path
import json,hashlib,itertools
import numpy as np
from scipy.optimize import brentq
root=Path('/tmp/cloudy-causal-rates')
protocol=Path('/tmp/exojax-escape-coupled/mceg/investigations/cloudy_discrepancy.json')
keys=['radiative_bound_bound_s','electron_bound_bound_cm3_s','photoionization_s','radiative_recombination_cm3_s','electron_ionization_cm3_s','three_body_recombination_cm6_s']
def parse(name):
 blocks=[]
 for line in (root/name/(name+'.out')).read_text().splitlines():
  a=line.split()
  if not a:continue
  if a[0]=='MCEG_BEGIN': b={'header':list(map(float,a[1:])), 'fb':[], 'bb':[], 'matrix':[], 'two_photon':[]}
  elif a[0]=='MCEG_FB':b['fb'].append(list(map(float,a[1:])))
  elif a[0]=='MCEG_BB':b['bb'].append(list(map(float,a[1:])))
  elif a[0]=='MCEG_MATRIX':b['matrix'].append(list(map(float,a[2:])))
  elif a[0]=='MCEG_2PH':b['two_photon'].append(list(map(float,a[1:])))
  elif a[0]=='MCEG_SOLUTION':b['solution']=list(map(float,a[1:]))
  elif a[0]=='MCEG_END':blocks.append(b)
 b=blocks[-1];b['number_of_readouts']=len(blocks)
 (root/(name+'_operator.json')).write_text(json.dumps(b,indent=2)+'\n')
 return b

def from_operator(b):
 header=np.array(b['header']);temp,ne,corr,n0,np_=header[3:]
 fb=np.array(b['fb']);size=len(fb);g,E=fb[:,3],fb[:,4]
 rad=np.zeros((size,size));coll=rad.copy()
 for row in b['bb']:
  u,l=map(int,row[:2]);A,loss,pump,Cul,gap=row[2:7]
  rad[u,l]=A*loss+pump*g[l]/g[u];rad[l,u]=pump
  coll[u,l]=Cul/ne;coll[l,u]=coll[u,l]*g[u]/g[l]*np.exp(-(E[u]-E[l])/temp)
 for u,l,A,up,dn in b['two_photon']:
  u,l=int(u),int(l);rad[u,l]+=A+dn;rad[l,u]+=up
 ion=fb[:,6]*corr/ne;three=ion*fb[:,7]
 alpha=fb[:,8]*fb[:,9]+fb[:,10]*fb[:,7]
 return dict(zip(keys,[rad,coll,fb[:,5],alpha,ion,three]))

def matrix(r,ne):
 trans=r[keys[0]]+ne*r[keys[1]]
 ion=r[keys[2]]+ne*r[keys[4]]
 A=np.diag(trans.sum(1)+ion)-trans.T
 s=ne*r[keys[3]]+ne**2*r[keys[5]]
 return A,s

def solve(r,ne=None):
 if ne is None:
  def residual(logne):
   n=np.exp(logne);A,s=matrix(r,n)
   return n*(1+np.linalg.solve(A,s).sum())/1e12-1
  ne=np.exp(brentq(residual,np.log(1e6),np.log(1e12),xtol=1e-13))
 A,s=matrix(r,ne);p=ne*np.linalg.solve(A,s)
 return {'ne':ne,'bound_cm3':p.tolist(),'n2':float(p[1:3].sum()),'n3':float(p[3:6].sum())}

cloud=parse('baseline5');cloff=parse('no_lmix5')
cr=from_operator(cloud);off=from_operator(cloff)
ours={k:np.array(v) for k,v in json.loads(Path('/tmp/cloudy-reference-run/escape_reference.json').read_text())['models']['5']['rates'].items()}
ne=cloud['header'][4];A,s=matrix(cr,ne)
result={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'cloudy_operator':cloud,'operator_reconstruction':{'matrix_relative_max':float(np.max(np.abs(A-np.array(cloud['matrix'])))/np.max(np.abs(A))),'source_relative_max':float(np.max(np.abs(s*cloud['header'][-1]-np.array(cloud['fb'])[:,-1]))/np.max(np.array(cloud['fb'])[:,-1])),'fixed_ne_independent_numpy_solution_relative_max':float(np.max(np.abs(np.linalg.solve(A,s)*cloud['header'][-1]/np.array(cloud['solution'])-1)))},'results':{},'cloudy_rates':{k:v.tolist() for k,v in cr.items()},'escape_rates':{k:v.tolist() for k,v in ours.items()}}
for mode,neval in [('fixed_ne',ne),('self_consistent_ne',None)]:
 vals={}
 vals['escape']=solve(ours,neval);vals['cloudy_full_operator']=solve(cr,neval)
 for name in ['n3_mixing','n2_mixing','all_mixing','bound_free','ionization','bb_collision','bb_radiation','all_continuum','nonmixing_collision','n3_mixing_and_bound_free','n3_mixing_and_ionization','n3_mixing_and_bf_and_ionization']:
  r={k:v.copy() for k,v in ours.items()}
  if 'n3_mixing' in name or name=='all_mixing':r[keys[1]][3:6,3:6]=cr[keys[1]][3:6,3:6]
  if name in ['n2_mixing','all_mixing']:r[keys[1]][1:3,1:3]=cr[keys[1]][1:3,1:3]
  if name in ['bound_free','all_continuum'] or 'bound_free' in name or '_bf_' in name:
   for k in keys[2:4]:r[k]=cr[k].copy()
  if name in ['ionization','all_continuum'] or '_ionization' in name:
   for k in keys[4:]:r[k]=cr[k].copy()
  if name=='bb_collision':r[keys[1]]=cr[keys[1]].copy()
  if name=='nonmixing_collision':
   r[keys[1]]=cr[keys[1]].copy();r[keys[1]][1:3,1:3]=ours[keys[1]][1:3,1:3];r[keys[1]][3:6,3:6]=ours[keys[1]][3:6,3:6]
  if name=='bb_radiation':r[keys[0]]=cr[keys[0]].copy()
  vals[name]=solve(r,neval)
 r={k:v.copy() for k,v in cr.items()};r[keys[1]][3:6,3:6]=ours[keys[1]][3:6,3:6]
 vals['cloudy_remove_n3_mixing']=solve(r,neval)
 result['results'][mode]=vals
result['cloudy_native_no_lmix']=dict(n2=float(np.array(cloff['solution'])[1:3].sum()),n3=float(np.array(cloff['solution'])[3:6].sum()),bound_cm3=cloff['solution'],ne=cloff['header'][4])
(root/'causal_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['operator_reconstruction'])
for mode, vals in result['results'].items():
 print(mode)
 for k,v in vals.items():print(k, f"n2={v['n2']:.8g} n3={v['n3']:.8g} ne={v['ne']:.8g}")
print('native no mixing',result['cloudy_native_no_lmix'])
