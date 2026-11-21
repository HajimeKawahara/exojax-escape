import numpy as np
from scipy.special import exp1,erf,gammainc
A0=5.29177210903e-9;T0=2.41889e-17;TE1RYD=157887.512402;ME=9.1093837015e-28
# Cloudy uses the standard hydrogen atomic weight (1.00794u), including natural isotopes.
M_U=1.66053906660e-24;M_H_U=1.00794;M_P_U=M_H_U-5.48579909065e-4
MRED=M_H_U*M_P_U/(M_H_U+M_P_U)*M_U

def psm(temp,ne,l,lp,tau):
 n=3; vel=np.sqrt(temp/TE1RYD/(MRED/ME));strength=9*n*n*max(l,lp)*(n*n-max(l,lp)**2)/2
 R12=2/(3*.5*vel**2*(2*l+1))*strength*2
 RD2=(temp/TE1RYD)/(8*np.pi*ne*A0**3);RC12=(.72*tau/T0*vel)**2
 um=R12/RD2;uc=RD2/RC12;u=np.sqrt(um*uc)
 nf=np.sqrt(np.pi/um**3)*erf(np.sqrt(um))/2-np.exp(-um)/um if um>1e-6 else 2/3
 if uc>=u:combined=4*gammainc(3,u)/u**3+2*exp1(u)-exp1(uc)
 else:combined=4*(gammainc(3,uc)+.25*np.exp(-uc)*uc**2)/u**3-np.sqrt(np.pi/um**3)*erf(np.sqrt(uc))/2+nf+exp1(um)
 debye=nf+exp1(um)
 scale=2*A0**3*np.sqrt(np.pi)/vel/T0*2*strength/(3*(2*l+1))
 return scale*(debye if uc<=1 else combined),scale*combined,uc
if __name__=='__main__':
 tau=1/(1.6724736e8+2.2448808e7)
 for ne in [1e8,1e9,1e10,8.215510376e11]:print(ne,psm(9000,ne,1,0,tau))
 critical=(9000/TE1RYD)/(8*np.pi*A0**3*(.72*tau/T0*np.sqrt(9000/TE1RYD/(MRED/ME)))**2)
 print('criticalne',critical)
 for f in [.999999,1,1.000001]:print(f,psm(9000,critical*f,1,0,tau))
