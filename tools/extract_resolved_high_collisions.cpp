#include "cddefines.h"
#include "hydro_vs_rates.h"
#include "helike_cs.h"
#include "phycon.h"
#include "physconst.h"
int main() {
 printf("# T,nu,lu,nl,ll,binding_lower_ryd,shell_strength,resolver,ionization_upper\n");
 for(double t : {4000.,8000.,14000.}) {
  phycon.te=t; phycon.sqrte=sqrt(t);
  for(long hi : {6L,10L,15L}) for(long lo : {4L,5L,6L,9L}) if(hi>lo) {
   double ip=13.5984345*1.602176634e-12/EN1RYD/(lo*lo);
   double cs=hydro_Lebedev_deexcit(0,0,hi,lo,ip);
   double ion=hydro_vs_ioniz(13.5984345*1.602176634e-12/EN1RYD/(hi*hi),t);
   for(long lh : {0L,1L,hi-1}) {
    if(hi>10 && lh!=0) continue;
    for(long ll : {0L,1L,lo-1}) {
    long upper_l=hi>10?-1:lh;
    double factor=CSresolver(0,hi,upper_l,2,lo,ll,2,10);
    printf("%.17e,%ld,%ld,%ld,%ld,%.17e,%.17e,%.17e,%.17e\n",t,hi,upper_l,lo,ll,ip,cs,factor,ion);
    }
   }
  }
 }
}
