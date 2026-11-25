#include "cddefines.h"
#include "hydro_vs_rates.h"
#include "helike_cs.h"
#include "phycon.h"
#include "physconst.h"
int main() {
 printf("# constants %.17e %.17e %.17e %.17e %.17e %.17e\n",TE1RYD,BOHR_RADIUS_CM,FINE_STRUCTURE,COLL_CONST,SPEEDLIGHT,EN1RYD);
 for(double t : {4000.,4500.,5802.,8000.,12000.,14000.}) {
  phycon.te=t; phycon.sqrte=sqrt(t);
  for(long hi=6;hi<=20;++hi) for(long lo=1;lo<hi;++lo) {
   double ip=.9994663/(lo*lo);
   double cs=hydro_Lebedev_deexcit(0,0,hi,lo,ip);
   long last=lo<=3?lo:1;
   for(long l=0;l<last;++l) {
    double factor=lo<=3?CSresolver(0,hi,-1,2,lo,l,2,3):1.;
    printf("%.17e,%ld,%ld,%ld,%.17e,%.17e,%.17e\n",t,hi,lo,l,ip,cs,factor);
   }
  }
 }
}
