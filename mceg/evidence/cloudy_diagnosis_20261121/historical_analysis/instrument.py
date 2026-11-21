from pathlib import Path
import difflib
p=Path('/tmp/cloudy-causal-rates/source/iso_level.cpp')
s=p.read_text(); original=s
anchor='\t\tnerror = 0;\n\n\t\tgetrf_wrapper'
assert s.count(anchor)==1
block=r'''
        // Investigation-only readout: retain the actual local H SE operator.
        // No coefficient, population, or convergence condition is changed.
        if( ipISO == ipH_LIKE && nelem == ipHYDROGEN )
        {
            fprintf(ioQQQ, "MCEG_BEGIN %ld %ld %ld %.17e %.17e %.17e %.17e %.17e\n",
                iteration, nzone, numlevels_local, phycon.te, dense.eden,
                dense.EdenHCorr, dense.xIonDense[ipHYDROGEN][0], dense.xIonDense[ipHYDROGEN][1]);
            for(long a=0; a<numlevels_local; ++a)
            {
                fprintf(ioQQQ, "MCEG_FB %ld %ld %ld %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
                    a, sp->st[a].n(), sp->st[a].l(), double(sp->st[a].g()), sp->st[a].energy().K(),
                    sp->fb[a].gamnc, sp->fb[a].ColIoniz, sp->fb[a].PopLTE,
                    sp->fb[a].RadRecomb[ipRecRad], sp->fb[a].RadRecomb[ipRecNetEsc],
                    sp->fb[a].RecomInducRate, sp->fb[a].RateLevel2Cont, sp->fb[a].RateCont2Level,
                    Save_creation[a]);
                fprintf(ioQQQ,"MCEG_MATRIX %ld",a);
                for(long b=0;b<numlevels_local;++b) fprintf(ioQQQ," %.17e",SaveZ[b][a]);
                fprintf(ioQQQ,"\n");
                for(long b=0;b<a;++b)
                {
                    fprintf(ioQQQ,"MCEG_BB %ld %ld %.17e %.17e %.17e %.17e %.17e",
                        a,b,double(sp->trans(a,b).Emis().Aul()),sp->trans(a,b).Emis().Ploss(),
                        sp->trans(a,b).Emis().pump(),sp->trans(a,b).Coll().ColUL(colld),sp->trans(a,b).EnergyK());
                    for(long c=0;c<ipNCOLLIDER;++c) fprintf(ioQQQ," %.17e",sp->trans(a,b).Coll().rate_coef_ul()[c]);
                    fprintf(ioQQQ,"\n");
                }
            }
            for(vector<two_photon>::iterator tnu=sp->TwoNu.begin();tnu!=sp->TwoNu.end();++tnu)
                fprintf(ioQQQ,"MCEG_2PH %ld %ld %.17e %.17e %.17e\n",tnu->ipHi,tnu->ipLo,
                    tnu->AulTotal,iso_ctrl.lgInd2nu_On?tnu->induc_up:0.,iso_ctrl.lgInd2nu_On?tnu->induc_dn:0.);
        }
'''
s=s.replace(anchor,block+anchor)
anchor2='\t\tif( sp->lgImgMatrix && save.img_matrix.matchIteration( iteration ) &&\n\t\t\t\tsave.img_matrix.matchZone( nzone ) )\n\t\t{\n\t\t\tvalarray<double> SavePops'
assert s.count(anchor2)==1
post=r'''
        if( ipISO == ipH_LIKE && nelem == ipHYDROGEN )
        {
            fprintf(ioQQQ,"MCEG_SOLUTION");
            for(long a=0;a<numlevels_local;++a) fprintf(ioQQQ," %.17e",creation[a]);
            fprintf(ioQQQ,"\nMCEG_END\n");
        }
'''
s=s.replace(anchor2,post+anchor2)
p.write_text(s)
Path('/tmp/cloudy-causal-rates/rate_readout.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original/source/iso_level.cpp',tofile='instrumented/source/iso_level.cpp')))
