# Coupled native thermal root and observer

The selected five-cell native model now closes temperature, radiation,
populations, electrons/composition/molecular weight and the tidal Parker wind.
Its original F-infinity criterion is unchanged: 8.42780068e-10 < 1e-9.
Only the three upper cells solve energy balance; the two dense lower cells
retain the declared imposed temperature/LTE extension. Physical, domain and
wind checks pass. Six accepted thermal updates and exact-input receipts are
retained. This is a discrete numerical root, not a resolved atmosphere or
verified substitute for the paper's Cloudy/sunbather calculation.

At the solved thermal profile, expansion cooling agrees with an independent
adaptive ODE within 0.240%. Native grid/domain checks retain their failed
64-Rp/1025 case as well as the passing 32-Rp/1025 comparison. Nearly exact
outer-domain invariance on this five-cell profile applies only to its inner
subsonic sample points; it is not a test of newly resolved outer cells.

The two displayed figures use the actual native root and continuum. The
processed spectrum does not describe the observed central trough and peak
amplitudes. No fit is claimed. Atmospheric resolution, atomic/transfer closure
adequacy, complete production gradients and the observational reduction remain
separate checks. The observer archive also contains fixed-temperature controls
and the limited joint angular/frequency/phase refinement, with explicit scope.

The observer uses the Zenodo data under CC BY 4.0: Zhang, Yapeng (2026),
doi:10.5281/zenodo.20707255. It implements an approximate cadence/master/mask
recipe, not the exact unpublished reduction. The archive includes attribution
and the original license record. Figure rendering was inspected.

All source and input hashes are pinned. Restore escape
4473ed7f2344e8edba012477e05c34b6c7da56a2 and ExoJAX
38cb2e44f153ad6d729e3253e235702de52920cf. Replay instructions are inside each
archive. Atomic data are restored from source rather than copied repeatedly.
No raw PHOENIX/FITS/SED, sampled stellar boundary/reference, executable,
compiler IR/cache or Krylov matrices is published. The archive audit lists
every included NPZ field. This evidence-only commit does not replace PR code.
