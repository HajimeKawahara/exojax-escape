# Qualified reference and causal radiation controls

These controls separate numerical conventions, reference inconsistencies and
physical omissions. They do not certify a converged solar atmosphere.

| Control | Completed evidence | Scope |
| --- | --- | --- |
| C23 continuum transfer | OU2 and OU3 diagnostic runs preserve standard line caching and match all five physical tables of the unmodified binary byte for byte when built with the same compiler flags | Pure H, prescribed temperature/density; these are different continuum closures |
| Finite-cell source integration | In the standard OU3 fixed-coefficient zone-127 control, source-then-attenuation differs from the exact uniform-cell beam by 43.306%; 64 subcells reduce this to 0.8222% | Frozen operator, not a re-solved atmospheric convergence bound |
| Fe Ly-alpha allocation | Tracking-only results are identical; excluding the historically added Fe destruction from generic OTS removes the duplicate local source assignment while retaining direct Fe exchange | Three completed prescribed-temperature columns; both retain local and between-column convergence failures |
| Excited-Ca allocation | Adding actual receiver opacity to the OTS denominator closes the one-zone allocation to float32 accuracy | Allocation-only control; no change to H destruction |
| Coupled excited-Ca correction | The same opacity enters H destruction beta, actual albedo/electron-escape path and OTS allocation; source identities and local allocation checks pass | Completed one-zone blackbody control; the planetary thermal response remains separate |

The Fe exclusion changes the processed spectrum by only 0.06055 ppm, while
fixed-temperature heat channels change materially (about 15% of the original
thermal scale at 1.3 Rp). Small fixed-temperature spectral changes cannot bound
the temperature feedback. Both three-column cases retain their common local
failure and about 11% change in n3 between the final columns. No reference
agreement or globally conserved broadband photon budget is inferred.

The native Ca source audit is explicitly static. It locates a shared Ca opacity
in H loss and the Ca receiver/electron/heat feedback; the C23 zero legacy variable
is not copied by the native path. Its four cross sections are a Ly-alpha-only
prescription, not a verified broadband excited-Ca photoionization/Milne model.

The continuum archive preserves an earlier every-pass line-cache variant as
historical evidence, the inactive-option attempt caused by an input blank line,
and the O2/O3 compilation mismatch. Only the standard-cache, matched-flags runs
are labeled readout-identical to unmodified C23. It also explains the 1.35-times
old-opacity incident correction, effective RR/OTS elimination and heat-channel
grouping. No reference inconsistency is copied into the native default merely
to force agreement. Matched-closure diagnostics and converged combined Fe/Ca
controls continue separately.

Each archive includes exact input/source hashes, actual operator readouts,
original/modified controls, limitations and reproduction instructions. Original
Cloudy attribution and license remain with excerpts. Stellar inputs must be
retrieved independently. No original PHOENIX/FITS/SED, stellar flux/reference
arrays, executables, compiler IR/cache or Krylov matrices is included. A stellar
velocity coordinate in the derived Fe observer is not a stellar flux table.
The observational products retain Zhang (2026), doi:10.5281/zenodo.20707255,
CC BY 4.0 attribution. This evidence-only commit is separate from PR code.
