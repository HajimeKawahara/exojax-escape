# Scoped thermal derivatives and combined photon accounting

The derivative archive retains the original failed sensitivity tolerance,
its paired accuracy correction, a smaller finite step, full thermal
implicit-JVP/independent-root agreement, and the original interrupted run.
Its full thermal spectral derivative error is 2.3939e-6 against 1e-4.
All four individual observer-grid refinements and their joint control
pass 0.1 supplied observation errors at the unchanged five-cell root.
These do not establish production reverse AD, atmospheric resolution,
reference equivalence or a NUTS parameter envelope.

The independent C23 combined Fe/Ca archive retains an initially inactive
one-cell diagnostic and a genuinely active three-cell blackbody control.
With direct Fe exchange retained and excluded from the generic OTS pool,
and actual excited-Ca opacity shared by H destruction and its receiver,
H loss equals generic OTS plus direct Fe to 2.22e-16 relative; the bin
source/receiver ledger closes to 4.47e-7. Tracking/one-switch paired
physical outputs are unchanged where required. Both approximations are
explicit reference controls, not native model changes. A complete solar
thermal atmosphere and numerical reference convergence remain separate.

All source/input hashes and failed controls are retained. Executed scripts
keep their original paths. The numerical certificate recheck validates
saved arrays, not an unexecuted physical replay. No original PHOENIX/FITS/
SED, stellar flux/reference arrays, executables, compiler caches or Krylov
matrices are published. Repeated atomic data are restored by source pins.
The derived observer retains Zhang (2026), doi:10.5281/zenodo.20707255,
CC BY 4.0 attribution. This immutable evidence is outside the PR file tree.
