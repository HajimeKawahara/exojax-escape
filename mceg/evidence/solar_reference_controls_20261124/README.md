# Native fixed-temperature and independent reference controls

These completed controls answer different questions. They do not establish a
converged solar thermal atmosphere, reference equivalence, a resolved observed
spectrum or inference readiness. The full implementation loop continues.

| Control | Result | Scientific scope |
| --- | --- | --- |
| Native five-cell radiation/composition/wind, ten retained metals, Fe II 716 and S II 72, H through n=15, explicit H/O and H/Fe coupling | Five Newton updates reach F_inf=1.13672e-10 against the unchanged 1e-9 criterion; physical/domain/wind checks and a coarse observer pass | Temperature is prescribed; no thermal equilibrium or resolution claim |
| Native 129-node wind against an independent adaptive ODE | Initial five-cell expansion cooling differs by 0.6553%; velocity by 0.0523% | Supports this prescribed initial profile only; repeat at the solved thermal profile |
| Sharp, prescribed 1000-point T/mu profile | Native expansion errors remain 10.6--21.1% across 129--1025 nodes; the independent ODE refines to 0.00620% | Rejects inference of energy-gradient accuracy from velocity/collocation residual alone |
| Readout-only C23.01 one-zone blackbody Ca II control | Actual excited-level receiver opacity is 12.9294 times the total opacity used for the common OTS field; the legacy opacity population is zero | A local reference-allocation counterexample, not the cause or magnitude of a planetary thermal/spectral difference |

The Ca diagnostic leaves all rates and opacities unchanged. Both requested
physical output tables are byte-identical to its unmodified companion.
The receiver cross sections are the four original C23 Ly-alpha values.
A conservative induced-recombination bound is 2.693e-9 of their forward
absorption, so it cannot explain the allocation discrepancy in this state.
No broad-continuum cross-section shape is inferred. Attribution of planetary
cooling differences and a physically coupled correction remain separate work.

Each archive supplies exact inputs, source pins, completed execution receipts,
restricted conclusions and replay instructions. The native archive preserves
its initial diagnostic-script failure separately; the corrected physical
diagnostic was executed. Its portable wrapper passed hash checks and shares
the executed evaluation AST, but a second expensive physics replay was not
performed merely to rename paths. Stellar data must be retrieved independently.

The ExoJAX export's 284 source/data files match Git revision
38cb2e44f153ad6d729e3253e235702de52920cf byte-for-byte. This provides an exact
reconstruction path, without inferring the history of how the export was made.
The native source/data are likewise restored by pinned Git revision, avoiding
duplicate atomic-data bundles. Earlier published archives remain unchanged.

No raw PHOENIX/FITS/SED arrays, sampled stellar boundary/reference arrays,
executables, objects, compiler IR, compilation cache or Krylov matrices are
published. Cloudy excerpts retain their license and attribution inside the
Ca archive. Figures for earlier controls remain in their original evidence
commits; this evidence-only commit does not replace the current PR source.
