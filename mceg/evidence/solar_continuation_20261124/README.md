# Scoped continuation controls, 2026-11-24

The five-cell prescribed-temperature native root closes radiation, composition
and the Parker wind at F_inf=2.1743e-12, with the original 1e-9 criterion. Its
coarse observer is finite. It uses the earlier nine-metal composition and
single-function continuous shielding/escape controls, not the new ten-metal
Fe-exchange candidate. It does not establish thermal equilibrium, spatial
resolution, observational processing or solar-reference adequacy.

The `src/escape` tree in this evidence commit is a byte-identical copy of the
historical five-cell source selected by its manifest. It is not the current
PR implementation. The archive replaces four duplicate atomic NPZ files with
an exact Git revision and hash-checked restore script. Lossless restoration
has been verified for all 41 source/data/license files; the original archive
SHA is retained. No stellar arrays are included. Its fetch/conversion recipe
and input hashes permit independent retrieval under the source terms.

The historical coarse-density finite differences substantially overestimated
expansion cooling. `flow_gradients.tar.gz` contains the paired diagnosis, the
repair at 4473ed7, independent spatial/parameter derivative tests and wind-grid
controls. The same regular velocity and T/mu interpolants now supply energy
derivatives; the solar path no longer differences raw sparse density samples.
At the recorded initial profile, 65 wind points miss the 1% cooling criterion
and 129 pass against 1025. The final thermal atmosphere must be refined again.
The chart holds the temperature/composition profile fixed, and does not depict
a solved thermal atmosphere. The plot script preserves its executed paths;
remap the archive locations for replay. Its plotted arrays are supplied.

`production_controls.tar.gz` contains the positive-iteration JAX Newton control
at 95ce166 and separate Fe integration receipts at c3378c7. The ten-metal,
two-cell analytic-blackbody JAX root takes two updates to F_inf=8.50875e-13,
matching the independent host root to 9.31e-13 in state coordinates. This is
not a cold-start atmosphere or full thermal/wind derivative verification.

`iron_lyalpha_helper.tar.gz` contains independent C23 primitive comparisons at
ce8f649: 210 width and 96 pumping cases plus six helper tests. The integration
checks cover balanced H/Fe photons and atomic energy, unchanged continuum OTS
at fixed state, and dense-LTE inactivity. Atmospheric/reference adequacy of
the empirical exchange is still unresolved.

`solver_causal_controls.tar.gz` preserves the Federman/K2 source-fit diagnosis
and the failed twelve-cell nine-metal continuation. Its best nonlinear
residual is 0.0014493, not converged. A later frozen-J inverse satisfies its
checked linear residual, which is not a nonlinear-root success. Work moved
to the sulfur/Fe candidate; the original failures remain recorded.

Every component has its own scope, source pins and raw receipts. These results
do not satisfy the six whole-atmosphere MCEG requirements. No executables,
compiler caches, raw PHOENIX/FITS/SED arrays or stellar-reference arrays are
included. Earlier evidence tags are unchanged.
