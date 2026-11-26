# Coupled H-alpha equilibrium and observation comparison

The implementation revision is `e9fcb7aee891a12de145c1bda75ed139a0c2718b`.
The accepted solution is the canonical CPU export using driver `b74aa0a767f9`.
Its complete 8,840-component residual is `1.458557719e-10`, the fresh-repeat
difference is zero, and the largest independently recomputed conservation
error is `4.44e-16`. The exporter accepted zero coordinate updates; its
coordinates exactly equal both the GPU solution and the independent CPU
crosscheck. See [accepted_root_review.json](accepted_root_review.json).

GPU execution accelerated the same equations. The archived driver
`3962d8cc4323` produced the numerical solution and passed the independent
CPU check: the maximum CPU–GPU residual difference was `1.456612608e-13`.
That GPU process exited with a completion-logging exception after writing
its verified results. A one-line logging repair changed no numerical-function
AST. The current-source CPU exporter then completed normally.
[backend_final_audit.json](backend_final_audit.json) preserves this distinction
and links the archived source; [cpu_crosscheck.json](cpu_crosscheck.json)
retains its original source pins.

The same-processing, fixed-GP comparison worsens at this fixed mass loss:

| Primary comparison | Baseline | Coupled atmosphere |
|---|---:|---:|
| Q | 38326.64894 | 38392.63176 |
| Blue residual, ppm | +409.06 | +594.61 |
| Core residual, ppm | -235.24 | -529.90 |

The complete-equilibrium claim is supported. The two conditional improvement
claims—lower Q and smaller absolute blue/core residuals—are contradicted.
Complementary pixels and fixed baseline nuisance parameters agree in direction.
This is not a significance estimate or a rejection of every parameter choice.
The MCEG reader intentionally returns nonzero for these contradicted improvement
assessments; their structural, source and artifact checks pass.
The [independent observation audit](observation_assessment_audit.json) links
the [final comparison](../halpha_geometry_fit_20261126/comparison.json).

The [quadrature controls](quadrature_review.json) hold populations and
temperature fixed. Angular/frequency refinements change effective rates by
less than `8e-8`; doubled wings change upward/downward rates by at most
`3.29e-5`/`1.58e-4`. The [heat interpretation](quadrature_interpretation.json)
covers H, free-free and H-alpha redistribution only. These are not a second
atmospheric root or a complete refined thermal residual.

Compact accepted artifacts are [final_state.npz](final_state.npz),
[observer_inputs.npz](observer_inputs.npz), [recipe.json](recipe.json),
[final_configuration.json](final_configuration.json), and
[final_summary.json](final_summary.json). Large physical/inverse files remain
external with recorded hashes. [solver_audit.json](solver_audit.json) records
the current tests. Earlier failed inverse, cache, memory and GPU-library
attempts are numerical diagnostics, not physical-model counterexamples.
Historical frozen receipts remain unchanged. The separate
[evaluation report](../../../doc/halpha_equilibrium.pdf) summarizes the result.
