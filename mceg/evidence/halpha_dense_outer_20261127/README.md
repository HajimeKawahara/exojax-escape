# Prescribed dense outer atmosphere

This campaign executes the conditional model proposed by the outer-reservoir
investigation. The complete native composition, radiation and thermal
residual is re-solved. The additional force supporting the modified flow is
prescribed and reported, not derived from a magnetic equilibrium.

Seventeen conditional roots, D=1 through D=20, pass exact-coordinate CPU
verification. Eight states (including D1) were processed through the matched
native observer. None of the seven higher-density cases improves the target
blue-minus-core residual in either profiled pixel selection. At D20 the
cell-5 n2 population is 332 times D1, but the source is 3.41 times D1;
the primary blue-minus-core residual worsens from 1701.60 to 1741.67 ppm.
The final report is `doc/halpha_dense_outer.pdf`.

The density multiplier is relative to the reference wind recomputed at each
candidate temperature and composition. It is unity below 1.3 Rp and reaches
D above 1.725 Rp through a smooth logarithmic ramp. Dividing velocity by the
same multiplier preserves mass flux. The exact changed velocity gradient
also enters expansion/compression heating. Actual densities relative to the
original atmosphere must therefore be reported separately from D.

- `protocol.json` declares the experiment and unchanged numerical gates.
- `review/` independently checks continuity, force and thermal identities,
  and records radial populations, opacity/source and geometry limitations.
- `observer/` compares accepted exported states at the same native samples,
  fixed covariance and complete-band target, using both pixel selections.
  D1 retains the previous spherical solution as a density-free control.
- The final CPU crosscheck verifies exact candidate coordinates without
  silently polishing them. Unavailable or failed stages remain explicit.
  GPU receipts keep their original pending-CPU flag; separate CPU receipts
  establish final qualification without rewriting execution history.

The experimental drivers are `examples/halpha_dense_outer.py`,
`examples/halpha_dense_crosscheck.py` and `examples/halpha_dense_observe.py`.
Production source modules are unchanged. External large checkpoints and
full radiation dictionaries retain their paths/hashes in the execution
receipts; compact evaluated states and review results accompany the report.

For a fresh replay, set `PYTHONPATH=src:examples:/tmp/exojax-spectrum-upstream/src`,
enable JAX x64, and retain the pinned external recipe, stellar SED, atomic
tables and baseline checkpoint. The experiment uses the existing dependency
snapshot; `environment.json` records the actual imports. A fresh GPU run is:

```sh
JAX_PLATFORMS=cuda JAX_ENABLE_X64=True \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
XLA_FLAGS='--xla_cpu_parallel_codegen_split_count=1 --xla_gpu_enable_cublaslt=false' \
python examples/halpha_dense_outer.py --output /tmp/halpha-dense-replay \
  --factors 1 1.2 1.3 1.4 1.5 1.75 2 3 5 6 7 8 9 10 12 15 20 \
  --secant-predictor
```

Run `examples/halpha_dense_crosscheck.py` with `JAX_PLATFORMS=cpu`, the
successful stage directories supplied to `--candidate`, a new `--output`
directory and `--export-full`. It re-evaluates the saved coordinates rather
than solving again. `examples/halpha_dense_observe.py --help` documents the
four state/review/raw-output/compact-output arguments for matched rendering.
The observer's `qualify_case.py` attaches the corresponding CPU receipt;
`collect_comparison.py` aggregates both baselines and nuisance choices.

The recorded execution used an initial attempt followed by smaller density
steps and a guarded secant initial guess. `attempts/history.json` and
`numerical_restart.json` distinguish import/export failures, the interrupted
unconverged point and accepted roots. The current predictor does not change
the physical equations or acceptance thresholds. Historical source snapshots
retain their original hashes. `collect.py` archives exact coordinates and
compact fields; `validate.py` verifies their identity. Large inverse
accelerators are omitted from the compact archive.

CPU attempt 05 retained a failed near-zero reference-force comparison at D8.
Attempt 06 uses a field-specific cancellation-aware floor, with the full
residual, thermal and conservation criteria unchanged. The D8 numerical
arrays are byte-identical across these checks; a deliberately perturbed
force still fails. The historical failure is preserved and excluded from
accepted receipts, rather than rewritten as a successful run.

Local statistical equilibrium, the native kinetic thermal closure, the
eight-cell radial grid and static CRD H-alpha population transfer remain
approximations. Neither a converged conditional root nor a smaller native
score establishes a magnetic configuration or a calibrated detection.
