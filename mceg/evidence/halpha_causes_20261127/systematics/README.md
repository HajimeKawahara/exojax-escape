# Stellar and calibration controls

The tested smooth stellar and calibration families do not explain the selected
phase 0.37615–0.39322 blue/core residual. These are conditional experiments with
the existing physical spectrum, native masks, full-data reference, fixed planet
RV/amplitude, and archived GP covariance. They do not establish an atmospheric
origin or identify a stellar oscillation.

The broad PHOENIX width drift is the most effective full-data control: maximum
fractional width changes are 0.64% and 0.97% in the two nights. Its exact finite
evaluation improves Q by 22.18 but reduces the target blue-minus-core residual
by only 76.4 ppm, from 1176.0 to 1099.5 ppm. The exact remaining B/C/R residuals
are (+551, −548, −185) ppm. Depth modulation at a fixed 7.58695-hour period and
smooth throughput curvature have much smaller contrast responses.

A target-only width drift can produce the required sign pattern, but requires
an 8.45% width modulation. Its exact finite full-data Q becomes worse by 1471;
the linearized approximation also becomes inaccurate. The other target-only
depth, shift, and curvature screens likewise worsen the full-data score.
These target-conditioned amplitudes are flexibility checks, not inferred
physical parameters. Width drift was selected for the full-data fit after
inspecting that screen, so its improvement is exploratory.

The receipt includes complementary-pixel predictions and fits withholding the
target phase, planet stripe, eclipse, or one order. A modest width component
survives these checks; a depth-pulsation fit trained without eclipse predicts
the omitted eclipse poorly. Noise hyperparameters used the full observations,
so these are conditional score checks rather than independent validation or
Bayes factors. Adjacent pixels and orders are not claimed independent.

`review.json` records all coefficients, exact/linear comparisons, source/input
hashes, and unsupported complete bands as null. `verification.json` records the
final hash and numerical checks. `systematics.png` and `.pdf` show the responses;
the held-out score panel uses a symmetric logarithmic axis to retain the large
negative eclipse result. Its target and phase curves use the linear fits;
exact full-fit target differences are at most 1.65 ppm.

The native log-model reconstruction agrees to 7.22e−10 ppm, the complete-band
baseline to 1.32e−10 ppm, and direct production log-likelihood improvements to
1.38e−11 in Q. The PHOENIX depth/width morphs are proxies around the same uniformly
rotating reference, not time-dependent stellar-atmosphere calculations. Their
failure does not rule out nonradial line-profile changes, line-spread-function
errors, telluric contamination, or unknown extraction covariance.

Replay from the repository (the input cache paths are recorded in the receipt):

```sh
taskset -c 2-9 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=True \
  XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1 \
  MPLCONFIGDIR=/tmp/escape-halpha-causes-systematics-mpl \
  PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src \
  python examples/halpha_cause_systematics.py \
  --output mceg/evidence/halpha_causes_20261127/systematics
```
