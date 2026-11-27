# Phase-band uncertainty

`review.json` records the executed generic covariance calculation in
`examples/halpha_phase_uncertainty.py`. The raw matrices and all 165 row
summaries remain under `/tmp/escape-halpha-phase-uncertainty/`, with hashes in
the receipt. Only combined rows and the late-core night/order breakdown are
expanded here. `contrasts.json`, produced by `extend_contrasts.py`, adds
within-phase blue-minus-core and matched-night core contrasts without
repeating the full calculation.

The phase edges are 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, and 0.435. Bands
are blue [-40,-16), core [-8,8), and red [16,40) km/s. Input maps retain the
full original common reference and GP. Their positive source weights were
independently recovered to check normalization, actual phase/band support,
and zero unsupported rows. No missing late-blue signal is extrapolated.

The primary covariance holds the original RV/amplitude fixed. A separate
local linear refitting sensitivity removes the two global nuisance
directions. Both calculations preserve covariance between phase bands. The
local covariance retains the fixed center; particularly on alternate pixels,
it is not the mean from a new full nuisance fit. Noise-draw means with actual
full-data nuisance refitting are supplied separately.

The generic operator reproduces the previous three-band calculation. For
each pixel selection, 256 Gaussian injections on actual support agree with
the explicit production GP pipeline to better than 8.1e-12 ppm; sample
variance/analytic variance ratios lie between 0.776 and 1.261. These checks
verify the numerical operator within its noise family, not observational
noise adequacy.

Time-block multipliers keep both orders and all wavelengths together, using
1, 2, 4, or 8 exposures and every boundary offset. The input is the already
fitted GP/common residual, which includes possible physical mean-model
errors and has also been reduced by fitting. Between-block independence,
few influential blocks, unknown extraction covariance, and many phase/order
comparisons prevent calibrated confidence claims. The two-SD proxy
envelopes are descriptive sensitivities.

The early core change illustrates the limitation: 0.325--0.35 to
0.35--0.375 changes by -1265 ppm on primary pixels but +42 ppm on alternate
pixels. The same-exposure blue-minus-core discrepancy in 0.375--0.40 is
1176/852 ppm, with actual complete support ending at phase 0.393223. Its
fixed-family standard deviation is 355/345 ppm, but empirical time-block
standard deviations range from 544--1052/386--901 ppm. All-offset two-SD
proxy envelopes include zero; this neither proves absence of structure nor
identifies the residual as noise.

Matching the late-core phase window on both nights gives night1-minus-night0
residuals of -936/-758 ppm. Fixed-family standard deviations are 438/439 ppm;
time-block ranges are 446--723/389--679 ppm. The sign persists across the
existing GP-family draws but not every two-SD block proxy. The original
night-specific common references and GP fits remain different, so this
conditional night difference is not a detection of atmospheric variability.

The main example and supplemental helper can be run with `PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src`,
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=True`, and single-thread BLAS. The executed
commands used cores 2--9, 64 archived noise draws, and 256 injections. No
atmospheric solve or production-code change is involved.
