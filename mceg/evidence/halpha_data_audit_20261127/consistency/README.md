# Native-data shape consistency

The processed data retain a positive blue band and a negative core without
subtracting any physical template. The combined feature survives interleaved
pixels, display-bin changes, removal of individual exposures, and removal of
contiguous exposure blocks. This establishes stability of a feature in the
archived, common-reference contrasts. It does not by itself establish an
intrinsic planetary profile or rule out upstream reduction effects.

The blue band is dominated by August 8. August 20 provides a weak independent
check of its detailed height: its central blue display bin has only three
exposures and six to nine native measurements per order. Late-phase residuals
also differ substantially from the early-phase residuals.

## Reproduction and definitions

```sh
taskset -c 10-17 env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/halpha-audit-mpl \
  PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src \
  python examples/halpha_data_consistency.py \
  --output mceg/evidence/halpha_data_audit_20261127/consistency \
  --raw-output /tmp/escape-halpha-data-consistency
```

The baseline native model, RV = 2.8601527471 km/s, amplitude = 1.1320524956,
posterior-median GP hyperparameters, and white scales are held fixed throughout.
The nominal Q is reproduced as 38326.64894003179. A NumPy implementation of the
same projected grey/slope GP enables the repeated selection checks; its four
conditional means match the existing JAX implementation to 2.3e-16 in log flux.
No GP hyperparameters or physical parameters are re-estimated.

The observed profiles subtract the weighted time-common spectrum separately
for each night/order/native pixel. They subtract neither the physical template
nor the GP. Their display weights retain the archived fitted white scales, so
"without a physical template" does not imply a model-independent noise
calibration. A shared reference can self-subtract a time-dependent planet
signal. Residuals consistently apply the same projection to data, baseline,
and the conditional GP mean.

The blue/core/red display bands select existing bin centers in [-40,-16),
[-8,8), and [16,40) km/s. The original eight-km/s grid is retained, including
its -0.138960 km/s edge offset. Display white errors and interleaved-pixel
differences are descriptive; adjacent pixels, orders, projected bins, and GP
residuals need not be independent. This audit does not turn those errors into
a detection significance. Q values for different retained samples are not
likelihood-ratio comparisons.

## Main executed comparisons

All entries are ppm. Positive residual means data exceed baseline plus
conditional GP. The same primary RV/amplitude is retained for alternate pixels,
unlike the separately profiled nuisance values in the earlier fit report.

| Selection | Observed blue | Observed core | Observed blue minus core | Blue residual | Core residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary pixels | 657.89 | -367.22 | 1025.12 | 409.06 | -235.24 |
| Alternate pixels | 536.50 | -334.24 | 870.74 | 286.92 | -199.60 |
| All native pixels | 596.16 | -350.54 | 946.70 | 335.40 | -219.37 |
| Science phase <0.40, rebuild reference/GP mean | 555.24 | -448.74 | 1003.97 | 409.51 | -200.21 |
| Science phase <0.40, full reference/GP mean fixed | 663.72 | -281.23 | 944.95 | 433.85 | -210.94 |
| Common night phase 0.4032--0.435, rebuild reference/GP mean | 358.31 | -500.99 | 859.31 | -157.68 | -186.28 |
| Common night phase 0.4032--0.435, full reference/GP mean fixed | 405.60 | -553.88 | 959.47 | -127.98 | -279.90 |

The early-phase blue excess persists under both treatments. In the common
late-phase interval, the blue residual changes sign while the negative core
residual remains. This is a conditional residual pattern, not an independent
measurement of atmospheric phase evolution. A phase cut can change the
reference, GP conditional mean, and display weights as well as the data.

The primary conditional GP contributes 106.46 ppm to the displayed blue-minus-
core difference, compared with 1025.12 ppm in the observed contrast. Conditioning
the same GP on zero planet gives a 63.67 ppm contribution. The grey/slope GP
does not create the entire feature, but this comparison does not test all
possible spectral systematics.

## Night and support limitations

August 8 retains 83 science exposures and no eclipse exposures. August 20
retains 21 science exposures and 15 eclipse exposures. August 8 supplies
97.25% of the combined blue-band display weight, 87.96% of the core weight,
and 84.71% of the red weight.

| Night/order | Primary observed blue | Alternate observed blue | Primary observed core | Alternate observed core |
| --- | ---: | ---: | ---: | ---: |
| August 8, order 0 | 581.59 | 444.64 | -179.83 | -132.94 |
| August 8, order 1 | 692.60 | 609.23 | -369.93 | -357.91 |
| August 20, order 0 | 589.83 | 670.70 | -894.89 | -990.63 |
| August 20, order 1 | 1124.63 | -135.25 | -840.86 | -647.88 |

On August 20, the -36.139 km/s blue bin has no support. The -28.139 km/s bin
uses only exposures 0--2 at phases 0.40320--0.40631: nine measurements in
order 0 and six in order 1 on primary pixels. The -20.139 km/s bin uses only
exposures 0--6, with 31 and 22 measurements. Core bins use approximately
14--19 exposures depending on pixel selection/order. The fixed 110--390 km/s
stellar-coordinate window is the reason that the different bands sample
different phase ranges. The August 20 blue value changes from 912.99 to
182.55 ppm between interleaved pixels; its primary white display error alone
is 529.70 ppm. Do not describe the detailed blue-peak height as replicated
with comparable precision on both nights.

## Selection controls

- Removing each of the 104 science exposures in turn leaves combined observed
  blue-minus-core between 979.79 and 1061.94 ppm, and residual blue-minus-core
  between 601.52 and 680.66 ppm. No single exposure removes the feature.
- Removing each of five contiguous science blocks per night gives observed
  differences 939.01--1080.24 ppm and residual differences 531.08--703.19 ppm.
  The blue and core residual signs persist in the combined data.
- All four stride-four pixel offsets retain positive observed blue-minus-core
  (812.67--1149.25 ppm). Removing the top 1% of native pixels by temporal
  weighted RMS leaves 1017.52 ppm. That last exclusion is explicitly
  exploratory and data-dependent, not a proposed replacement reduction.
- Changing the stellar-window lower boundary from 110 to 120 or 130 km/s,
  while holding the early science phase selection <0.40 fixed, changes
  observed blue-minus-core from 1003.97 to 996.43 or 992.35 ppm. Residual
  differences are 609.72, 598.90, and 602.49 ppm. This particular edge-mask
  change does not explain the early-phase discrepancy.
- Shifting the eight-km/s display bins by four km/s leaves an observed
  difference of 902.85 ppm. Twelve-km/s bins give 823.42 ppm. These displays
  cover different precise velocities and show the effect of bin placement.
- Extending science selection to phase 0.45 adds one August 8 and ten
  August 20 exposures. None supplies a native blue/core sample inside the
  adopted stellar window. They change the common reference: the combined
  observed difference becomes 1031.66 ppm. Only observed values are reported
  for this extension; the saved uneclipsed atmosphere is not an ingress model.
- An eclipse-only reference exists only for August 20. All eclipse exposures
  give observed blue/core 716.93/-683.23 ppm. Using early (0.47--0.48) or late
  (0.48--0.50) eclipse subsets gives blue values 1105.39 and 367.06 ppm; core
  values remain about -674 to -684 ppm. Reference uncertainty matters for the
  sparse second-night blue band.

## Artifacts and limits

`review.json` contains compact case tables, bin support summaries, exclusion
extrema, input/source hashes, and raw artifact hashes. Full exposure identities,
weights, and all exclusion results are in the pinned external
`selection_details.json`; `profiles.npz` and `exposure_bands.npz` retain the
arrays. Per-exposure band arrays use the exact velocity windows, whereas the
summary display bands select existing bin centers.

`pixel_night_comparison` shows observed profiles and fixed-baseline residuals.
`phase_support` shows science exposure weights and per-exposure blue-minus-core.
`observed_frame_maps` shows the same time-centered data against archived
wavelength and planet-frame velocity, without physical or GP subtraction.
Blank regions are missing support or excluded ingress, not zero flux. The
plots do not identify a unique astrophysical or instrumental origin.

This audit uses only the two available nights. Archive wavelength-frame
provenance, upstream telluric/blaze/stellar corrections, and correlated
uncertainties are separate audit components. The data and production source
are unchanged.
