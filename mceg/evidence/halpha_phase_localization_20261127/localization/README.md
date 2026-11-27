# Phase localization with complete native support

The most consistent fine-bin direction is a blue deficit and core/red excess
in the fixed baseline during the supported part of phase 0.375--0.40. The
actual common-support exposures cover 0.37615--0.39322 on August 8. This sign
pattern occurs in both orders and both native-pixel interleaves. Earlier
core residuals vary strongly with order and pixel selection. Complete blue
coverage is absent after phase 0.39322 and throughout August 20, so the late
full-blue residual is not measured by these data.

These are conditional residual summaries of processed data. They do not
identify a unique atmospheric effect or reconstruct an intrinsic spectrum.
Conditional and block uncertainty controls are recorded separately in the
sibling uncertainty evidence; descriptive white errors here are not detection
significances.

## Declared quantity and replay

```sh
taskset -c 10-17 env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/halpha-phase-mpl \
  PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src \
  python examples/halpha_phase_localization.py \
  --output mceg/evidence/halpha_phase_localization_20261127/localization \
  --raw-output /tmp/escape-halpha-phase-localization
```

The phase edges are fixed at 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.435;
the coarse sensitivity uses 0.30, 0.35, 0.40, 0.435. The baseline RV and log
amplitude remain 2.8601527471 km/s and 0.1240323529 for both interleaves.
Every group retains its complete original science/eclipse samples for the
common reference and conditional GP calculation. Only the displayed linear
coefficients change; no reference or GP is refitted to a phase subset.

Blue/core/red are exactly [-40,-16), [-8,8), [16,40) km/s. Each fixed band is
divided into four-km/s subbins anchored at these edges. Original inverse-error
weights form each native subbin mean; subbins and eligible exposures then
receive equal weight. Simultaneous orders receive equal weight. A complete
band requires at least one valid measurement in every subbin and valid pixel
centers at or outside both band edges, in both orders. This conservative rule
does not assume a pixel response or a spectrum beyond the mask edge.

Three separate products are retained:

- `common_complete`: all three bands use the same exposures with complete
  support for every band in both orders. This is the primary shape comparison.
- `band_complete`: each band uses its own fully supported exposures. This
  retains late core/red information, but different bands need not sample the
  same phases.
- `legacy_display`: the previous eight-km/s display with its original bin and
  white-error weighting, restricted in displayed phase. Partial bands are
  allowed only in this traceability product. They are not full-band estimates.

Unsupported rows are marked explicitly and have null values; their numerical
linear maps are zero only as an interface convention. No plotted zero is
substituted for missing data. A plotted phase coordinate is the declared-bin
midpoint; metadata give the actual supported range and weighted mean.

## Complete-band results

The table gives data minus baseline minus conditional GP, in ppm, using common
exposures for all three bands. A positive value means the baseline plus GP
underpredicts the processed data. Counts refer to physical exposures on
August 8; both orders are used for every exposure.

| Declared phase interval | Exposures | Blue primary / alternate | Core primary / alternate | Red primary / alternate |
| --- | ---: | ---: | ---: | ---: |
| 0.30--0.325 | 14 | +455.57 / +595.92 | +57.48 / +41.73 | +149.96 / +86.07 |
| 0.325--0.35 | 16 | +424.86 / +365.83 | +476.93 / -129.82 | +224.32 / +294.40 |
| 0.35--0.375 | 16 | +307.14 / -9.27 | -788.43 / -88.08 | -339.66 / +19.00 |
| 0.375--0.40, actual support to 0.39322 | 12 | +590.52 / +402.49 | -585.45 / -449.22 | -215.84 / -223.31 |
| 0.40--0.425 | 0 | unsupported | unsupported | unsupported |
| 0.425--0.435 | 0 | unsupported | unsupported | unsupported |

The 0.375--0.40 supported subset has the same residual signs separately in
both orders, for both interleaves. In contrast, the large primary core
residual at 0.325--0.35 is +974.69 ppm in order 0 and -20.82 ppm in order 1;
the alternate-pixel combined value is -129.82 ppm. The 0.35--0.375 discrepancy
also weakens markedly on alternate pixels. These fine structures should not
all be promoted to equally secure phase-dependent physical requirements.

Coarsening the primary comparison to 0.35--0.40, with the same complete-band
support ending at 0.39322, gives blue/core/red residuals +428.59/-701.44/-286.59
ppm on primary pixels and +167.20/-242.86/-84.84 ppm on alternate pixels.
The overall direction remains, while the magnitude depends on pixel selection
and the phase interval averaged.

## Coverage and the late core

Primary complete-band counts over all selected exposures are 58/74/83 for
blue/core/red on August 8 and 0/11/15 on August 20. Alternate counts are the
same except for August 20 red, with 13 complete exposures. The last primary
blue exposure is at phase 0.39322256; the last complete core exposures are
0.41806004 and 0.41873014 on the two nights. The last phase bin, 0.425--0.435,
therefore measures only the complete red band, not complete blue or core.

With band-specific support, the 0.40--0.425 core residual is -523.46/-421.14
ppm (primary/alternate), using 12 August 8 and 11 August 20 exposures. It is
mostly an August 20 effect: the corresponding per-night values are
-100.29/+20.09 ppm on August 8 and -985.10/-902.48 ppm on August 20. This is
not a common complete-blue/core/red comparison.

`match_nights.py` is an independently replayable follow-up that leaves the
405-row map interface unchanged. It restricts both nights to the
support-defined inclusive window [0.4032027722, 0.4180600420], with ten
complete-core exposures per night. At fixed full-data reference/GP, core
residuals are -40.96 versus -977.26 ppm (primary) and -87.12 versus -844.74
ppm (alternate). The night-20 minus night-8 differences are -936.30 and
-757.62 ppm. Thus the small difference in the original phase-bin endpoints
does not explain the conditional night difference. Separate nightly
references and fitted GP means remain: these are not absolute intrinsic
night-to-night flux differences. `match_nights.json` records the observed,
baseline and GP contributions separately; uncertainty controls consume its
separate matched maps.

The original partly supported late blue display is retained only as an audit
trail. Its apparent residual shrinkage cannot establish agreement across the
full [-40,-16) km/s blue band.

## Map interface and verification

`/tmp/escape-halpha-phase-localization/maps.npz` contains primary/alternate
maps `selection__g0__maps` through `selection__g3__maps`, each shaped
(405 rows, native exposures, native pixels). The original group order is
night0/order0, night0/order1, night1/order0, night1/order1. Row labels and
support metadata are in the adjacent `rows.json`. Main fine common-complete
all-night rows are 0--17, phase-major and then blue/core/red. Fine
band-complete all-night rows are 126--143.

Each map is the positive normalized average followed by the unchanged
weighted time-common-reference projection. It acts on dimensionless native
log flux. GP subtraction and nuisance-direction refitting are outside this
map and must be applied separately by uncertainty/response calculations.
`raw_maps` retain the positive pre-reference coefficients; native arrays and
the exact frozen baseline/GP are saved alongside them. Raw hashes and source
pins are in `review.json`.

The baseline Q is exactly 38326.64894003179. Numerical controls confirm native
arrays and frozen GP equality to the original loader, common-mode
annihilation, zero unsupported maps, nonnegative raw coefficients, legacy
full-phase display parity, and equality of the coarse complete-band maps to
the exposure-weighted fine maps. `controls.json` records executed values;
`validate_maps.py` preserves the replay source. This validates the declared
linear summaries, not their unique astrophysical interpretation.

`phase_residuals.pdf/png` separates the three products and the two pixel
interleaves. `phase_support.pdf/png` shows eligible physical exposures.
All previous audit scripts, records and atmosphere results remain unchanged.
