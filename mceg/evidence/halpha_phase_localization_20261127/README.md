# Phase localization of the H-alpha residual

The [four-page report](../../../doc/halpha_phase_localization.pdf) identifies
which complete bands can be measured and where the conditional residual signs
are consistent. It does not establish a calibrated physical phase evolution.
The preceding audit and its receipts remain unchanged.

- The same-exposure blue/core contrast is most consistently discrepant at the
  first night's actual supported phases 0.3761–0.3932. Both complementary pixel
  selections and both orders have blue-positive and core-negative residuals.
- Complete blue coverage ends at 0.39322 and is absent on the second night.
  Complete core coverage ends at 0.41873. A partly observed band is not a
  measurement of the complete peak or trough.
- In a common late-phase window with ten exposures per night, the core
  residual differs by −936/−758 ppm between nights. The two references and GP
  histories remain different; this does not identify intrinsic variability.
- The fine-bin and night-difference uncertainties depend strongly on the
  assumed covariance. Time-block proxies include possible physical residuals,
  have few effective contributors, and do not calibrate a significance.

The protocol fixes phase edges and exact velocity bands. Complete-band means
use uniform 4 km/s subbins and equal exposure/order weights. Three products
separate common B/C/R support, band-specific complete support, and the old
partly supported display. Raw maps and covariances are hashed local artifacts;
receipts, figures, source and replay helpers are committed.

## Replay

Restore the same saved atmosphere, stellar template, GP samples and public
residual files used by the previous audit. Paths and hashes are recorded in
the receipts; the archived numerical inputs are not downloaded by these scripts.
From the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src
python examples/halpha_phase_localization.py \
  --output mceg/evidence/halpha_phase_localization_20261127/localization \
  --raw-output /tmp/escape-halpha-phase-localization
python mceg/evidence/halpha_phase_localization_20261127/localization/validate_maps.py
python mceg/evidence/halpha_phase_localization_20261127/localization/match_nights.py
python examples/halpha_phase_uncertainty.py \
  --output mceg/evidence/halpha_phase_localization_20261127/uncertainty
python mceg/evidence/halpha_phase_localization_20261127/uncertainty/extend_contrasts.py
python examples/halpha_phase_controls.py
make -C doc phase-localization-pdf
```

The numerical controls cover reference annihilation, old-display agreement,
coarse/fine linear consistency, unsupported bands, covariance propagation,
256 fixed-family noise injections per pixel selection and finite native GLS
recovery. Independent implementations agree on the precision and response
matrices. These checks verify the diagnostic calculations; they do not validate
the archive's unknown covariance or earlier stellar/telluric reduction.
