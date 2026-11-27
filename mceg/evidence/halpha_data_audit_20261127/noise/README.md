# Native H-alpha noise audit

`review.json` is an executed diagnostic from
`examples/halpha_noise_audit.py`. It reuses the archived observations, spectra,
and GP samples; it performs no atmosphere solve and changes no production code.
Source and input hashes are checked before and after execution.

## Results

- The existing baseline is reproduced: Q = 38326.64894003179, blue residual
  +409.057 ppm, core residual -235.245 ppm, and blue-minus-core residual
  644.301 ppm. The corresponding observed contrast is 1025.115 ppm.
- The common-reference operation includes its induced covariance. The new
  NumPy coefficient calculation agrees with the production GP mean within
  1.1e-16 and covariance within 9.3e-23. No confirmed implementation defect was
  found in this scoped projection/GP audit.
- Archive errors are empirical proxies, not independently supplied photon
  noise. In the retained stellar window, their median ratios to the full-time
  scatter of the saved flux are 1.139, 1.096, 1.076, and 1.043 for the four
  night/order groups. Errors also vary with exposure and are not exactly a
  time-scale times a wavelength-scale. The complete upstream construction
  cannot be reconstructed from these arrays.
- Replacing data-dependent log weights `(1+r)^2/sigma^2` by `1/sigma^2` changes
  the blue-minus-core residual by +1.158 ppm. This specific approximation does
  not explain the feature.
- With 64 archived joint noise draws and a fresh RV/amplitude fit for each,
  blue-minus-core residuals span 633.374--665.952 ppm at their 5th--95th
  percentiles. These are noise-family sensitivity percentiles, not a new
  posterior interval. All 64 optimizations succeeded. Removing the GP while
  retaining the white scales gives 753.706 ppm.

The GP is a grey-plus-slope time model. The full native eclipse data retain
adjacent-pixel residual correlations of 0.236 and 0.168, compared with
independent-pixel conditional simulation ranges of about +/-0.03. Keeping
alternate pixels suppresses that effect, but does not establish independent
noise: some weaker negative spectral and temporal departures remain. There is
no clear excess quadratic spectral mode in the eclipse control. There are only
15 eclipse exposures on the second night, and none on the first.

## Covariance and exposure support

The blue-minus-core white standard deviation, including common-reference
off-diagonals, is 132.218 ppm. Applying the same GP estimator to both the data
and simulated noise gives 131.313 ppm with fixed planet parameters. Local
linear RV/amplitude refitting gives 129.577 ppm in the assumed model family.
These numbers are conditional covariance calculations, not calibrated
significances for a physically fitted atmosphere.

An empirical time-block multiplier calculation retains all wavelengths and
both orders together within each block. Its blue-minus-core standard deviation
is 153 ppm for one-exposure blocks, 160--186 ppm for two exposures, 193--209 ppm
for four, and 229--251 ppm for eight. Block-boundary offsets generate the
ranges. The largest blocks have only about 3--5 effective contributors to the
contrast variance. They include any actual phase-dependent mean-model error
as well as noise, assume independence between blocks, and do not calibrate a
p-value or recover slower upstream systematics.
The multiplier input is the residual after common-reference and GP fitting,
which has already been reduced by those fits. It is not a reconstruction or
bootstrap calibration of the raw extraction noise and latent GP process.

The per-exposure contrast does not reveal a shared order fluctuation on the
first night: observed and residual order correlations are respectively
0.00028 and -0.00835 over 70 supported exposures. On the second night, only
seven exposures have any simultaneous blue/core coverage, including partial
bands; the residual correlation -0.658 lies within the broad conditional
independent-order range [-0.788, 0.755]. The moving-line native residual
correlation over [-40,40) km/s is also small (-0.0119 and -0.0114 after
standardization). The second-night blue-only native comparison has just 28
matched samples. These findings do not establish cross-order independence.
The similar observed/residual curves within an order mostly reflect their
shared data, not agreement between the two orders.

## Interpretation limits

Q values from different covariance or weight choices are not likelihood-ratio
tests: the covariance determinants and, for changed weights, contrast measures
also change. The recorded Q values only monitor each conditional nuisance fit.
Simulation envelopes are marginal discrepancy diagnostics, not simultaneous or
calibrated tests. Posterior draws explore the existing GP family, not arbitrary
spectral systematics. These controls leave phase-dependent sampling, physical
variability, and incompletely known upstream processing as distinct issues.

Run with CPU float64 and the archived ExoJAX import tree:

```sh
taskset -c 2-9 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1 MPLCONFIGDIR=/tmp/escape-halpha-noise-mpl PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src python examples/halpha_noise_audit.py --output mceg/evidence/halpha_data_audit_20261127/noise --simulations 128 --posterior-draws 64
```
