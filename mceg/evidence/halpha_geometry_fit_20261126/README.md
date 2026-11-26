# Geometry experiment in native observation space

`examples/halpha_geometry_fit.py` accepts a complete physical snapshot and
recomputes its H-alpha branch extinction, emissivity, and thermal widths before
rendering the spherical dayside disk and halo. The continuum must also come
from that snapshot. It is not an emissivity-multiplier experiment.

Required snapshot fields are `temperature_k[8]`, `bound_cm3[8,18]`,
`radial_velocity_kms[8]`, `wavenumber_cm[1201]`,
`continuum_extinction[8,1201]`, and `continuum_emission_hz[8,1201]`.
The frequency grid must be the archived observer grid. Recompute the last two
fields with `model.continuum_coefficients_at(T, physical, wavenumber_cm * c_cgs)`
at the new solution, including its electron density and chemical populations.
Qualification of the physical root remains a separate requirement.
Include `electron_cm3[8]` and pass a matching `--baseline-physical` snapshot to
record electron-density changes. The physical receipt also records T, radial
velocity, n2/n3 sums, branch source functions eta/alpha, and local radial
line-center optical depths. These are recomputed from both states. The local
optical-depth diagnostic excludes intercell velocity shifts; the observer
retains the actual velocity field.

The comparison retains the original phases, day/night prescription, inner
boundary, stellar normalization, instrumental convolution, native pixels,
common-spectrum projection, and posterior-median GP covariance. Both shapes
profile the same two nuisance parameters: one RV and one global amplitude.
The candidate is also evaluated at the baseline's fitted nuisance parameters.
The original even-pixel selection and the alternate odd-pixel selection are
both evaluated with the same archived GP hyperparameters, held fixed between
templates. By-night/order residual summaries retain globally profiled parameters.
The native Q statistic retains GP covariance; display band averages use white
errors only and are descriptive, not significances.

```sh
taskset -c 6 env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/escape-halpha-mpl \
  PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src \
  python examples/halpha_geometry_fit.py \
  --state-npz /path/to/new/fixed_inputs.npz \
  --root-review /path/to/new/root_review.json \
  --baseline-physical /tmp/escape-halpha-equilibrium-current/call_001_baseline.npz \
  --output /tmp/escape-halpha-geometry-fit/new_fine
```

The raw per-phase spectra and projected arrays are kept outside git. Their
hashes and all private input hashes are retained in the compact JSON review.
The supplied ExoJAX source path is the same compatible checkout used by the
archived observer; the installed system package lacks its `AdbNist` export.

The archived physical template reproduces Q = 38326.64894003179. The displayed
blue/core/red **model plus conditional GP** means are 248.8352, -131.9788, and
966.6503 ppm. These are not residuals: the corresponding **data minus model**
means are 409.0567, -235.2447, and -23.1787 ppm. The windows are [-40,-16),
[-8,8), and [16,40) km/s, respectively.

`baseline_controls.json` records end-to-end controls using the same archived
atmosphere on both sides. Recomputed branch coefficients agree to 2.3e-16
relative error, and the full seven-phase rerender differs by at most 0.00655 ppm
from the archived fine-grid observer. The resulting Q differences are -0.000633
on the primary pixels and -0.000355 on alternate pixels. The rendering functions
and their constants are unchanged from that executed control; the later analysis
additions are checked against its saved spectra and pinned review. No result in
this baseline control is evidence for an improvement of a new physical model.

The accepted canonical CPU solution is
`/tmp/escape-halpha-equilibrium-final-cpu`. Its final residual infinity norm is
1.45856e-10, its repeated full evaluation is identical, and all physical and
input guards pass. The exported state exactly matches the independently checked
GPU checkpoint; the canonical CPU run takes zero Newton steps. The entry checks
and source/input hashes are in `cpu_export_entry.json` and `comparison.json`.

Full nonlocal spherical H-alpha feedback worsens this fixed-recipe fit. The
following band values are data minus (planet plus conditional GP), in ppm:

| Pixels | Template and nuisance treatment | Q | Blue residual | Core residual |
| --- | --- | ---: | ---: | ---: |
| Primary | Baseline, RV/amplitude profiled | 38326.64894 | +409.06 | -235.24 |
| Primary | Spherical coupling, baseline RV/amplitude | 38401.20496 | +549.03 | -560.50 |
| Primary | Spherical coupling, RV/amplitude profiled | 38392.63176 | +594.61 | -529.90 |
| Alternate | Baseline, RV/amplitude profiled | 38012.94009 | +309.60 | -206.82 |
| Alternate | Spherical coupling, baseline RV/amplitude | 38079.59761 | +444.10 | -508.42 |
| Alternate | Spherical coupling, RV/amplitude profiled | 38072.75188 | +476.61 | -488.04 |

The profiled Q increases by 65.98282 and 59.81179 for the two pixel selections.
All four night/order contributions worsen in each selection. Both the positive
blue residual and negative core residual increase in absolute value, including
with nuisance parameters fixed. These are conditional comparisons at fixed GP
hyperparameters, not significances or evidence ratios. The figure reports the
increase in Q directly. In the JSON receipt, `delta_Q_improvement` is baseline
Q minus candidate Q, so its negative values mean worsening.

The actual branch source functions at the dominant inner emitter (cell 3,
1.2583 Rp) fall to 0.84465--0.84484 of baseline, while those at the upper receiver
(cell 4, 1.5125 Rp) rise to 1.44369--1.48450. Their inner-to-upper contrast falls
to 0.56898--0.58520 of baseline. The total local radial optical depths at the
three branch centers rise by about 4.2% in cell 3 and fall by about 2.6% in cell 4.
Temperature, electron density, and radial velocity change by less than 0.3% in
these two cells. The self-consistent population response thus reduces the
contrast needed by the observations in this experiment. This result does not
exclude other physical parameter choices.

Observer refinement changes the maximum native model value by less than
0.001881 fitted white standard deviations. Fine-minus-coarse Q is -0.11537 on
primary pixels and +0.03194 on alternate pixels. These numerical changes are
small compared with the physical-model degradation. `comparison.json` records
the individual night/order scores, all band weights and descriptive white-error
diagnostics, full layer responses, and raw artifact hashes.

`observed_comparison.pdf` and `observed_comparison.png` show the same projected
data with both profiled templates and their respective conditional GP means,
followed by data-minus-total-model residuals. `comparison_rows.tex` provides
the numerical rows including fitted RV and amplitude. The raw coarse and fine
results are in `/tmp/escape-halpha-geometry-fit/final_cpu_coarse` and
`/tmp/escape-halpha-geometry-fit/final_cpu_fine`.

`nuisance_multistart.json` checks initial RV values -15, -10, 0, +10, and +15
km/s at initial log amplitude zero, spanning the entire allowed RV interval.
All 20 local optimizations succeed across both templates and pixel selections.
The best Q differs from the original profile by less than 1e-9 in every case;
all starts return the same spectral alignment. This is a specific control for
missed alignment minima, not a proof of a global optimum.

`free_model_reference.json` verifies all 35 archived input/script hashes and
exactly recovers the previously processed free-model scores under the same
primary observation/GP settings: Q = 38304.82193 for its original fixed shape
and Q = 38288.89491 for its previously optimized shape. These reference shapes
have different physical freedom and retain their original plane geometry;
they are not additional fits performed for this comparison.
