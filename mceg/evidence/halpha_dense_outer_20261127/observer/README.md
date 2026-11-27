# Supplied dense-atmosphere observer diagnostic

`examples/halpha_dense_observe.py` recomputes the three H-alpha line branches
from a supplied atmosphere and renders the same spherical dayside disk and halo
used for PR13. It retains the supplied continuum and radial flow, and the
original instrumental response, rotation, zonal flow and native observation
operator. The physical qualification belongs to the separately pinned state
review; successful rendering is not an equilibrium or force-balance test.

The archived PR13 fine baseline is reused. Both native pixel interleaves use
the same saved GP hyperparameters. Within an interleave, all candidates share
the covariance and reference, while their conditional GP means are recomputed.
Results retain original fixed RV/amplitude and, separately, refit those two
shared parameters over both nights/orders and all retained phases. Each refit
uses the original bounds and two optimization starts. Q is descriptive.

The primary target is the sealed `common_complete` phase row 3: blue
[-40,-16), core [-8,8), red [16,40) km/s, with uniform 4 km/s subbins and equal
exposure/order weights. Both interleaves use the same 12 August-8 exposures,
phase 0.3761495834–0.3932225607. August 20 has no complete target support and is
reported as null. Other complete phase rows, individual target orders and the
four native group scores are retained to show collateral changes.

## Replay

Run from the repository root with the archived data and ExoJAX checkout present:

```sh
taskset -c 18-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/escape-dense-mpl \
  PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src \
  python examples/halpha_dense_observe.py \
  --state-npz CANDIDATE/fixed_inputs.npz --state-review CANDIDATE/review.json \
  --raw-output /tmp/escape-halpha-dense-observe/CASE \
  --output mceg/evidence/halpha_dense_outer_20261127/observer/CASE
```

Required state arrays are `temperature_k`, `bound_cm3`,
`radial_velocity_kms`, `wavenumber_cm`, `continuum_extinction` and
`continuum_emission_hz`. This observer retains the original eight atmospheric
cells and seven phase nodes. Fine rays do not establish atmospheric radial or
phase convergence. Render caches reject a changed state, recipe or source.

`baseline_parity` uses `--baseline-only` and performs no transfer. `D1` is the
previous accepted sphere-radiation equilibrium without density enhancement.
Its spectra were reused with these additional arguments:

```sh
--state-npz mceg/evidence/halpha_equilibrium_20261126/observer_inputs.npz \
--state-review mceg/evidence/halpha_equilibrium_20261126/accepted_root_review.json \
--candidate-spectra /tmp/escape-halpha-geometry-fit/final_cpu_fine/spectra.npz \
--render-review /tmp/escape-halpha-geometry-fit/final_cpu_fine/review.json \
--rendered-state /tmp/escape-halpha-equilibrium-final-cpu/fixed_inputs.npz
```

Reuse checks the old spectrum and state hashes, every physical array used by
the observer, renderer/dependency source hashes, phase/velocity/baseline arrays
and rendering settings. The compact receipts pin all inputs and external raw
arrays; PDF/PNG show the separately refitted candidate and baseline. Fixed
nuisance results remain in each receipt. D1 separates changing the radiative
closure from subsequently increasing the outer density.

## Continuation exports

The predeclared observer cases are D=2, 5, 10 and 20, processed in that order.
`watch_candidates.py` waits for each successful stage summary and verifies the
state export and solver source/input hashes before invoking the observer. It
records additional D1 differences and the fitted RV/log-amplitude distances
to both optimization bounds in `stage_assessment.json`. A fit within 1% of
either allowed parameter width is flagged; optimizer success alone does not
establish an unconstrained or global optimum.

GPU exports may be rendered while their independent CPU checks are pending.
Their stage assessments remain explicitly provisional. Final physical
qualification must be supplied by the independent CPU receipts before these
cases are used in a final physical assessment. A failed or absent stage is
never replaced by an unaccepted intermediate state.

An additional exploratory D=1.2 case was requested before any of its observer
values had been evaluated, to use the first converged density increment while
the predeclared larger factors continued. `qualify_case.py --cpu-review PATH`
attaches the independent CPU receipt after checking its output hash, exact
checkpoint coordinates and all physical arrays used by the observer. Add
`--exploratory` for this extra case. The GPU stage summary is left unchanged;
only `stage_assessment.json` receives the final CPU qualification.

`collect_comparison.py` updates `comparison.json` and `comparison_table.md`
from the saved observer receipts. It reports fixed/profiled residuals and Q
against both PR13 and D1, blue-minus-core residuals, individual order results,
other-phase changes, and the separate processed planet and conditional-GP
contributions. These are native log-flux contrasts multiplied by 1e6. The
processed planet term includes the original reference projection; it is not
the intrinsic planet spectrum. Disk and halo are combined in the saved renders.

At the highest accepted density, `refine_target.py --case OBSERVER_CASE
--raw-output RAW_DIRECTORY` evaluates the authorized finite ray-grid control:
only phase nodes 0.375 and 0.400 use nlon=144, nlat=72, radial points=12,
azimuths=144 and subsegments=12. The other five phase spectra remain identical.
Its `target_ray_refinement.json` reports target-band changes with fixed and
refitted nuisance parameters. This brackets the 12 target exposures but does
not establish global observer, phase-grid, or atmospheric radial convergence.
The same control also tries RV starts at -15, -7.5, +7.5 and +15 km/s for
the original and refined templates on both interleaves, retaining the same
bounds and fixed GP covariance. Agreement checks local-start sensitivity,
not global optimality; original observer receipts are preserved.

The single column-selected extra case follows `column_selection_protocol.json`.
`select_column_case.py --cpu-directory CPU_RECEIPTS --output REPLAY_JSON`
compares D12 and D15 with D10 after independent CPU checks, using only their
1.725–3 Rp n=2 columns and the existing #1 column reference. It verifies the
copied structure columns against exported populations and never reads an
observer score. The recorded selection is protected from overwrite. This is
an empirical column-reference selection, not independent statistical evidence
or a match to the full radial opacity/source profile. Extra cases carry an
asterisk in the comparison table.

`upper_column_bracket_selection.json` records a later adaptive decision, made
before evaluating D15 observations: add the independently accepted D15 so
D12/D15 bracket the #1 outer column while D20 is still solving. D15 is an
additional exploratory case, not a predeclared density test or a second
winner of the nearest-column rule. The original D12 selection is unchanged.

`audit_receipts.py` checks saved source/data/raw/figure hashes, matching target
support and nuisance references, explicit unsupported rows, and the Q and
band-decomposition identities in the assembled comparison. It performs no
new fit or transfer calculation.
