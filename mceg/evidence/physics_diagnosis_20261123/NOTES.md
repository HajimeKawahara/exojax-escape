# Legacy / multilevel hydrogen diagnosis

This archive records finite diagnostic experiments, not a replacement physical
model or a validated observational likelihood. The scientific provider source
is unchanged from merged PR #9 (provider commit
`5898d686202fb83f5bce66a0d44e7f2de8d080a5`). ExoJAX was loaded from the extracted
source at `38cb2e44f153ad6d729e3253e235702de52920cf`.

## What was compared

- `legacy_inputs/` preserves the actual PR #1 parameters, profiles and spectra.
  `legacy_audit.*` inspects these and the previous NLTE diagnostic outputs.
- `hybrid_controls.*`, `hybrid_metrics.json` and `hybrid_arrays.npz` re-synthesize
  all three legacy models and substitute their source, opacity, temperature and
  velocities into the current observer. These substitutions do not satisfy
  statistical equilibrium or the wind equations. A source-only substitution
  does not improve the fit.
- `matched_fit/` retains all local-fit outcomes, including worse basins, and the
  selected minima's arrays and numerical checks. The parameter domains are
  explicit. These searches do not explore temperatures above 14000 K or restore
  the old independent excited-column normalization.
- `rate_budget.*` re-solves local charge/populations under selective incident
  occupations. Gross incoming rates include resonance cycles; they are not net
  escaping photon production. The all-Lyman-off inversion is diagnostic-only.
- `irradiation_controls.*` repeats selected interventions at all atmospheric
  layers and generates spectra. All displayed cases have valid roots. The
  unchanged case agrees with the public forward path. The higher-Lyman x0.1
  manual case also passes layer and atomic-cutoff refinements.
- `blackbody_uv_budget.*` integrates the currently prescribed blackbody and
  compares its UV shape to a published PHOENIX example. This is not the exact
  stellar input used in Zhang et al. (2026).

All spectrum scores use the same 87 observations and supplied uncertainties,
with an independently profiled constant offset. They use phase 0.375 and omit
the full observation reduction. Scores are uncertainty-normalized residual
RMS, not reduced chi-square, significance or model-selection evidence.
The common re-synthesis grid is 128 layers, 12 x 6 disk points and 601
wavenumbers; small differences from the original saved fits are expected.
The figures use the branch-profile-weighted rest-center source. The matched-fit
summary additionally reports an explicitly labeled integrated-coefficient
source diagnostic, which is a different quantity.

## Interpretation and remaining work

The old good fits realize a strong decrease of source brightness with altitude
and a freely normalized excited column. They do not uniquely determine the
true atmosphere. Current blackbody Ly-beta pumping and higher cascades sustain
a much brighter upper source. Selective Lyman interventions demonstrate this
dependency, but uniform suppression also removes deep emission and fails to
repair the full spectrum. Changing all Lyman lines can even increase n3/n2.

Prioritize a realistic boundary stellar spectrum and frequency/depth-dependent
radiation coupled to the populations. Then evaluate thermal balance, the lower
temperature domain and ionization-dependent density/pressure together with the
wind. Recover the full observation operator independently. Broadening, H-minus
and multidimensional geometry remain possible residual issues; the natural
lifetime control alone does not test those mechanisms.

The MCEG contract gives scoped numerical results separate Assessments and leaves
physical adequacy unknown with a FollowUp and limited research-priority
Decision. The provider's `test_radiation_feedback` task stays open: no coupled
RT+SE repair was evaluated here. There is no new Cloudy run in this archive.

## Replay and provenance

Executed Python scripts are preserved byte-for-byte, including their original
absolute paths. On a different machine, copy them to a disposable checkout and
remap those paths; do not overwrite existing user directories. The required
mapping is:

| Original location | Retained content / source |
| --- | --- |
| `/tmp/escape-physics-diagnosis` | Provider checkout at the revision above, with this archive extracted to `results/physics_diagnosis` |
| `/home/kawahara/exojax-escape/results/three_models` | `legacy_inputs/` |
| `/tmp/exojax-escape-nlte-spectrum` | Same provider source; `results/nlte_asymmetry/` comes from `prior_inputs/` |
| `/tmp/exojax-spectrum-upstream/src` | ExoJAX source revision above |
| `/tmp/escape-matched-fit` | `matched_fit/` |
| `/tmp/escape-model-audit` | Writable output for `legacy_audit.py` |

Use the provider's `src`, `examples`, `tools`, and the pinned ExoJAX `src` on
`PYTHONPATH`; use CPU JAX with x64 enabled as in each script. The execution used
Python 3.12, JAX/jaxlib 0.10.0, NumPy 2.1.0, SciPy 1.17.1, Matplotlib 3.10.1 and
pandas 2.2.3. NIST requests are replaced by the retained provider test fixture.
Run the audit and hybrid scripts before the local rate-budget script. Run
`matched_fit/fit.py --mode nonisothermal` and `--mode isothermal`, then the
matching `check.py` commands. The low-mass control is a separate fit script.
Full fitting/compilation takes several minutes. Diagnostic outputs should be
written to a fresh directory; rate-budget input NPZ files must not be overwritten
concurrently (the original failed lazy-read attempt and retry are recorded).

MCEG is external. Check `investigation.yaml` from the provider root using the
MCEG revision and commands retained with the review. A passing handling check
records a response to an unresolved issue; it does not validate the atmosphere.

## Sources

- Y. Zhang et al. (2026), [Sections IV–V](https://arxiv.org/html/2606.23899v1#S4).
- A. Garcia Munoz & P. C. Schneider (2019),
  [stellar irradiation](https://arxiv.org/html/1910.00267v1).
- Y. Zhang (2026), [Zenodo 20707255](https://doi.org/10.5281/zenodo.20707255),
  CC BY 4.0. Plots contain the public averaged data; large HDF5 files are absent.
