# Model contracts and evidence

These provider-owned records use [MCEG](https://github.com/HajimeKawahara/mceg)
to connect model definitions, code, evidence, scoped Assessments and Decisions.
`escape` does not import or depend on MCEG. Use a separate authoring environment;
the scientific package and its ordinary tests work without MCEG.

## Current status

`supported`, `contradicted` and `unknown` describe an evidence-support Assessment
for one Claim revision, implementation/data Subject and Context. A numerical
failure is a contradiction only when a valid comparison violates that Claim's
criterion. A crash, missing artifact or untested physical proposition remains
inconclusive. Historical judgments retain their original records.

| Model or claim | Current recorded finding | Scope and remaining work |
| --- | --- | --- |
| H-alpha geometry, radial Doppler transfer and photon recycling | Parallel fixed-state diagnostics; physical profile improvement remains `unknown` | The [coupling contract](contracts/escape.halpha_coupling.yaml) and [report](../doc/halpha_coupling.pdf) retain source-separated absorption, escape and conversion outcomes. Populations and thermal balance are not re-solved; no improved observational fit is claimed. |
| Isothermal/nonisothermal Parker wind, with or without linear tides | Formulation and sampled numerical checks are `supported` | Prescribed temperature and constant mean molecular weight; no energy equation or planetary accuracy claim. See the [wind contract](contracts/escape.parker_wind.yaml). |
| Six-state local hydrogen NLTE and H-alpha transfer | Historical formulation/numerical reviews are `supported` | Their exact source and test scopes are retained; atmospheric accuracy and posterior convergence are `unknown`. See the [local NLTE contract](contracts/escape.hydrogen_nlte.yaml). |
| Extended local atom to observer H-alpha spectrum | Formulation, sampled gradients and the 128-layer numerical preset are `supported` | Six independent refinements and a joint control pass the 0.1-uncertainty criterion at seven conditions; 53 targeted tests pass. Atmospheric accuracy remains `unknown`. See the [spectrum contract](contracts/escape.nlte_spectrum.yaml). |
| Identical unattenuated irradiation in every observer-model layer | Approximation adequacy is `unknown` | A dedicated Equation, Assessment, open radiation-feedback FollowUp and provisional-use Decision track this known concern. Optical-depth diagnostics motivate a comparison; they do not measure the resulting H-alpha model error. |
| Earlier 32- and 64-layer observer presets | Resolution Assessments are `contradicted` | Their layer refinements change a pixel by 0.36711 and 0.10437 supplied uncertainties, respectively, above the unchanged 0.1 criterion. Both source/context records are retained. |
| Coupled radiation, populations, temperature and finite flow | Scoped numerical reviews are `supported` | Finite boundary problems; transonic selection, moving-line transport and planetary accuracy remain unverified. |
| New connected atmosphere to observer | Three scoped implementation Assessments are `supported` | The actual `HydrogenAtmosphereModel.spectrum` path includes radiation/SE, thermal flow and a variable-mu transonic wind. Current-source regression receipts accompany the review; physical adequacy is separate, and its [use manifest](uses/yapeng_atmosphere.yaml) remains **not ready**. |
| Pure-H reduction for the selected solar-composition target | Dominant-heating adequacy is `contradicted` | Controlled solar references contain substantial omitted metal heating and molecular cooling. Exact budget fractions change with continuum-grid refinement and are not converged. The author abundance choice is still unknown. See the [connected-atmosphere review](evidence/coupled_atmosphere_20261123/review.json). |
| Native solar composition, radiation and thermal wind | Selected eight-cell complete-root Assessment `supported`; five other use Assessments `unknown` | The [complete-root review](evidence/selected_e8_complete_20261125/review.json) verifies the production residual, physical guards and conservation at this saved equilibrium. Derivatives, independent reference agreement, omitted physics, numerical resolution and observational use remain unverified. The [solar use](uses/solar_proxy.yaml) requires all six checks and remains **not ready**; its scope is the selected solar proxy, not the author's unavailable inputs. |
| Same-source two-cell line-sampling control | Front observer `contradicted`; average observer and scoped host derivatives `supported` | The [paired review](evidence/solar_discretization_20261124/review.json) retains both converged population roots. Only within-cell averaging gives an applicable finite observer in this control. A pinned Decision rejects the failed case; these narrow checks do not satisfy complete solar use. |
| Two-cell production controls | Scoped root, finite-observer and implicit-primitive Assessments are `supported` | The [root review](evidence/solar_root_controls_20261124/review.json) distinguishes the ten-metal continuous-Doppler root/observer from the earlier nine-metal JAX reverse derivative. Zero primal iterations verify the implicit primitive, not cold-start or thermal convergence. |
| Wind/energy connection and continued roots | Sparse-density energy derivative `contradicted`; scoped repair and numerical roots `supported` | The [continuation review](evidence/solar_continuation_20261124/review.json) preserves a five-cell fixed-temperature root, positive JAX iterations, and the excessive expansion cooling from coarse density differences. The repair differentiates the same wind/profile interpolants. Velocity convergence alone does not establish energy-derivative convergence; the final thermal profile is checked separately below. |
| Independent wind and reference photon controls | Fixed-temperature native root and initial-profile wind `supported`; sharp-profile wind and C23 excited-Ca allocation `contradicted` | The [reference-control review](evidence/solar_reference_controls_20261124/review.json) separates the ten-metal fixed-temperature root, independent ODE accuracy, and a readout-only one-zone reference counterexample. Their final thermal and planetary effects remain under investigation. |
| Five-cell coupled thermal root and final-profile wind | Scoped numerical Assessments are `supported` | The [thermal review](evidence/solar_thermal_root_20261124/review.json) records a full residual of **8.43e−10**, with energy balance in three upper cells and a prescribed two-cell lower boundary. Final-profile expansion cooling agrees with an independent ODE within **0.240%**. The native spectrum misses the observed central trough and peak amplitudes; atmospheric resolution, physical adequacy and production gradients remain unresolved. |
| Thermal spectral sensitivity and observer sampling | Original loose sensitivity `contradicted`; refined local checks `supported` | The [gradient review](evidence/solar_gradients_20261124/thermal_review.json) separates the paired tolerance repair from the full thermal JVP/independent-root agreement (**2.39e−6** relative). Individual and joint observer refinements pass **0.1 supplied errors**. Production reverse AD, atmospheric resolution and the NUTS domain remain separate. |
| Reference Fe/Ca photon accounting | Original and single-correction failures stay `contradicted`; corrected local budgets are `supported` | The [photon review](evidence/reference_photon_budgets_20261124/review.json) compares four source variants with actual Fe activity and original/readout parity. Separate Decisions reject incomplete budgets. Three-cell algebraic identities do not establish solar thermal/reference convergence. |
| Matched hydrogen continuum and reference cooling | Scoped OU2 roots/reference/refinements `supported`; original constant-temperature RR-cache claim `contradicted` | The [matched review](evidence/qualified_hydrogen_reference_20261124/matched_ou2_review.json) records **less than 1%** local population differences on 267 cells and a separate cache-refresh heat control. This is a fixed-temperature pure-H slab, not the native bidirectional solar atmosphere. |
| Earlier nmax=4/5 finite atom versus Cloudy | `supported` in the eight declared cases | Maximum n2/n3 difference **1.100%**, below **5%**; reference refinement changes them by less than **0.049%**, below **1%**. |
| Original finite-atom agreement with Cloudy | New reviewed Assessments are `contradicted` | Original n3 errors **32.65%** and **30.22%** exceed **5%**. Original `unknown` records remain as history. |
| Original and repaired atom-size convergence | New reviewed Assessments are `contradicted` | nmax=4 to 5 changes n3 by **11.46%** in the original atom and **7.735%** in the repaired atom, exceeding **5%**. These original finite-atom claims remain false; the enlarged model has separate criteria below. |
| Enlarged hydrogen atom (default nmax=15, supported through 20) | New convergence and reference reviews are `supported` | Baseline n3 changes **0.2335%** from 15 to 20; maximum n2/n3 change across seven conditions is **1.4713%**. The fourteen matched Cloudy comparisons differ by at most **2.0315%**, below the unchanged **5%** criterion. |
| Earlier spectrum families and NIST/Rydberg center choices | Five inventory entries remain without dedicated contracts | Existing population or wind checks do not validate their complete geometry, transfer, broadening or observational applicability. |

The new default uses summed Anderson collision strengths for n=4,5; explicit
`vs80` preserves the previous approximation. Older atom/spectrum Assessments
retain their original source pins and do **not** automatically validate this
changed default. Current selected-path tests and independent collision anchors
are recorded in the connected-atmosphere review. The new solar path connects
metals, molecules and H/O overlap to the shared radiation and thermal equations;
it is a separately assessed candidate, not an automatic upgrade of the earlier
pure-H validation.

The [Cloudy investigation contract](contracts/escape.cloudy_investigation.yaml)
connects the original failure, paired rate-block experiments, physical choices
and repaired comparisons. The [coupled contract](contracts/escape.coupled_hydrogen.yaml)
retains the earlier formulation and execution reviews. The new contradiction
reviews use the measured criteria, not a reinterpretation of all failed tests.

The [enlarged-atom contract](contracts/escape.atom_convergence.yaml) records why
high-shell recombination and cascades change the low populations. It adds explicit
shells through n=20, chooses nmax=15 as the default and retains the original
4-to-5 contradiction. The nmax=10 candidate failed the dilute Cloudy convergence
check (6.96%); it is not silently omitted. Independent n=30/40 reference tails
and continuum refinement provide separate controls. These finite, sampled tests
do not establish infinite-atom or atmospheric accuracy.

The original baseline scope correction and finite-comparison Decision remain
unchanged. A new bridge Evidence and FollowUp resolution connect that historical
investigation to the enlarged-model result, without declaring its old Claim true.

Two diagnostic hypotheses still have `unknown` support Assessments even though
their prose records contrary indications: their cited numerical experiments
passed, and a separate failed hypothesis criterion has not been recorded.
They are not mechanically converted to `contradicted`. Whole-atmosphere physical
validation and the observational NUTS analysis also remain `unknown`.

The [inventory](models.yaml) records source/data hashes and model boundaries.
Five earlier spectrum/line-center entries remain uncontracted, and model-boundary
reviews remain pending human review.
`mceg coverage .` therefore still exits **1**; this is an explicit registration
and review gap, not evidence that the models are physically false.
The [earlier repository review](evidence/repository_review_20261121/receipt.json)
retains its original scope. Current enlarged-atom evidence is in the
[new review](evidence/atom_convergence_20261121/review.json).
The [observer review](evidence/nlte_spectrum_20261122/review.json) records the
separate spectrum campaign and its two rejected layer counts.

## Equations and code correspondence

| Definition | Human-readable model | MCEG record |
| --- | --- | --- |
| Finite H-alpha transport and H-alpha/Ly-beta recycling | [Coupling investigation PDF](../doc/halpha_coupling.pdf) | [Coupling contract](contracts/escape.halpha_coupling.yaml) |
| Extended local atom, branch profiles and observer transfer | [NLTE spectrum PDF](../doc/nlte_spectrum.pdf) | [Spectrum contract](contracts/escape.nlte_spectrum.yaml) |
| Prescribed radiation field and its unresolved adequacy | [Radiation-closure supplement](../doc/radiation_closure.pdf) | `radiation_closure` Equation and `prescribed_field_review` Assessment in the spectrum contract |
| Enlarged hydrogen atom and shell-convergence investigation | [Atom convergence PDF](../doc/atom_convergence.pdf) | [New contract](contracts/escape.atom_convergence.yaml) |
| Parker wind, prescribed nonisothermal temperature and linear tides | [Parker wind PDF](../doc/parker_wind.pdf) | [Equations and Bindings](contracts/escape.parker_wind.yaml) |
| Local six-state NLTE and H-alpha branch weights | [Local NLTE PDF](../doc/hydrogen_nlte.pdf) | [Historical local contract](contracts/escape.hydrogen_nlte.yaml) |
| Radiation, thermal balance and finite reactive flow | [Coupled-model PDF](../doc/coupled_model.pdf) | [Coupled contract](contracts/escape.coupled_hydrogen.yaml) |
| Connected radiation/SE, thermal flow and variable-mu wind | [Atmosphere PDF](../doc/coupled_atmosphere.pdf) | [Eight Equations, five helper Bindings and scoped Assessments](contracts/escape.coupled_atmosphere.yaml) |
| Solar composition, common radiation, dense lower boundary and thermal wind | [Solar model PDF](../doc/solar_atmosphere.pdf), [line-transfer supplement](../doc/escape_probability.pdf) | [Solar contract](contracts/escape.solar_atmosphere.yaml) and six [use requirements](uses/solar_proxy.yaml) |
| Repaired atomic rates and the Cloudy investigation | [Investigation PDF](../doc/cloudy_investigation.pdf) | [Investigation contract](contracts/escape.cloudy_investigation.yaml) |

The wind contract links momentum balance, the regular sonic point, the numerical
residual and implicit derivatives to the existing Python functions. Linear tides
enter the effective inward gravity as `GM/r^2 - tidal_coefficient_s2 * r`.
Temperature is supplied by the caller; this is distinct from the finite-flow
model that solves an energy equation.

Equation and Binding pins identify reviewed definitions and correspondences;
they do not prove mathematical equivalence. The historical PDFs and generated
appendices retain their original bytes and pins. In particular, the investigation
PDF's discussion of a future negative status describes the earlier MCEG reader;
the current vocabulary is documented above. Rebuilding an old document must not
silently renew its historical scientific review.

## Authoring environment and inspection

Use MCEG revision `00c219df4846fa92b71178310217184147a49cfd` from
[MCEG #27](https://github.com/HajimeKawahara/mceg/pull/27), or a compatible later
reader. This adds purpose-scoped `readiness` alongside `--require-handling` and
the three Assessment states. Package and schema version numbers are independent.

```bash
python -m venv /tmp/escape-mceg
/tmp/escape-mceg/bin/python -m pip install 'mceg @ git+https://github.com/HajimeKawahara/mceg.git@00c219df4846fa92b71178310217184147a49cfd'
source /tmp/escape-mceg/bin/activate
```

From this provider checkout:

```bash
python -m mceg check
python -m mceg coverage .
python -m mceg explain mceg/contracts/escape.cloudy_investigation.yaml --claim current_atom_truncation --investigation
python -m mceg equations mceg/contracts/escape.parker_wind.yaml --output doc/parker_wind_equations.tex --check
python -m mceg check mceg/contracts/escape.nlte_spectrum.yaml --assessment resolution_review --root .
python -m mceg check mceg/contracts/escape.nlte_spectrum.yaml --assessment gradient_review --root .
python -m mceg check mceg/contracts/escape.nlte_spectrum.yaml --decision observer_layer_count
```

The observer appendix is the historical definition at `1f4968f`; adding the
radiation-closure Equation does not rewrite that evidence. See
[document build instructions](../doc/README.md) for checking the historical
appendix and building the separate radiation supplement.

### Keep known questions visible

The connected atmosphere has a separate scientific-use check:

```bash
python -m mceg readiness mceg/uses/yapeng_atmosphere.yaml --root .
python -m mceg check mceg/contracts/escape.coupled_atmosphere.yaml --assessment reference_agreement_review --require-handling --root .
```

The manifest selects the actual spectrum entry point and pins required
Assessments for the same Subject/Context. It checks implementation evidence
separately from Cloudy matching, composition, line redistribution/two-photon
reabsorption, the lower atmosphere, resolution and observation processing.
Source and artifact integrity must also pass. A declared helper Binding is a
static correspondence; it is not proof that the selected spectrum calls it.

| Recorded support | A scoped FollowUp handles the concern | Scientific use ready |
| --- | --- | --- |
| `unknown` | Yes | No |
| `contradicted` | Yes | No |
| Every requirement `supported` | Not a substitute for evidence | Only if scope and all integrity checks also pass |

For the current atmosphere, readiness exits **1** as intended. The solar-target
Decision records the user's composition choice; it does not waive missing
physics or convert the contradicted pure-H reduction into support.

Before interpreting observer-model fits, run the external handling audit:

```bash
python tools/review_nlte_spectrum.py --python /path/to/mceg-environment/bin/python
python -m mceg explain mceg/contracts/escape.nlte_spectrum.yaml --assessment prescribed_field_review --investigation
```

The audit requires MCEG's `--require-handling` feature. It checks
`prescribed_field_review` and `atmospheric_review` explicitly, including source
and artifact hashes. An older MCEG reader, a missing record, or a failed check
causes a nonzero exit; there is no fallback that silently skips this review.
Both questions currently remain `unknown`, with open Assessment-targeted
FollowUps. The radiation-field Decision permits exploratory use while keeping
physical inference unvalidated. The broader FollowUp retains observation
processing and other atmospheric gaps.

A successful handling audit means that a response to each selected question is
recorded. It does **not** establish acceptable physical error, complete the
FollowUps, or approve an observational NUTS analysis. These checks are outside
`src/escape`, and MCEG remains absent from runtime and ordinary test dependencies.
Newly recognized physical concerns must still be registered as scoped Claims
and Assessments; MCEG cannot discover omissions from assumption prose.

The [recorded handling regression](evidence/radiation_review_20261123/handling_check.json)
checks the original missing response, both added responses, and a temporary
copy with the radiation response removed. Only the missing-response cases fail
`handling_coverage`; scientific support remains `unknown` throughout. This is
a test of tracking behavior, not a new radiation calculation.

Selecting a `contradicted` Assessment makes its `recorded_support` check **FAIL**
and the command exit **1**. Inspect the named checks: that expected scientific
outcome is distinct from a broken artifact hash or stale Equation pin.
Without an Assessment selector, `check` lists historical judgments without
choosing one as the implicit current state. `UNKNOWN` alone may exit zero.
For the wind, incomplete external source traces and static inspection of JAX
decorators remain `UNKNOWN`; these are separate from the recorded test outcome.

The observer connection changes whole-file source hashes in the shared wind
and atomic modules. The wind and enlarged-atom reviews keep their existing pins;
replay them against the preceding checkout, not the changed source:

```bash
git worktree add --detach /tmp/escape-atom-reviewed b124a5dc54bc44573e0c354ef92eed304f91f738
python -m mceg check mceg/contracts/escape.parker_wind.yaml --assessment numerical_review --root /tmp/escape-atom-reviewed
python -m mceg check mceg/contracts/escape.atom_convergence.yaml --assessment convergence_review --root /tmp/escape-atom-reviewed
python -m mceg check mceg/contracts/escape.atom_convergence.yaml --assessment reference_review --root /tmp/escape-atom-reviewed
```

The inventory hashes describe the currently inspected files and do not renew
historical evidence. The earlier investigation's `current_numerical_review` and reference/truncation
reviews now belong to historical source at `719ba25`, before the shell extension.
The older local/coupled reviews likewise belong to their recorded Subjects. To inspect the original coupled source, use its historical
checkout instead of refreshing old pins:

```bash
git worktree add --detach /tmp/escape-reviewed 6df616b
python -m mceg check mceg/contracts/escape.hydrogen_nlte.yaml --assessment numerical_review --root /tmp/escape-reviewed
```

The observer's `coarse_resolution_review` uses source `79ffd569001ee238469f3e32bb432e6bb2fa63f0`
and `intermediate_resolution_review` uses `b53e0a54b15996716a6a0e9f8bdad9ecfdc68286`.
To replay either, copy the new `mceg/evidence/nlte_spectrum_20261122/` directory
into the same relative location in its historical checkout, then check the
current spectrum contract with that checkout as `--root`. Only the expected
`recorded_support` check fails; source and artifact integrity checks must pass.

The new contradicted reviews in the coupled contract also cite
`mceg/evidence/assessment_update_20261121/review.json`. To check those reviews
against the historical root, copy this one new review artifact into the same
relative path there. Preserve all historical source files and evidence.

The [historical replay](evidence/historical_replay_20261121/checks.json) and
[pre-optimization replay](evidence/historical_repaired_replay_20261121/check.json)
retain their exact checked scopes. The earlier
[completion result](evidence/cloudy_completion_review_20261121/readiness_after_review.json)
records readiness of the finite-atom comparison at that time; it is not a new
approval of every current follow-up or physical model.

## Evidence and repeatable verification

The observer-spectrum tool runs independently of MCEG:

```bash
JAX_PLATFORMS=cpu python tools/verify_nlte_spectrum.py --layers 128 --record results/nlte-check-001
```

The recorded 128-layer preset passes all seven cases, with maximum changes of
0.06465 supplied uncertainties among the independent refinements and 0.04886
in the joint control. The 32- and 64-layer failures remain explicit contradictions;
the [layer-count Decision](contracts/escape.nlte_spectrum.yaml) retains their role
in choosing 128 layers. The tool checks complete-spectrum forward/reverse derivatives against finite
differences, separates first-call and warmed evaluation time, and refines
the atom size, layer count, disk quadrature, wavelength grid, continuum
quadrature and Parker grid independently. At each of seven declared conditions,
it also combines all six refinements in a joint control. In each comparison,
the maximum change at the 87 supplied observation wavelengths must be below
0.1 times the corresponding supplied uncertainty. This numerical scale does
not validate the uncertainties as a likelihood or guarantee a continuous
inference domain. Every attempt retains source hashes, environment, output and
the actual worker exit status in a new directory. The runner does not assign
scientific Assessments. The current example retains atmospheric profile plots
and an observation overlay; its optional synthetic NUTS smoke has not been run
as part of this update.
The [compact metrics](evidence/nlte_spectrum_20261122/metrics.json) retain each
case's gate outcomes and the [execution record](evidence/nlte_spectrum_20261122/execution.json)
retains actual exit codes and input identities. Full logs, source manifests,
spectral arrays, XML and figures live in the separately pinned
[raw archive](evidence/nlte_spectrum_20261122/archive.json).

The new numerical gates retain the original failed `atom_truncation_accuracy`
gate and add separate enlarged-atom gates. Run the new verification with:

```bash
python tools/verify_coupled.py --gate enlarged_atom_convergence \
  --pythonpath /path/to/exojax/src --record results/atom-check-001
python tools/diagnose_atom_convergence.py --output results/atom-budget-001.json
```

The second command reconstructs the fixed-density Schur-complement attribution
and then solves charge conservation independently. The [new case manifest](investigations/atom_convergence_cases.json)
reuses the seven original unique conditions at nmax=15 and 20; its original 5%
criterion is unchanged. The [archive record](evidence/atom_convergence_20261121/archive.json)
identifies full raw executions stored in the separate evidence tag, including
failed intermediate attempts. Only compact records are kept in this branch.

The [wind execution](evidence/parker_wind_20261121/receipt.json) records the new
Parker/nonisothermal/tidal test run. The original
[local NLTE receipt](evidence/hydrogen_nlte_20261120/receipt.json) records 49
targeted tests. Its [example report](evidence/hydrogen_nlte_20261120/example.json)
also records a 32-warmup/32-sample synthetic NUTS smoke run: an execution check,
not a posterior-convergence claim.

The repaired [numerical receipt](evidence/coupled_repair_20261121_attempt03/receipt.json)
passes ten numerical gates and the baseline reference gate; the separate
atom-truncation gate fails. The [combined record](evidence/combined_verification_20261121/receipt.json)
accounts for 205 distinct tests (206 executions). These historical executions
are not silently reassigned to newly changed code. The
[optimized campaign audit](evidence/cloudy_campaign_audit_20261121_optimized/audit.json)
retains the eight-case comparison and reference-refinement results.

Full raw outputs and repeated source snapshots live in the immutable
`evidence/cloudy-20261121` tag. Follow the [retention policy](evidence/README.md)
and [restoration instructions](../doc/cloudy_reference.md) before replaying or
auditing those campaigns. Restore into a fresh ignored directory:

```bash
mkdir -p results/evidence-20261121
git archive evidence/cloudy-20261121 mceg/evidence | tar -x -C results/evidence-20261121
python tools/verify_cloudy_cases.py run --reference-batch results/evidence-20261121/mceg/evidence/cloudy_repair_cases_20261121_attempt02 --resolution .025 --record results/cloudy-replay-001
```

For a fresh external calculation, supply Cloudy explicitly as described in the
reproduction guide. [The case manifest](investigations/cloudy_validation_cases.json)
and [protocol](investigations/cloudy_discrepancy.json) define the original criteria.
Do not change a tolerance to turn a failed scientific claim into a pass.

The provider-owned [gate workflow](workflows/coupled.json) and
[runner](../tools/verify_coupled.py) execute tests and retain receipts. They never
edit Assessments, Decisions or definition pins:

```bash
python tools/verify_coupled.py --gate atom_truncation_accuracy
python tools/verify_coupled.py --gate atom_truncation_accuracy --pythonpath /path/to/exojax/src --record results/coupled-attempt-001
```

The first command displays the plan; the second runs prerequisite gates and the
criterion into a new directory. Numerical solving and physical adequacy remain
separate. A measured truncation failure can coexist with numerical conservation
and agreement with an equally truncated Cloudy atom.

## Maintaining the records

Preserve earlier Claims, Equations, Evidence and Assessments when revising their
meaning. Add a new scoped review and, when a choice is reconsidered, a new
Decision revision with `supersedes`. The original basis hashes should still
match the original records. Source drift or a changed definition requires review,
not a hash refresh presented as new scientific evidence.

Keep working outputs under ignored `results/` and retain only the compact
receipts and artifacts required by the reviewed claims. Contracts, documents
and compact evidence are included in source distributions, not wheels.
Installed-package discovery (`--package`) is not configured.

MCEG's native investigation view connects hypotheses, experiments, choices and
follow-ups, but it has no generic numerical `run` API or scheduler.
`FollowUp.requires` remains human-readable text. Execution policy and model
acceptance still require explicit scopes and criteria. Clear inventory
`review_pending` only after a maintainer reviews model boundaries and links.
