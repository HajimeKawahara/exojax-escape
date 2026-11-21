# Model contracts and evidence

These provider-owned records use [MCEG](https://github.com/HajimeKawahara/mceg)
to track definitions, model choices and scoped evidence. `escape` does not import
or depend on MCEG. Keep MCEG in a separate authoring environment; the scientific
package and its ordinary tests do not need it.

## Current Cloudy investigation

The [investigation contract](contracts/escape.cloudy_investigation.yaml) records
why the first comparison failed, the experiments that isolated the responsible
rates, and the physical choices adopted. The [mathematical document](../doc/cloudy_investigation.pdf)
explains the rate equations and their code bindings. The
[reference instructions](../doc/cloudy_reference.md) cover external execution,
rate diagnostics and retained-output replay.

The original n3 discrepancy was dominated by missing n=3 orbital mixing.
A held-out dilute case then exposed a different n=2 mixing closure. Both were
isolated with paired forward/reverse interventions. The final eight-case
comparison has a maximum n2/n3 error of **1.100%**, below the unchanged **5%**
criterion. All matched Cloudy refinements change populations by less than
**0.049%**, below the declared **1%** control. The
[current audit](evidence/cloudy_campaign_audit_20261121_optimized/audit.json)
verifies the optimized-source replay and its raw reference bytes. The
[earlier audit](evidence/cloudy_campaign_audit_20261121/audit.json) retains the
original gzip bookkeeping failure. The execution-only batching optimization
changes all compared populations by at most 2.20e-14 relative; old and optimized
implementations have distinct Subjects and Bindings.
The [current numerical receipt](evidence/coupled_repair_20261121_attempt03/receipt.json)
passes all ten numerical gates, and the separate baseline reference gate passes.
The [combined test record](evidence/combined_verification_20261121/receipt.json)
accounts for all **205 distinct collected tests**, with 206 executions because
one test is repeated. The [completion review](evidence/cloudy_completion_review_20261121/checks.json)
records current scoped Assessments and no unresolved required FollowUps. Infinite-atom convergence and planetary accuracy
are separate claims; some individual higher-shell collision rates still differ
substantially even when these summed populations agree. Increasing the maximum
shell from 4 to 5 still changes n3 by about 7.74% in escape and 7.58% in Cloudy,
above the separate 5% truncation target.

The [frozen protocol](investigations/cloudy_discrepancy.json) and
[eight-case manifest](investigations/cloudy_validation_cases.json) precede the
new experiment outcomes. The [n2 follow-up](investigations/cloudy_n2_followup.json)
explicitly records its later chronology. The provider loop is:

1. Retain a failed comparison and select a discriminating hypothesis.
2. Change one paired rate block, repeat with a reverse intervention, and save
   actual outputs and execution identity.
3. Record a sourced Decision, implement its equations, and verify numerics and
   derivatives.
4. Repeat every unchanged validation case and reference refinement. A failure
   reopens diagnosis; it does not end the implementation task.

MCEG already has Evidence, scoped Assessments, Decisions, FollowUps and native
investigation navigation. The earlier stop was the agent/provider's finite-stage
choice, not a restriction imposed by MCEG. `FollowUp.requires` is human-readable
instruction, not an executable dependency graph. This repository's external
runner supplies execution and readiness policy without importing MCEG into
`escape`.

```bash
python -m mceg explain mceg/contracts/escape.cloudy_investigation.yaml \
  --claim repaired_reference_agreement --investigation
python tools/verify_cloudy_cases.py run --executable /path/to/cloudy.exe \
  --resolution .025 --record results/cloudy-fine-001
python tools/verify_cloudy_cases.py run \
  --reference-batch mceg/evidence/cloudy_repair_cases_20261121_attempt02 \
  --resolution .025 --record results/cloudy-replay-001
python tools/verify_cloudy_cases.py readiness \
  --batch mceg/evidence/cloudy_repair_cases_20261121_attempt03 \
  --reference-control mceg/evidence/cloudy_repair_cases_20261121_attempt01 \
  --verification /path/to/current-numerical-receipt.json
```

The final current-source reviews can be audited directly:

```bash
python -m mceg check mceg/contracts/escape.cloudy_investigation.yaml \
  --assessment repaired_reference_agreement_review --root .
python -m mceg check mceg/contracts/escape.cloudy_investigation.yaml \
  --assessment current_numerical_review --root .
python -m mceg check mceg/contracts/escape.cloudy_investigation.yaml \
  --assessment current_truncation_review --root .
```

The last Assessment intentionally remains `unknown` with failed Evidence:
the measured 7.735% shell sensitivity exceeds 5%. Its separate adequacy failure
is not hidden by the passing finite-atom comparison.

Readiness independently verifies coverage, retained raw hashes, current core
identity, ten numerical gates, all reference refinements, and resolved native
FollowUps. It never writes a scientific Assessment. A declaration that code ran
cannot substitute for a passing comparison or for physical adequacy.

## Models and definitions

- [Historical coupled hydrogen contract](contracts/escape.coupled_hydrogen.yaml), schema
  **0.4**, and [mathematical document](../doc/coupled_model.pdf): spherical static
  radiation moments, total material energy, variable composition and finite
  reactive inflow. Static SE, static thermal equilibrium and advecting flow have
  separate equations and acceptance scopes. Transonic mass-loss selection and
  physical adequacy for KELT-9b remain separate questions.
- [Inventory](models.yaml): all packaged Python sources, inspected hashes and
  model boundaries. Examples, observations and external dependencies are outside
  its source scan; boundary reviews remain pending.
- [Local NLTE contract](contracts/escape.hydrogen_nlte.yaml), schema **0.4**:
  six-state hydrogen rates, local charge-conserving statistical equilibrium,
  implicit derivatives and H-alpha branch weights. Temperature, density and
  pumping radiation are prescribed. The contract includes **Equations**,
  function **Bindings**, scoped **Assessments** and model **Decisions**.
- [NLTE mathematical document](../doc/hydrogen_nlte.pdf), its
  [editable explanation](../doc/hydrogen_nlte.tex) and
  [generated equation appendix](../doc/hydrogen_nlte_equations.tex): adopted
  approximations, units, solver residual and code correspondence.
- [Historical two-level contract](contracts/escape.hydrogen_populations.yaml),
  schema **0.2**, [original document](../doc/model.pdf), and
  [six-test receipt](evidence/hydrogen_populations_20261117/receipt.json): retained
  as separate evidence. These records do not validate the new NLTE model.
- [New execution receipt](evidence/hydrogen_nlte_20261120/receipt.json): 49
  atomic-rate, local-equilibrium and spectrum-integration tests. The separately
  retained [example report](evidence/hydrogen_nlte_20261120/example.json) records
  a forward example and a 32-warmup/32-sample single-chain synthetic NUTS smoke
  run, with finite samples and no reported divergences.

A Decision records what was adopted and why. For this implementation the choices
are the six-state truncation, external radiation, approximate atomic rates,
restricted temperature interval and actual-population H-alpha branch sum.
Their remaining limitations and reconsideration conditions stay visible even
when physical accuracy is `unknown`.

For the preceding local-NLTE receipt, numerical support is limited to its actual
saved execution and selected files. That receipt does not establish atomic-data
accuracy, local equilibrium in an outflow, a coupled radiation/thermal solution,
a matched Cloudy comparison or a faithful KELT-9b atmosphere. The short
synthetic NUTS run checks execution through the differentiable spectrum; it
does not establish posterior convergence or accurate parameter recovery.
The complete wind/spectrum families retain their separately recorded gaps.

## Inspect the records

Schema 0.4 authoring uses MCEG source revision
`692759a77449abd006218b7c1a09cd43c43d2ff3` or a compatible later reader. The original
v0.2 adoption used `fff7afe36f36627e002d27a953a53344ab22bad3`; its meaning and
historical execution are preserved. Package version numbers and schema numbers
are different. For an isolated companion environment:

```bash
python -m venv /tmp/escape-mceg
/tmp/escape-mceg/bin/python -m pip install \
  'mceg @ git+https://github.com/HajimeKawahara/mceg.git@692759a77449abd006218b7c1a09cd43c43d2ff3'
```

The old contracts retain their exact reviewed source pins. Their source reuse
checks belong to the historical checkout; changed current source should be
reported as stale, not silently repinned. The
[retained replay](evidence/historical_replay_20261121/checks.json) checks all
19 historical evidence-support Assessments against revision `6df616b`. The
[pre-optimization comparison audit](evidence/historical_repaired_replay_20261121/check.json)
separately materializes exact `0fa8bc9` source plus its retained execution artifacts;
it does not refresh the old Subject to the optimized source.
Create an equivalent read-only inspection checkout, then use it as the root:

```bash
git worktree add --detach /tmp/escape-reviewed 6df616b
```

From the `exojax-escape` checkout, using the authoring environment:

```bash
python -m mceg check
python -m mceg coverage .
python -m mceg check mceg/contracts/escape.hydrogen_nlte.yaml \
  --assessment formulation_review --root /tmp/escape-reviewed
python -m mceg check mceg/contracts/escape.hydrogen_nlte.yaml \
  --assessment numerical_review --root /tmp/escape-reviewed
python -m mceg explain mceg/contracts/escape.hydrogen_nlte.yaml \
  --decision atom_truncation --investigation
python -m mceg equations mceg/contracts/escape.hydrogen_nlte.yaml \
  --output doc/hydrogen_nlte_equations.tex --check
```

`check` may report `UNKNOWN` while exiting zero. Assessment checks compare pinned
local files and artifacts; they do not rerun tests. Decision checks compare the
recorded basis and its declared dependencies without promoting physical support.
The equation command checks generated bytes, not mathematical equivalence.
`coverage` intentionally exits **1** while human reviews and known uncontracted
model families remain. Neither pending review nor incomplete coverage establishes
physical validity or invalidity.

The historical checks remain available:

```bash
python -m mceg check mceg/contracts/escape.hydrogen_populations.yaml \
  --assessment numerical_review --root /tmp/escape-reviewed
python -m mceg check mceg/contracts/escape.hydrogen_populations.yaml \
  --assessment formulation_review --root /tmp/escape-reviewed
```

Contracts, documents and evidence are included in source distributions, but not
wheels. Installed-package discovery (`--package`) is not configured.

## Maintain the evidence

Review changed source, atomic data and assumptions before updating inventory
hashes. For a new execution, retain its command, logs, environment, actual
outcomes and selected source/artifact hashes. Add a scoped Subject, Context and
Assessment; refreshing hashes alone does not renew an earlier result. Keep the
historical two-level contract, document and receipt unchanged.

Edit new mathematical definitions in the NLTE contract, regenerate its TeX
appendix and build `make -C doc nlte-pdf`. Inspect the resulting PDF before
recording a formulation review. Equation/Binding pins identify the reviewed
correspondence; Decision basis pins identify the reasons used for a choice.
If those premises change, retain the old choice and perform a new review.
Do not silently refresh a stale decision to imply that it was reconsidered.

The coupled-model contract keeps the new implementations separate from these
historical checks. Larger-atom adequacy, matched references, angular/moving
radiation transport and transonic selection still need their own evidence.
Clear `review_pending` only after a maintainer reviews model boundaries and
contract links.

## Historical baseline and current numerical execution

The retained [coupled receipt](evidence/coupled_20261121_attempt01/receipt.json)
records **eight passing numerical gates and two failed accuracy gates**, with
all declared inputs unchanged. It contains 57 test executions (56 distinct
cases); a pytest pass means the measurement ran correctly, not that its physical
accuracy criterion passed. The comparison data have their own
[Cloudy provenance and raw-file hashes](evidence/cloudy_reference_20261121/provenance.json).

The n=3 reservoir changes by **11.46%** from maximum shell 4 to 5, exceeding the
5% truncation criterion. Against the retained Cloudy reference, n=3 differs by
**32.65%** and **30.22%** for maximum shells 4 and 5; n=2 differs by 4.24% and
4.43%. These are known failures of the declared criteria. Their Assessments
remain `unknown` under the current schema, with failed Evidence and explicit
reasons. Formulation and numerical support cover only the recorded scopes.

Original draft Decisions are retained as revision 1. Reviewed choices are
revision 2, with explicit `supersedes` links and relevant passing or failed
Assessments in their basis. No current choice is inferred from a timestamp.

```bash
python -m mceg check mceg/contracts/escape.coupled_hydrogen.yaml \
  --assessment definition_review_assessment --root /tmp/escape-reviewed
python -m mceg check mceg/contracts/escape.coupled_hydrogen.yaml \
  --assessment reference_agreement_assessment --root /tmp/escape-reviewed
python -m mceg explain mceg/contracts/escape.coupled_hydrogen.yaml \
  --decision higher_shell_atomic_data --decision-revision 2 --investigation
```

[coupled.json](workflows/coupled.json) declares test selections, prerequisites,
metric thresholds and what each gate cannot prove. It is provider configuration,
not a new MCEG schema or an automatic scientific reviewer. The stdlib-only
[runner](../tools/verify_coupled.py) runs outside `src/escape` and imports no MCEG.
First inspect the plan, then record a new attempt:

```bash
python tools/verify_coupled.py --gate atom_truncation_accuracy
python tools/verify_coupled.py --gate atom_truncation_accuracy \
  --pythonpath /path/to/exojax/src --record results/coupled-attempt-001
```

The second command executes prerequisite gates too. Each output directory must
be new: failed attempts are retained, not overwritten. Receipts include actual
commands, exit codes, JUnit outcomes/properties, log hashes, declared source
snapshots and input hashes before and after execution. Missing or skipped checks
stay incomplete; failed prerequisites block dependent gates. Any source change
during the run invalidates its review readiness. External source revisions and
package metadata are recorded, but are not a complete environment lock.

Numerical consistency and physical adequacy have different gates. In particular,
`atom_truncation_accuracy` requires the n=3 population to change by at most 5%
between nmax=4 and 5 at the declared conditions. A numerically converged atom may
fail this criterion. Integration gates do not depend on that accuracy result;
the failure must remain visible when deciding whether the atom is adequate.
The independent reference stage compares a controlled, optically thin pure-H
fixed-temperature calculation with Cloudy C23.01. It is a finite-atom comparison
with different atomic data, not equivalent physics or a full atmosphere
benchmark. Actual disagreement is retained; no observed-spectrum validation
follows from setting up or running Cloudy.

After a run, inspect its scope, metrics, source identity and limitations. A
qualified author may then attach an immutable execution Evidence record and a
scoped Assessment. The runner never edits contracts, changes scientific support
or refreshes Equation, Binding or Decision pins. `mceg check` subsequently checks
those retained records; it does not repeat the scientific execution.

## MCEG limitations demonstrated by this workflow

MCEG currently has no generic numerical `run` command or receipt API. The provider
runner is an explicit local execution policy that executes declared tests and preserves their
inputs and outcomes. A future shared execution interface could standardize these
records; scientific support would still require an explicit, scoped review.

Schema 0.4 `evidence_support` Assessments allow only `supported` and `unknown`.
That vocabulary cannot distinguish an untested proposition from one contradicted
by a retained measurement. The failed n=3 accuracy criteria therefore use
`Evidence.result: failed`, explicit metrics and reasons, while the corresponding
Assessment remains `unknown`. An optional future negative/refuted Assessment status could
express that difference directly. No MCEG schema or runtime dependency
was changed for this implementation.

A typed prerequisite/outcome interface could make automated action selection
easier. Existing `FollowUp.requires` text is not a scheduler. A frozen execution
checkout is also preferable to before/after source hashes: an edit followed by
a revert can evade endpoint detection. The exploratory first campaign retains
that limitation, and the final physical campaign used a frozen source. These
execution-policy issues do not prevent using the current MCEG schema for the
hypothesis, experiment, Decision and repeated-verification loop.
