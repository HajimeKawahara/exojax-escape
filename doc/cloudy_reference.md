# Reproduce and investigate the Cloudy comparison

`tools/cloudy_reference.py` runs an explicitly supplied Cloudy executable in a
new directory, saves its input and raw outputs, and normalizes the hydrogen
populations. It is external verification tooling: neither Cloudy nor MCEG is a
runtime dependency of `escape`. The script never downloads or modifies Cloudy.

## Restore the historical evidence when needed

The current tree keeps the receipts, review summaries and artifacts directly
referenced by MCEG. Full raw campaigns, repeated input snapshots and intermediate
diagnostics are preserved at Git tag `evidence/cloudy-20261121`, commit
`c75bc5b819febcf1b3bed232ccebd4a889b6d829`. See the
[retention policy](../mceg/evidence/README.md) for the separation.

For a full reference audit or replay, restore the original evidence paths under
ignored `results/` from a repository checkout. Use a fresh destination:

```bash
mkdir -p results/evidence-20261121
git archive evidence/cloudy-20261121 mceg/evidence | tar -x -C results/evidence-20261121
```

This reads local Git objects and makes no network request. If the tag is missing,
fetch it explicitly with `git fetch origin tag evidence/cloudy-20261121`.
Source distributions contain the compact review records, not Git history;
use a versioned repository checkout for the full historical audit. The archive
preserves failed attempts, original receipt bytes and dirty-worktree source
snapshots. Removing their loose copies from the current tree does not create
new execution evidence or change their scientific outcomes.

## Prepare Cloudy C23.01

The retained reference used the official 338,434,070-byte archive, SHA256
`a9ad2dc037e88f552389de0e483d68f54310976dee32154ed13830882aedae0b`.
The following are reproduction instructions; the historical execution records
are in `mceg/evidence/cloudy_reference_20261121/`.
From a separate build directory, with `ESCAPE_REPO` pointing to this checkout:

```bash
curl -L https://data.nublado.org/cloudy_releases/c23/c23.01.tar.gz -o c23.01.tar.gz
echo 'a9ad2dc037e88f552389de0e483d68f54310976dee32154ed13830882aedae0b  c23.01.tar.gz' | sha256sum -c -
tar -xzf c23.01.tar.gz
patch -d c23.01 -p1 < "$ESCAPE_REPO/mceg/evidence/cloudy_reference_20261121/diagnostic_only.patch"
make -C c23.01/source -j4
```

The patch guards diagnostic accesses to absent resolved n=4 states and exposes
the small collapsed atom in line reports. It changes no rates or solution
equations. Its original failures, exact diff and license are retained. Cloudy's
data installation must be available to the executable. A separately instrumented
diagnostic binary must have its own source/patch provenance; the driver records
the actual executable hash but does not infer its source from that hash.

## Execute a fresh comparison

From the escape checkout, with its dependencies installed:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python tools/cloudy_reference.py \
  --executable /path/to/c23.01/source/cloudy.exe \
  --protocol mceg/investigations/cloudy_discrepancy.json \
  --record results/cloudy-baseline-001 --compare-escape
```

Defaults reproduce a pure-H, fixed-temperature, optically thin one-zone model:
T=9000 K, nH=1e12 cm^-3, a 10000 K blackbody diluted by W=0.025,
thickness 1e-4 cm, and continuum resolution scale 0.05. The six states through
n=3 are resolved; n=4,5 are collapsed, top-off and molecules are disabled.
New default inputs omit the unused full continuum optical-depth dump.
`--nmax 4`, `--thickness-cm 5e-5`, or `--resolution .025` give separate atom-size,
thickness and continuum-refinement checks. Each invocation needs a new directory.

`--compare-escape` evaluates the current finite atom on a separate continuum
quadrature and records absolute relative differences in n2 and n3. The default
5% tolerance is written before execution. Cloudy execution and model agreement
have separate statuses: a completed run with failed agreement exits nonzero and
keeps its outputs. The driver does not loosen tolerances or update Assessments.
The implemented escape atom currently supports nmax=3,4,5; Cloudy alone can be
run with larger collapsed atoms by omitting `--compare-escape`.

The receipt retains the actual command, executable hash, input configuration,
protocol copy/hash, Python/package information, source hashes before and after
execution, parsed results and raw artifact hashes. Saved `e-` and `H+` densities
avoid deriving charge solely from the lower-precision hydrogen-conditions table.
Malformed, multi-zone, unfinished or failed Cloudy outputs cannot become a
successful reference. Imported atomic sources and executable/data provenance
still require the explicit scientific review recorded in MCEG.

New runs retain large outputs as deterministic gzip files. Receipt artifact hashes
refer to the original uncompressed bytes; `artifact_storage` also records the
stored path and its hash. `tools/cloudy_archive.py` can archive older runs without
altering their receipt bytes. Its default retains every file; an explicit size
limit lists each omitted auxiliary file and its original hash. Keep fresh run
directories under ignored `results/`; review which records are needed before
adding evidence to Git. The historical campaigns require the restoration above
to make all their raw artifacts available locally.

To compare a changed escape implementation with an already completed reference,
replace `--executable` with `--reference /path/to/previous/run`, keeping all
physical options identical. This verifies every original artifact hash, copies
the original receipt and raw outputs, and reparses the Cloudy results. It then
runs the current escape implementation in a new directory. It never reuses the
old escape comparison or its support labels. A previous escape source-identity
failure does not invalidate independently verified Cloudy bytes.

## Controlled process experiments

Repeat `--command` for native Cloudy switches selected by an investigation. For
example, the C23 input `database H-like collisions l-mixing off` disables all
same-n mixing, including n=2. It is not an isolated n=3 experiment. Every extra
command is retained verbatim; a changed model must not be described as an
identical-physics comparison. More selective rate-block swaps belong in the
investigation's separate diagnostic driver and evidence.

For selective interventions, build a separate Cloudy copy with the reporting
patch above, followed by the read-only operator patch:

```bash
patch -d c23.01 -p1 < "$ESCAPE_REPO/tools/cloudy/rate_readout.patch"
make -C c23.01/source -j4
```

This prints the actual local rate matrix, source vector, process coefficients
and solved populations. It changes no rate or convergence condition. Run this
binary through `cloudy_reference.py` as above, then reconstruct its final
operator and replace selected paired forward/inverse rate blocks:

```bash
python tools/cloudy_rates_diagnosis.py \
  --cloudy-output results/cloudy-instrumented-001/model.out.gz \
  --legacy-rates mceg/evidence/cloudy_diagnosis_20261121/attribution_replay/legacy_rates.json \
  --protocol mceg/investigations/cloudy_discrepancy.json \
  --record results/cloudy-attribution-001
```

The diagnostic first requires its independent matrix, source and linear solve
to reproduce the saved Cloudy operator. It then compares fixed-electron-density
interventions and a separate pure-H charge closure with **frozen coefficients**.
The latter does not re-evaluate density-dependent mixing at each trial density;
it is an attribution experiment, not the production population solver. Saved
legacy rates keep the pre-repair baseline reproducible after defaults change.
The optional `--no-mixing-output` retains a native all-mixing-off control.

Historical scripts and numerical arrays remain byte-exact in the evidence tag.
After restoration, they are under
`results/evidence-20261121/mceg/evidence/cloudy_diagnosis_20261121/historical_analysis/`.
These snapshots keep their original path assumptions and terminology. The portable
`attribution_replay/` record labels the frozen-coefficient calculation explicitly
and independently reproduces those interventions. The exact old escape source
hashes are linked in `legacy_source/provenance.json`. The two mixing-grid studies
separately test n=2 and n=3 against actual Cloudy proton coefficients; differences
between Cloudy's cutoff shortcut and the continuous runtime integral are reported,
not absorbed by a fitted correction.

## Run the declared validation cases

`tools/verify_cloudy_cases.py` attempts every case in the registered manifest,
including cases after a disagreement. For example:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python tools/verify_cloudy_cases.py run \
  --executable /path/to/c23.01/source/cloudy.exe \
  --cases mceg/investigations/cloudy_validation_cases.json \
  --record results/cloudy-cases-001 --resolution .025
```

Use a separate coarser batch to check reference refinement. A fresh readiness
audit verifies coverage, original artifact hashes, source identities, numerical
gates, reference refinement and unresolved investigation actions. It does not
trust a saved `passed` label or turn numerical agreement into a scientific
Assessment. Failed and superseded attempts remain counterevidence.

After restoring the archived campaign, audit the optimized comparison against
its coarse reference control and the retained numerical receipt:

```bash
python tools/verify_cloudy_cases.py readiness \
  --batch results/evidence-20261121/mceg/evidence/cloudy_repair_cases_20261121_attempt03 \
  --reference-control results/evidence-20261121/mceg/evidence/cloudy_repair_cases_20261121_attempt01 \
  --verification results/evidence-20261121/mceg/evidence/coupled_repair_20261121_attempt03/receipt.json
```

The full raw hashes are still required. Missing files are verification failures;
the summary receipts alone cannot pass a raw-data audit. For a new implementation,
record fresh comparisons and numerical verification before assessing readiness.

The earlier one-off collector was not a reusable repair loop. This driver makes
execution reproducible; the [investigation protocol](../mceg/investigations/cloudy_discrepancy.json)
defines hypotheses, causal comparisons and the conditions for continuing or
closing the investigation. A population discrepancy is a trigger for that
process, not a successful end state.
