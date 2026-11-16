# Model contracts and evidence

These provider-owned records use [MCEG](https://github.com/HajimeKawahara/mceg)
to track model assumptions, evidence and unresolved scientific questions.
`escape` does not import or depend on MCEG. Its installation, numerical tests
and examples work without MCEG; use a separate environment for record inspection.

## Initial scope

- [Inventory](models.yaml): all files in `src/escape`, with inspected file hashes
  and known model choices. Examples, observations and external dependencies are
  outside the inventory scan.
- [Population contract](contracts/escape.hydrogen_populations.yaml): pure-H
  continuity, diluted blackbody occupation and isolated two-level balance.
- [Test receipt](evidence/hydrogen_populations_20261117/receipt.json): the actual
  execution of `tests/test_populations.py`, with logs, environment and hashes.

The numerical assessment is limited to those recorded tests. Physical accuracy
for an escaping atmosphere remains unknown. In particular, this is not an
ionization or multilevel NLTE solver, and it does not solve a trapped radiation
field. Other wind, spectrum and line-center choices remain explicitly
uncontracted. All model boundaries and contract links await human review.

## Inspect the records

Follow MCEG's [setup guide](https://github.com/HajimeKawahara/mceg/blob/main/docs/en/initialization.md).
This adoption was checked with MCEG commit
`fff7afe36f36627e002d27a953a53344ab22bad3` (contract schema `0.2`, inventory `0.1`).
For an isolated companion environment:

```bash
python -m venv /tmp/escape-mceg
/tmp/escape-mceg/bin/python -m pip install \
  'mceg @ git+https://github.com/HajimeKawahara/mceg.git@fff7afe36f36627e002d27a953a53344ab22bad3'
```

From the `exojax-escape` checkout:

```bash
/tmp/escape-mceg/bin/python -m mceg check
/tmp/escape-mceg/bin/python -m mceg explain
/tmp/escape-mceg/bin/python -m mceg coverage .
/tmp/escape-mceg/bin/python -m mceg check \
  mceg/contracts/escape.hydrogen_populations.yaml \
  --assessment numerical_review --root .
```

`check` validates records and can report `UNKNOWN` while exiting zero. The
selected assessment also checks its pinned local files and artifacts; it does
not rerun tests. `coverage` currently exits **1**, reporting pending human
reviews and known uncontracted models. These gaps are intentional records of
remaining work, not evidence of physical validity or invalidity.

The contracts and evidence are included in source distributions for their
snapshot test, but not in wheels. Installed-package discovery (`--package`)
is not configured by this initial adoption.

The scientific checks and snapshot consistency check use the provider's normal
test environment, without MCEG:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/test_populations.py tests/test_model_evidence.py
```

## Maintain the evidence

Review changed code, data and model choices before updating inventory hashes.
For new numerical evidence, rerun the relevant tests, retain their receipt and
artifacts, and add a Subject, Context and Assessment for the tested revision.
Preserve earlier evidence in version history; refreshing hashes alone does not
renew an old result. Keep physical support separate from numerical agreement.
Clear `review_pending` only after a human reviews the boundary and contract links.

The planned hydrogen NLTE work adds atomic rates and level/ionization balance,
coupled radiation, variable-mu winds, continuum opacity and eventually thermal
balance. Register each implemented model and its actual validation as it lands.
Cloudy comparisons, grid/level convergence and derivative checks must supply
their own evidence; this initial record does not establish those future claims.
