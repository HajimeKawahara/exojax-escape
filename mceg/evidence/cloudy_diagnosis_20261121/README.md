# Cloudy discrepancy investigation records

These files retain experiments, including failures. They do not by themselves
approve a scientific model or replace the declared comparison criteria.

| Record | Purpose |
| --- | --- |
| `historical_raw/` | Original instrumented baseline, native all-mixing-off run and rejected two-photon command. |
| `baseline/`, `no_mixing/` | Fresh executions through the reusable external driver, with command, binary hash, input, protocol and receipts. |
| `rates_T*_n*/` | Twelve actual Cloudy operator readouts spanning temperature and density. |
| `historical_analysis/` | Byte-exact first attribution, dilute-case diagnosis, n=2/n=3 rate checks, scripts and logs. |
| `legacy_rates.json`, `legacy_source/` | Immutable pre-repair rate arrays and source files matching the original recorded hashes. |
| `attribution_replay/` | New execution of the portable diagnostic on retained raw outputs and immutable legacy rates. |
| `historical_bound_free/` | Original paired photoionization/recombination intervention and prototype snapshots. |
| `final_residual/` | Remaining differences after the physical repairs, with exact comparator arrays. |
| `instrumentation_provenance.json` | Official archive identity, both diagnostic patches and observed instrumented binary/source hashes. |

The original scripts retain their historical path assumptions. Use the portable
tools documented in `doc/cloudy_reference.md` for new executions. Snapshot
manifests record archival time and original byte hashes; they do not invent
execution provenance that the original experiment did not capture.

In the first historical attribution, `self_consistent_ne` means charge closure
with coefficients frozen at the reference state. The portable result calls this
`frozen_coefficients_charge_closure`. It does not recompute density-dependent
Cloudy physics. In `final_residual`, the inherited result key `legacy` denotes
the current repaired comparator; the accompanying rate arrays remove ambiguity.

The portable replay first reconstructs the actual Cloudy destruction matrix,
source vector and solution independently. Only then does it exchange paired
rate blocks. Reproducing a population discrepancy by replacing a block explains
a cause; production repairs still need the independent, held-out campaign and
numerical/gradient verification.

Raw `.out.gz` files are deterministic, lossless gzip archives. Each archive
manifest records both original and stored hashes. The initial diagnostic
archives explicitly omit large auxiliary continuum tables (their hashes and
reasons remain listed); all operator readouts are retained. The separate
eight-case comparison campaigns retain every original artifact, including their
large continuum tables. No old execution receipt is rewritten to mark a failed
attempt successful.
