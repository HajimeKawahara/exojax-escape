# Repository instructions

## Working style

- Communicate in Japanese. Write code, documentation, commit messages and PR descriptions in English.
- Prefer the smallest coherent implementation and reviewable diff. Permission to investigate does not imply permission to ship every experimental artifact.

## Experimental work

- For an unadopted or unsuccessful trial whose deliverable is a record, default to two files: one concise result report and one compact historical MCEG investigation record.
- Keep scratch scripts, logs, checkpoints, intermediate states, generated figures/PDFs, environment dumps and repeated validation receipts outside the tracked tree, such as in `/tmp` or an existing ignored output directory.
- Remove experiment-only runtime changes, drivers and tests from the proposed change when the implementation is not adopted. Retain independently useful fixes or reusable functionality only when they belong to the requested deliverable.
- Preserve the tested conditions, outcome, relevant verification, limitations and provenance in the concise record. Link existing immutable archives when available; do not first commit a large output collection merely to create an archive.
- Keep MCEG proportional to the deliverable. Historical numerical success, failed fit improvement and unknown physical adequacy are different judgments. Do not register removed experimental implementations as current models or imply that historical checks qualify current code.
- Two files is a default for record-only trials, not a universal limit. Preserve necessary correctness fixes, existing contract consistency and explicitly requested reproducibility requirements.

## Before committing or updating a PR

- Stage intended paths explicitly. Inspect the final file list, diff statistics and content against the actual PR base, including the updated base of a stacked PR.
- Remove generated outputs, duplicate records, temporary helpers and unrelated formatting changes from the diff. Keep tests that verify retained functionality; do not add tests solely to memorialize a discarded experiment.
- Preserve unrelated user changes and existing history. Never alter scientific outcomes or evidence references merely to make checks pass.
- For record-only cleanup, verify references, report hashes and tree differences. Do not rerun expensive scientific calculations just to regenerate discarded artifacts.
