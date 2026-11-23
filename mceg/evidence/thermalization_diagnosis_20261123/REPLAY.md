# Deep thermalization and upper photon escape

This is research evidence for Issue #10, not a runtime implementation or a
repaired spectrum. The provider source snapshot is
`1e4ad8456ef0876a4c377df7cf37e82ca7e3e050`; ExoJAX was
`38cb2e44f153ad6d729e3253e235702de52920cf`. Figures and this evidence tag are
outside the normal code PR.

Extract `analysis.tar.gz` at the provider checkout root. It creates
`results/darkening_diagnosis/` with executed scripts, numerical outputs,
literature/code audit notes, and the saved stellar-comparison population
arrays. `manifest.json` identifies every archived file. No full PHOENIX
stellar table is redistributed. Retrieve the original FITS files with
`examples/fetch_phoenix.py`; `stellar_metadata.json` retains their identity.

The main conclusions are in `notes/physics.md`, `rates/findings.md`, and
`thermalization/interpretation.md`. The first plot uses the full atom for its
upper panels, with artificial net-rate limits below. The second uses only
the isolated H-alpha 3d-to-2p branch and frozen opacity. Neither is a new
observational fit. The zero-bracket experiment is not a transfer solution.

The research scripts preserve their original execution paths. For replay
elsewhere, remap these prefixes in a disposable copy:

| Original path | Replay destination |
| --- | --- |
| `/tmp/escape-stellar-spectrum` | Provider checkout at the commit above |
| `/tmp/escape-stellar-spectrum/results/stellar_halpha/arrays.npz` | Extracted `stellar_input_arrays.npz` |
| `/tmp/escape-stellar-sed-data` | Directory containing the retrieved stellar FITS files |
| `/tmp/escape-darkening-rates` | Extracted `rates/` directory |
| `/tmp/escape-darkening-thermalization` | Extracted `thermalization/` directory |
| `/tmp/escape-darkening-rethink` | Extracted `notes/` directory |

Apply the specific input-file mapping before the provider-prefix mapping.
For the isolated-line script's local `phoenix10000_surface_flux.npz`, use
`examples/stellar_halpha.py:load_surface`. It returns `(frequency, flux,
metadata)`; save `np.savez(path, frequency_hz=frequency,
surface_flux_nu=flux)`. This intermediate is not redistributed. The rates
script installs the existing offline NIST fixture
from `tools/verify_nlte_spectrum.py` for the observer line center.

Execution used Python 3.12.2, JAX/jaxlib 0.10.0 with CPU float64, NumPy 2.1.0,
SciPy 1.17.1, pandas 2.2.3, and Matplotlib 3.10.1. Install the provider's
example dependencies, including Astropy. Put provider `src`, pinned ExoJAX
`src`, provider `examples`, and provider `tools` on `PYTHONPATH`. Run:

```bash
python results/darkening_diagnosis/rates/net_rates.py
python results/darkening_diagnosis/rates/trapping_limits.py
python results/darkening_diagnosis/thermalization/line_scattering.py
python results/darkening_diagnosis/notes/plot_diagnosis.py
```

`line_scattering.py` now runs all four retained cases and refinements.
`run.log` records the initial four-case execution; `refinement.log` records
the additional blackbody frequency/depth refinements. The final metrics
combine these executions. The depth metric averages brightness temperatures
of subdivided cells; it is not an average intensity. Profile-tail coverage
and the absence of a full wing-convergence study remain explicit.

Use external MCEG revision `e45c2b57042bdedb3effa60914cac08c46a34af7` or a
compatible later reader. With this evidence directory and expanded archive
in the checkout, check each Assessment using the exact IDs in
`investigation.yaml`, for example:

```bash
python -m mceg check mceg/evidence/thermalization_diagnosis_20261123/investigation.yaml --assessment physical_review --require-handling --root .
```

Numerical support applies only to the declared diagnostics. Physical repair
remains unknown; the new FollowUp does not resolve the provider's existing
radiation-feedback or observation-processing obligations. No new Cloudy run
was made. MCEG remains external to `escape` at runtime.
