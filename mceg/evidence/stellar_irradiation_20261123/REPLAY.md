# Stellar-boundary experiment

Extract `analysis.tar.gz` into a disposable provider checkout at
`5898d686202fb83f5bce66a0d44e7f2de8d080a5`. It includes the proposed scientific
source snapshots and results, identified by the manifest hashes. The final
provider MCEG contract lives in the implementation PR, not inside this archive.
ExoJAX source was `38cb2e44f153ad6d729e3253e235702de52920cf`.

Install the example dependencies (Matplotlib, SciPy, Astropy). Execution used
Python 3.12.2, JAX/jaxlib 0.10.0, NumPy 2.1.0, SciPy 1.17.1, pandas 2.2.3 and
Matplotlib 3.10.1. Enable CPU JAX; scripts enable float64. Provider `src` and
the pinned ExoJAX `src` must precede other installations on PYTHONPATH.

```bash
python examples/fetch_phoenix.py
python examples/stellar_halpha.py --wavelength results/stellar_input/wavelength.fits --flux results/stellar_input/phoenix10000.fits
python tools/verify_stellar_spectrum.py --wavelength results/stellar_input/wavelength.fits --flux results/stellar_input/phoenix10000.fits
python results/stellar_research/stellar_opacity_budget.py --input results/stellar_halpha/arrays.npz --output results/stellar_opacity
```

The actual diagnostic run used the identical offline NIST row from
`tools/verify_nlte_spectrum.py:NIST_ROW` when constructing the model. The
verification tool installs this fixture itself. It contains only the H-alpha
center used by the previous observer; hydrogen pumping frequencies come from
the existing atomic energies. Full test execution receipts are under
`results/stellar_verification_final`.

Full original PHOENIX tables are not redistributed: the fetch script checks
the exact public files. The library requests citation to Husser et al. (2013),
but an explicit redistribution license was not found. `metadata.json` retains
URLs, hashes, units and scalar diagnostics. `inspect_phoenix.py` is the original
one-off inspection script and retains `/tmp/escape-stellar-sed-data` paths;
remap those paths for inspection elsewhere. Normal example/verification
commands accept paths explicitly. The old-model absorber review in
`absorbers.md` is background; the new SED optical depths are in
`results/stellar_opacity/stellar_opacity_budget.json`.

The main plot retains only complete SED comparisons plus a projected rotation
sensitivity. Deliberately mixed line/continuum fields, which can produce very
large signals, remain in metrics/arrays. No atmosphere is refitted. Radiation
still has no depth-dependent feedback, and the observer's stellar denominator
and full observation-processing operator are unchanged. Numerical convergence
does not validate the stellar FUV core or a planetary atmosphere.

Observations in plots: Y. Zhang (2026),
https://doi.org/10.5281/zenodo.20707255, CC BY 4.0.
Stellar library: https://arxiv.org/html/1303.5632.
