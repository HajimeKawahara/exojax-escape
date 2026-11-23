# Connected atmosphere: computational evidence

This archive separates the production pure-H connection, independent Cloudy
references, and experimental native microphysics. It does **not** establish a
solar atmosphere equivalent to the unpublished author configuration.

`manifest.json` maps original execution paths to archive paths and records
SHA-256 digests. Failed comparisons and inconclusive process terminations are
retained. A small nonlinear residual, a passing derivative test, or a recorded
FollowUp does not establish scientific adequacy.

## Inputs and licensing

- Cloudy C23.01: the official release URL and tarball, executable and abundance
  hashes are in `escape-reference-provenance.json`. Source adaptations and
  coefficient tables retain the Cloudy license. Build the unmodified release
  with the recorded compiler; diagnostic binaries only add readouts.
- sunbather: use the recorded source revision and `solveT.py` hash. The
  strict reference driver requires the physical energy residual to converge;
  its optional public smoothing criterion is identified separately.
- PHOENIX: full stellar FITS/SED tables are **not** redistributed. Retrieve them
  using the provider's `examples/fetch_phoenix.py` and `stellar_input/metadata.json`.
  Compare hashes before replaying. The flux is a specified proxy, not a
  recovered author input. Incident stellar spectral-mesh refinement is a
  separate requirement from the observer wavelength grid.
- Observation: the provider's `data/zenodo_20707255/NOTICE.txt` documents
  the CC BY 4.0 measurement. `observation/` records the archived exposure times
  and the approximate reconstruction; the unpublished reduction is not known.

## Replaying the retained calculations

Install the provider at the source revision recorded in each receipt, with
ExoJAX revision `38cb2e44f153ad6d729e3253e235702de52920cf`, Python 3.12.2,
JAX/jaxlib 0.10.0, NumPy 2.1.0 and SciPy 1.17.1. Use float64 and CPU execution.
The original commands and CPU affinity are retained in `execution/`.

The maintained external entry points are:

```bash
python tools/verify_atmosphere_reference.py --help
python tools/cloudy_atmosphere.py --help
python examples/cloudy_halpha.py --help
```

`cloudy_atmosphere.py` defaults to the actual C23 hydrogen atom: resolved
through n=10, 15 collapsed shells through n=25, and topoff enabled. Historical
n=15/resolved-through-3 controls require `--hydrogen-atom native-matched`.
Use `--iterations 0` for prescribed-temperature/density controls; it does not
solve thermal equilibrium. Full chemistry requires `--molecules`.

The same archived Cloudy populations enter the observer. Supplied n=21..25
states require the archived high-n cross-section fixture. Their direct
continuum contribution is tested separately; a small observer contribution
does not license truncating their cascades in the population solve.

Standalone prototype scripts preserve their original absolute paths in order
to retain exact source identities. In a clean replay environment, restore the
paths given by the manifest or replace those path constants explicitly. Their
C++ harnesses include original C23 function bodies and link `source/libcloudy.a`;
the build uses `g++ -O2 -std=c++11 -no-pie -fno-pie -I<C23/source>`.
Executables and full Cloudy source/data releases are not bundled.

The seven-metal kernel requires the original STOUT/CHIANTI/ADfA files from the
recorded C23 release. Recombination and charge-exchange fits read that same
data directory. The restricted Cota table is included with its license and
source-extraction harness; its documented domain is not silently extended.

## Interpretation

- `slabs/`: fixed-temperature pure-H radiation/population controls and
  frequency/spatial refinements. These are not solved planetary atmospheres.
- `composition/` and `metal_diagnostic/`: controlled heating comparisons.
  Their exact percentages depend on the stated continuum grid.
- `prototypes/`: independently checked coefficient/operator kernels and
  attempted connections. Read each receipt's scope and failure status before
  treating it as support for an entire model.
- `figures/`: derived planet spectra and profiles. Published NPZ copies omit
  the PHOENIX-derived stellar reference array; input hashes remain available.

The repository's `mceg/uses/yapeng_atmosphere.yaml` selects the actual production
spectrum entry point. Prototype success cannot satisfy a requirement for code
that does not execute that prototype. The scientific readiness result remains
false until all required physical and numerical checks have support.
