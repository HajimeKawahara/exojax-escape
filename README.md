# exojax-escape-hydrogen

Hydrogen escape analysis with ExoJAX. Model definitions, fitted spectra,
equations, and limitations are in [PR #1](https://github.com/HajimeKawahara/exojax-escape-hydrogen/pull/1).

## Layout

- `src/escapeh/`: forward model, hydrogen populations, and Parker wind.
- `examples/`: three-model fitting and illustrative spectrum generation.
- `tests/`: physical limits, conservation, and derivative checks.
- `data/`: the original Zenodo observation and attribution.
- `results/`: generated spectra, parameters, and figures (not tracked).

## Setup

Use Python 3.10+ and an ExoJAX checkout containing
[PR #836](https://github.com/HajimeKawahara/exojax/pull/836), tested at commit
`773f44b8af03d2198c82f1fc6f0022c278fb2c6d`.
With that checkout at `../exojax`, run from this repository:

```bash
python -m pip install -e ../exojax
python -m pip install -e '.[examples,test]'
```

## Run

```bash
JAX_PLATFORMS=cpu python examples/fit_three_models.py --check-convergence
JAX_PLATFORMS=cpu python examples/generate_spectra.py --check-convergence
JAX_PLATFORMS=cpu python -m pytest -q
```

The fit saves spectra, atmospheric profiles, parameters, and a comparison figure
under `results/three_models/`. Use `--plot-only` with the same output directory
to regenerate the figure from saved fits. The illustrative example reproduces
the original temperature and velocity profiles under `results/demo/`.
Both scripts accept `--output`.

The first model construction downloads NIST H I lines through RADIS and caches
them locally. Tests use an offline NIST row. The bundled observation is licensed
under CC BY 4.0; see [attribution and column interpretation](data/zenodo_20707255/NOTICE.txt).

For an individual spectrum:

```python
import jax
from escapeh.hydrogen_forward import HydrogenForwardModel

jax.config.update("jax_enable_x64", True)
model = HydrogenForwardModel(phase=0.375, temperature_top_k=15000.0)
contrast = model.spectrum(
    mass_loss_g_s=1e13, excited_fraction=1e-9,
    ionized_fraction=0.5, wind_temperature_k=9000.0,
)
velocity_kms = model.velocity_kms
```

`prescribed_spectrum`, `continuity_spectrum`, and `spectrum` correspond to
Models 1, 2, and 3. `contrast` is the continuum-subtracted planet / blackbody-star
flux ratio, expressed as a fraction; positive velocity denotes recession.
The effective Parker temperature is separate from the prescribed kinetic
temperature. See the docstrings for parameters, units, and assumptions.

The default line center uses NIST H I energies. For the former Rydberg
approximation at fixed oscillator strength, use
`HydrogenForwardModel(wavelength_model="rydberg")` or
`python examples/generate_spectra.py --wavelength-model rydberg --output results/rydberg`.
