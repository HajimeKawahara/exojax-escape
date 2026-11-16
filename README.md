# exojax-escape-hydrogen

Hydrogen escape analysis with ExoJAX. Generate H-alpha emission from a prescribed
multilayer atmosphere and compare it with observations. Model assumptions and
manual parameter choices are described in the pull request.

## Setup

Use Python 3.10+ and an ExoJAX checkout containing
[PR #836](https://github.com/HajimeKawahara/exojax/pull/836), tested at commit
`e676c6c7f4a88d07cc8dde1cc0329acac991975d`.
With that checkout at `../exojax`, run from this repository:

```bash
python -m pip install -e ../exojax
python -m pip install pytest
```

## Run

```bash
JAX_PLATFORMS=cpu python generate_spectra.py --check-convergence
JAX_PLATFORMS=cpu python compare_manual.py
JAX_PLATFORMS=cpu python -m pytest -q
```

The example saves `model.png`, `spectra.csv`, and `parameters.json` under
`results/demo/`. Use `--output`, `--phase`, or `--nlayer` to change the run.
No external atmospheric tables or observational downloads are required.
The first model construction downloads NIST H I lines through RADIS and caches
locally; subsequent runs reuse that cache. Tests use a small offline NIST row.
The bundled Zenodo observation is licensed under CC BY 4.0; see
[data attribution and column interpretation](data/zenodo_20707255/NOTICE.txt).
`compare_manual.py` compares two prescribed atmospheres using Parker velocities
and saves residuals and parameter values under `results/manual/`. It adds a
constant residual baseline to the model; the bundled observations remain unchanged.

For an individual spectrum:

```python
import jax
from hydrogen_forward import HydrogenForwardModel

jax.config.update("jax_enable_x64", True)
model = HydrogenForwardModel(phase=0.375)
contrast = model.spectrum(
    n2_column=3e14, departure_floor=0.08,
    wind_temperature_k=12000.0, mean_molecular_weight=0.5,
)
velocity_kms = model.velocity_kms
```

`contrast` is the continuum-subtracted planet / blackbody-star flux ratio,
expressed as a fraction. Positive velocity denotes recession. See the model
docstrings for parameter bounds, units, and geometric assumptions.
`wind_temperature_k` is a constant effective Parker-flow temperature, separate
from the prescribed line-forming temperature set by `temperature_top_k`.
Set the assumed planet mass with `HydrogenForwardModel(planet_mass_mj=...)`.
The comparison figures and equations are explained in the pull request.

The default `wavelength_model="nist"` uses the NIST H I level-energy difference
(approximately 6564.632 Angstrom in vacuum). For the former Rydberg line-center
approximation, use `HydrogenForwardModel(wavelength_model="rydberg")` or
`python generate_spectra.py --wavelength-model rydberg --output results/rydberg`.
`compare_manual.py` accepts the same flag. This is a line-center experiment at
fixed NIST oscillator strength, not a recreation of every old coefficient.
The Lorentz width uses the isolated-transition approximation `A/(4*pi*c)`;
it omits additional lifetime and collisional broadening.
