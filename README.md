# exojax-escape-hydrogen

Hydrogen escape analysis with ExoJAX. Generate multilayer H-alpha spectra using
Parker mass continuity and an isolated two-level population balance. Model
assumptions and comparisons with observations are described in the pull request.

## Setup

Use Python 3.10+ and an ExoJAX checkout containing
[PR #836](https://github.com/HajimeKawahara/exojax/pull/836), tested at commit
`773f44b8af03d2198c82f1fc6f0022c278fb2c6d`.
With that checkout at `../exojax`, run from this repository:

```bash
python -m pip install -e ../exojax
python -m pip install pytest
```

## Run

```bash
JAX_PLATFORMS=cpu python fit_three_models.py --check-convergence
JAX_PLATFORMS=cpu python compare_populations.py --check-convergence
JAX_PLATFORMS=cpu python generate_spectra.py --check-convergence
JAX_PLATFORMS=cpu python compare_manual.py
JAX_PLATFORMS=cpu python fit_spectrum.py
JAX_PLATFORMS=cpu python -m pytest -q
```

`fit_three_models.py` fits the three density/population choices and saves spectra,
profiles, diagnostics, and a comparison figure under `results/three_models/`.
Use `--plot-only` with the same output directory to regenerate saved-fit plots.
`compare_populations.py` saves the rate-model comparison, atmospheric profiles,
spectra, and parameters under `results/populations/`. Use `--output` to change
the destination. `generate_spectra.py` retains the prescribed examples under
`results/demo/` and accepts `--phase` and `--nlayer`.
No external atmospheric tables or observational downloads are required.
The first model construction downloads NIST H I lines through RADIS and caches
locally; subsequent runs reuse that cache. Tests use a small offline NIST row.
The bundled Zenodo observation is licensed under CC BY 4.0; see
[data attribution and column interpretation](data/zenodo_20707255/NOTICE.txt).
`compare_manual.py` compares two prescribed atmospheres using Parker velocities
and saves residuals and parameter values under `results/manual/`. It adds a
constant residual baseline to the model; the bundled observations remain unchanged.
`fit_spectrum.py` saves an exploratory least-squares comparison, spectra, and
parameters under `results/fit/`. These three older scripts explicitly call
`prescribed_spectrum` to retain the previous exponential/sigmoid comparisons.
`prescribed_spectrum` is Model 1; `continuity_spectrum` is Model 2
(continuity with the prescribed ratio); `spectrum` is Model 3 (transition rates). Interpretation,
equations, atomic-data references, and limitations are in the pull request.

For an individual spectrum:

```python
import jax
from hydrogen_forward import HydrogenForwardModel

jax.config.update("jax_enable_x64", True)
model = HydrogenForwardModel(phase=0.375, temperature_top_k=15000.0)
contrast = model.spectrum(
    mass_loss_g_s=1e13, excited_fraction=1e-9,
    ionized_fraction=0.5, wind_temperature_k=9000.0,
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
