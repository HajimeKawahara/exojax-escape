# exojax-escape-hydrogen

Hydrogen escape analysis with ExoJAX. Generate H-alpha emission from a prescribed
multilayer atmosphere and compare it with observations. Model assumptions and
manual parameter choices are described in the pull request.

## Setup

Use Python 3.10+ and an ExoJAX checkout containing
[PR #833](https://github.com/HajimeKawahara/exojax/pull/833)
(tested against commit `a0b3fba805d7f7d7fb6ef9adc2370f225d13093b`).
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
The bundled Zenodo observation is licensed under CC BY 4.0; see
[data attribution and column interpretation](data/zenodo_20707255/NOTICE.txt).
`compare_manual.py` saves a reproducible hand-adjusted comparison, residuals,
and parameter values under `results/manual/`. It adds a constant residual
baseline to the model; the bundled observations remain unchanged.

For an individual spectrum:

```python
import jax
from hydrogen_forward import HydrogenForwardModel

jax.config.update("jax_enable_x64", True)
model = HydrogenForwardModel(phase=0.375)
contrast = model.spectrum(n2_column=3e14, departure_floor=0.08, outflow_kms=10.0)
velocity_kms = model.velocity_kms
```

`contrast` is the continuum-subtracted planet / blackbody-star flux ratio,
expressed as a fraction. Positive velocity denotes recession. See the model
docstrings for parameter bounds, units, and geometric assumptions.
The comparison figures and equations are explained in the pull request.
