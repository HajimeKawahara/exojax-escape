# exojax-escape

Hydrogen escape analysis with ExoJAX. Model definitions, fitted spectra,
equations, and limitations are in [PR #1](https://github.com/HajimeKawahara/exojax-escape/pull/1).

## Layout

- `src/escape/`: forward model, hydrogen populations, and Parker wind.
- `examples/`: three-model fitting and illustrative spectrum generation.
- `tests/`: physical limits, conservation, and derivative checks.
- `doc/`: [mathematical definitions in TeX and PDF](doc/README.md).
- `mceg/`: [model contracts, evidence and open reviews](mceg/README.md); optional
  tooling kept outside the package dependencies.
- `data/`: the original Zenodo observation and attribution.
- `results/`: generated spectra, parameters, and figures (not tracked).

The [MCEG status overview](mceg/README.md#current-status) separates supported
numerical checks, measured contradictions and unverified physical accuracy.
The finite hydrogen atom now defaults to `nmax=15` and supports explicit shells
through `nmax=20`. The [convergence investigation](doc/atom_convergence.pdf)
explains the original truncation failure and the scoped Cloudy comparisons.
Smaller atoms remain explicit numerical experiments; convergence must be checked
again for new atmospheric conditions.
The same extended atom now supplies the observer H-alpha spectrum, including
separate natural widths for its three resolved branches. The
[spectrum equations](doc/nlte_spectrum.pdf) describe that connection and its
numerical checks; physical accuracy for the observed atmosphere remains unverified.
The [Parker-wind equations](doc/parker_wind.pdf) cover prescribed-temperature
nonisothermal winds and linear tides, with code bindings and scoped evidence.

## Setup

Use Python 3.10+ and an ExoJAX checkout containing
[PR #836](https://github.com/HajimeKawahara/exojax/pull/836), tested at commit
`38cb2e44f153ad6d729e3253e235702de52920cf`.
With that checkout at `../exojax`, run from this repository:

```bash
python -m pip install -e ../exojax
python -m pip install -e '.[examples,test]'
```

## Run

```bash
JAX_PLATFORMS=cpu python examples/fit_three_models.py --check-convergence
JAX_PLATFORMS=cpu python examples/generate_spectra.py --check-convergence
JAX_PLATFORMS=cpu python examples/compare_parker.py
JAX_PLATFORMS=cpu python examples/compare_tides.py
JAX_PLATFORMS=cpu python examples/nlte_halpha.py
JAX_PLATFORMS=cpu python examples/coupled_hydrogen.py
JAX_PLATFORMS=cpu python -m pytest -q
```

The fit saves spectra, atmospheric profiles, parameters, and a comparison figure
under `results/three_models/`. Use `--plot-only` with the same output directory
to regenerate the figure from saved fits. The illustrative example reproduces
the original temperature and velocity profiles under `results/demo/`.
Both scripts accept `--output`.
`compare_parker.py` saves a prescribed-temperature wind comparison and gradient
checks under `results/parker_comparison/`; it does not fit the observations.
`compare_tides.py` compares that wind with and without linear stellar tides under
`results/tidal_comparison/`, including density and implicit-gradient checks.

The first model construction downloads NIST H I lines through RADIS and caches
them locally. Tests use an offline NIST row. The bundled observation is licensed
under CC BY 4.0; see [attribution and column interpretation](data/zenodo_20707255/NOTICE.txt).

`nlte_halpha.py` solves six resolved low states plus statistical shells from
n=4 through n=15 by default, together with ionization, then generates H-alpha.
Use `--nmax` for finite-atom comparisons through n=20. Temperature and irradiation
are prescribed; mass continuity and the Parker velocity set the total density.
The figure retains population, temperature, velocity and ionization profiles
alongside the Zenodo observation and generated spectrum. Outputs remain in
`results/nlte/`. The former six-state relaxation diagnostic is not applied to
the enlarged atom. This example is not a fit or a reproduction of the paper's
coupled atmosphere. See the [spectrum equations and assumptions](doc/nlte_spectrum.pdf)
and [MCEG contract](mceg/contracts/escape.nlte_spectrum.yaml); the
[six-state document](doc/hydrogen_nlte.pdf) retains the historical formulation.
For this model import `HydrogenNLTEForwardModel` from `escape.nlte_forward`;
its default wind is nonisothermal with prescribed constant molecular weight.
It defaults to 128 atmospheric layers. The example and numerical verification
use 12 by 6 disk nodes and 601 spectral points; the class retains the inherited
24 by 12 disk nodes and 1201 spectral points unless explicitly configured.
It needs float64 and gas temperatures within approximately 5802--14000 K.
An optional `--nuts-smoke` requires NumPyro and samples synthetic spectral
data for 32 warmup and 32 retained steps; it checks execution, not convergence.
It has not been run as part of this extended-spectrum verification.
MCEG remains an external authoring/checking tool, with no runtime import or
dependency in `escape`.

Run the numerical spectrum checks separately:

```bash
JAX_PLATFORMS=cpu python tools/verify_nlte_spectrum.py --layers 128 --record results/nlte-check-001
```

The tool checks complete-spectrum derivatives, evaluation time and independent
refinements of the atom, atmospheric, disk, wavelength, continuum and wind grids.
The recorded 128-layer preset passes all independent refinements and a joint
refinement of every numerical axis. The largest changes are 0.06465 and 0.04886
times the supplied uncertainty, respectively. The earlier 32- and 64-layer
failures remain in the MCEG record; the criterion was unchanged.
At seven specified conditions, every sampled observation pixel must change by
less than 0.1 times its supplied uncertainty. Those uncertainties provide a
numerical scale; this does not establish a likelihood or continuous prior domain.
Each execution retains its source identities, outputs and exit status in a new
directory, including failed attempts.
On the recorded eight-core CPU configuration, median warmed evaluations take
1.45 s for the spectrum and 6.49 s for the scalar objective and gradient.
These are performance measurements for this preset, not a NUTS convergence test.

`coupled_hydrogen.py` compares three finite-domain problems: radiation and
populations at fixed temperature, radiative thermal equilibrium, and an
advecting flow with solved temperature, velocity and ionization. It saves
profiles and conservation diagnostics under `results/coupled_hydrogen/`.
The [coupled equations](doc/coupled_model.pdf) define the approximations and
[verification workflow](mceg/README.md) records their tests separately from
physical accuracy. This is a reduced-model experiment: the enlarged atom's
finite sampled convergence checks do not establish convergence of every coupled
atmosphere, and static Eddington transport and a finite inflow branch do not
establish a transonic planetary atmosphere.
The [Cloudy investigation](doc/cloudy_investigation.pdf) explains the orbital
mixing and bound-free updates to this atom. See the
[reproduction instructions](doc/cloudy_reference.md) for external comparisons
and the recorded diagnostic/repair loop; Cloudy is not a package dependency.

For an individual spectrum:

```python
import jax
from escape.hydrogen_forward import HydrogenForwardModel

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
In the default isothermal mode, the effective Parker temperature is separate
from the prescribed kinetic temperature. See the docstrings for parameters,
units, and assumptions.

Set `HydrogenForwardModel(wind_model="nonisothermal")` to use the same kinetic
temperature for pressure, populations, and line broadening. In this mode,
`temperature_top_k` controls the wind too; `wind_temperature_k` is unused and
must not be treated as a fitted parameter. The isothermal mode remains the default.
The nonisothermal solver assumes a prescribed smooth temperature, constant mean
molecular weight, and a single regular sonic point between 1 and 32 Rp. It does
not solve an energy equation. Model equations, numerical
checks, and figures are in [PR #2](https://github.com/HajimeKawahara/exojax-escape/pull/2).

Set `tidal_coefficient_s2` on `HydrogenForwardModel` to include the outward
acceleration `tidal_coefficient_s2 * r` in either wind mode; its default is zero.
The coefficient is `3 * G * Mstar / a**3` in s^-2. For a 1.48-day orbit and
negligible planet/star mass ratio, use `3 * (2 * pi / (1.48 * 86400))**2`.
The isothermal mode uses the numerical solver when tides are enabled.
This is the paper's one-dimensional linear tidal approximation, with spherical
mass conservation, not a full Roche geometry. The same coefficient is a
differentiable keyword of `escape.nonisothermal.nonisothermal_parker_velocity`.
See [PR #3](https://github.com/HajimeKawahara/exojax-escape/pull/3) for the equations,
comparison figure, and numerical checks. Older imports must change from
`escapeh` to `escape`.

The default line center uses NIST H I energies. For the former Rydberg
approximation at fixed oscillator strength, use
`HydrogenForwardModel(wavelength_model="rydberg")` or
`python examples/generate_spectra.py --wavelength-model rydberg --output results/rydberg`.
