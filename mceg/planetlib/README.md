# Planetlib adoption

This companion contract adopts **7 partial models and 39 equations** from
Planetlib commit `a83b3fb255c7a383c6a8ef502328266f4dd16bff`, against provider
source `18064c9f743c38651b94c2a53d3890d2032852bb`. The inventory links it from the
isothermal wind, coupled hydrogen model and Cloudy investigation. Original
contracts, equation revisions, execution artifacts and judgments are retained.
The new correspondence Assessments are all `unknown`; their pins identify the
new review targets without transferring Planetlib or historical provider support.

| Planetlib model (revision) | Provider correspondence | Scope |
| --- | --- | --- |
| `radiation.line.coefficients` (2) | `CoupledHydrogenModel.radiation_state` line contributions | Volume populations, ordinary Hz, signed extinction, spontaneous emissivity per steradian; supplied profiles and population closure |
| `escape.energy.continuum` (1) | `radiation_state`, `solve_moment_transfer`, `reactive_flow_residual` | Spherical flow, physical `coupling=1`, integrated `F=4*pi*H`, full excitation/ionization energy and escaping two-photon cooling |
| `escape.energy.finite_volume` (1) | `moment_geometry`, `spherical_shell_volume`, transfer and flow residuals | Common radial faces, quadrature and right-cell source values; per-steradian shell volume; exact algebra does not establish resolution convergence |
| `escape.parker.isothermal_wind` (1) | `sonic_radius_cm`, `parker_velocity` | Constant temperature and molecular weight, no tides; returned km/s must be multiplied by 100000 for catalog cm/s |
| `atomic.balance.milne_saha` (1) | `atom_saha_factors`, `atom_milne_population`, `atom_bound_free_rates`, `continuum_coefficients` | Common cross section and spectral weights; bound-free coefficients selected with free-free and H-minus disabled; no full-atom LTE claim |
| `atomic.coulomb.step` (1) | `_orbital_cross_sections` normalized downward recurrence | `branch=2` plus, `branch=1` minus; channel-specific index domains and `E=1+n^2*k^2` |
| `atomic.coulomb.seeds` (1) | The same recurrence's top and next seeds | Integer `n` through 20; missing channels at small `n` are omitted; no transferred floating-point validity from the old low-shell atom |

All table model IDs have prefix `planetlib.`. The original Kramers approximation
remains historical or explicitly selected comparison behavior. The Coulomb
adoption concerns the current hydrogenic default. The LTE population and spectral
coordinate models, the standalone two-state production model, and the tidal or
nonisothermal wind are not implicitly adopted into this NLTE/no-tide selection.

Bindings describe source locations and only directly representable arguments.
The native scalar-result mapping does not encode tuple components or struct
fields. For example, continuum coefficients are summed over levels; the rate
integrands are summed with `weight_hz`; `radiation_state` returns fields, not a
single line coefficient. The Parker return conversion and derived sound speed
are explicit here and in the tests. No scalar result mapping is invented for
these projections.

## Maintenance and reproduction

- **Maintained mathematical definitions:** each selected upstream
  `Equation.sympy` in the fixed Planetlib catalog.
- **Maintained local declarations:** `adoption.yaml` (Subjects, Contexts,
  Claims, Bindings and unknown Assessments) and `imports.lock.json` (catalog-file
  SHA-256 and native model dependency digests).
- **Generated distribution artifact:** `contract.yaml`, a standalone MCEG 0.6
  contract. Never edit its imported definitions independently. Rebuild and review
  it after editing local declarations or explicitly selecting a new library pin.

The generated snapshot allows native inspection without a Planetlib checkout;
it is not a second formula editing location. Existing Planetlib `build` resolves
the fixed definitions and source ancestry, without importing upstream Subjects,
Contexts, Claims, Bindings or Assessments. No resolver or formula parser is copied
into this repository. The numerical `escape` package has no new runtime dependency.

Use a separate authoring environment and an immutable Planetlib checkout:

```bash
python -m venv /tmp/escape-planetlib-env
source /tmp/escape-planetlib-env/bin/activate
python -m pip install -r mceg/planetlib/requirements.txt -e '.[test]'
git clone https://github.com/HajimeKawahara/planetlib.git /tmp/planetlib-pinned
git -C /tmp/planetlib-pinned checkout --detach a83b3fb255c7a383c6a8ef502328266f4dd16bff
python /tmp/planetlib-pinned/tools/planetlib.py build mceg/planetlib/adoption.yaml \
  --lock mceg/planetlib/imports.lock.json --output /tmp/escape-planetlib.yaml
diff -u mceg/planetlib/contract.yaml /tmp/escape-planetlib.yaml
python -m mceg mceg/planetlib/contract.yaml
JAX_PLATFORMS=cpu python -m pytest mceg/planetlib/test_adoption.py -q
python -m mceg check mceg/planetlib/contract.yaml \
  --assessment escape.planetlib.energy.pending --root .
```

The dedicated CI runs this fixed-catalog rebuild and the optional adoption tests.
Ordinary tests still do not require MCEG; adoption tests skip when its 0.6 reader
or SymPy is absent. Native `load_contract`, `mathematical_model_digest`,
`to_sympy`, and `mceg.check.check_contract` perform decoding, identity checks,
selected-file checks and binding AST inspection. `equation_digest` and
`binding_digest` created the explicit review pins. Test-only evaluation uses
native-decoded SymPy expressions, independently supplied numerical inputs, and
the actual provider functions. There is no local schema or mathematical parser.

MCEG requires valid native records, explicit references and pins for positive or
negative judgments. Its generated-expression example is optional. Planetlib's
single maintained catalog source and the provider's deterministic snapshot check
are project policies. The pinned specifications are
[symbolic source](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/symbolic.md),
[models](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/mathematical_models.md),
[bindings and pins](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/equations.md), and
[assessment history](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/assessment_lifecycle.md).

## Review impact and limits

The new test selection compares catalog expressions with source outputs for the
Parker sonic point/ODE, Saha-Milne factors and quadrature, actual coupled line
coefficients and material heating, conservative radiation and heated material
cells, and Coulomb recurrence reconstruction at `n=1,2,3,10,20`. It does not run a
new Cloudy campaign, prove all code paths equivalent, or reassess physical
applicability. The Coulomb reconstruction's final cross-section normalization
is provider-specific test projection outside the adopted recurrence model.

A changed imported definition or connection requires reviewing the matching new
`escape.planetlib.*` Claim. The relevant historical review targets remain:

| Existing contract | Claims to inspect when making a later scientific change |
| --- | --- |
| `escape.parker_wind.yaml` | `formulation_correspondence`, `numerical_verification` for the no-tide isothermal subset |
| `escape.coupled_hydrogen.yaml` | `definition_review`, `atomic_balance`, `radiation_conservation`, `thermal_balance`, `reactive_flow_balance`, `coupled_flow_balance` |
| `escape.cloudy_investigation.yaml` | `bound_free_data`, `repaired_definition`, `current_numerical_consistency`, with their historical Subjects |
| `escape.atom_convergence.yaml` | `extended_formulation`, `extended_numerics`, `extended_shell_convergence`, `matched_reference_agreement` |

Those Claims and all original Assessments are unchanged. Updating a lock or pin
is not a new execution or review. Model-boundary review remains pending, and
uncontracted spectrum families remain uncontracted. Schema success, selected
numerical agreement, source correspondence and atmospheric truth remain separate.
