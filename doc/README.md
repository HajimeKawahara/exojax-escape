# Mathematical model definitions

The [H-alpha coupling investigation](halpha_coupling.pdf), with
[editable TeX](halpha_coupling.tex), compares finite geometry, radial Doppler
transfer and H-alpha/Ly-beta recycling at the saved PR12/13 atmosphere.
Its [MCEG contract](../mceg/contracts/escape.halpha_coupling.yaml) separates
fixed-state transport checks from the unresolved population and observed-profile
response. Build with `make -C doc halpha-coupling-pdf`.

The current solar-composition candidate is defined in
[solar_atmosphere.pdf](solar_atmosphere.pdf), with [editable TeX](solar_atmosphere.tex).
It couples populations, radiation, thermal balance and the tidal Parker wind.
The [solar readiness manifest](../mceg/uses/solar_proxy.yaml) separates a verified
numerical solution from unresolved physical and inference requirements.

The preceding pure-hydrogen [connected atmosphere](coupled_atmosphere.pdf), with [editable TeX](coupled_atmosphere.tex),
defines the native radiation/population/temperature/wind root used by
`HydrogenAtmosphereModel.spectrum`. It separates implementation checks from
physical equivalence to the Cloudy/sunbather atmosphere. The current pure-H
composition and static redistribution remain reduced approximations; the
purpose-scoped [readiness manifest](../mceg/uses/yapeng_atmosphere.yaml) does not
authorize the model for that scientific use. Build with `make -C doc atmosphere-pdf`.

The [radiation-closure supplement](radiation_closure.pdf), with
[editable TeX](radiation_closure.tex), explains the uniform unattenuated
stellar field and its unresolved effect. It links the new
`radiation_closure` Equation and implementation Binding to the
`prescribed_field_review` Assessment, its controlled-comparison FollowUp and
provisional-use Decision. This review adds no radiation feedback to the runtime.
Build it with `make -C doc radiation-pdf`.

The extended atom-to-observer connection is defined in
[nlte_spectrum.pdf](nlte_spectrum.pdf), with [editable explanation](nlte_spectrum.tex),
a generated [equation appendix](nlte_spectrum_equations.tex), and its
[MCEG contract](../mceg/contracts/escape.nlte_spectrum.yaml). It connects the
default nmax=15 local population solve to prescribed-temperature Parker flow,
branch-specific radiative lifetimes and the existing ExoJAX intensity transfer.
It also defines finite-sample derivative checks and an observer-pixel refinement
criterion of 0.1 times the supplied observational uncertainty. These numerical
criteria do not establish atmospheric accuracy or a validated likelihood.
Use `make -C doc spectrum-equations` and `make -C doc spectrum-pdf` to rebuild it.
The equation target reads the historical contract at `1f4968f` to preserve the
original artifact hashes; it requires a Git checkout containing that revision.
New radiation-closure records are explained in the supplement above.

The current observer uses three H-alpha centers from the atom's n/l energies.
The [branch-center supplement](observer_branch_centers.pdf), with
[editable TeX](observer_branch_centers.tex), explains this change and its
`branch_profiles@2` code bindings. It preserves the historical common-center
document. Build with `make -C doc spectrum-centers-equations spectrum-centers-pdf`.
Full fine-structure and pressure broadening remain separate limitations.

The enlarged hydrogen atom is described in [atom_convergence.pdf](atom_convergence.pdf),
with [editable explanation](atom_convergence.tex), a generated
[equation appendix](atom_convergence_equations.tex) and its
[MCEG contract](../mceg/contracts/escape.atom_convergence.yaml). It explains the
missing high-shell recombination/cascade paths, stable atomic kernels and the
choice of nmax=15 compared with nmax=20 and larger independent Cloudy atoms.
The original nmax=4-to-5 contradiction remains in the historical contract.
Use `make -C doc atom-equations` and `make -C doc atom-pdf` to regenerate it.

The prescribed-temperature wind is defined in [parker_wind.pdf](parker_wind.pdf),
with [editable explanation](parker_wind.tex) and a
[generated equation appendix](parker_wind_equations.tex). Its
[MCEG contract](../mceg/contracts/escape.parker_wind.yaml) covers the isothermal
limit, nonisothermal Parker wind, linear stellar tides, regular sonic crossing
and implicit derivatives. Temperature and mean molecular weight remain prescribed;
this wind does not solve thermal balance. Use `make -C doc parker-equations`
and `make -C doc parker-pdf` to regenerate the document.

The coupled model is defined in [coupled_model.pdf](coupled_model.pdf), with
[editable explanation](coupled_model.tex) and a
[generated equation appendix](coupled_model_equations.tex). It separates static
radiation/statistical equilibrium, thermal equilibrium and a finite spherical
reactive-flow branch. Equations and verification requirements were recorded
before implementing these new components. A finite inflow branch does not select
a transonic mass-loss eigenvalue. The [gate workflow](../mceg/workflows/coupled.json)
keeps numerical checks, atom-size accuracy and independent reference evidence
separate; a small equation residual is not an atmospheric validation.

The preceding local NLTE model is described in
[hydrogen_nlte.pdf](hydrogen_nlte.pdf), with its editable explanation in
[hydrogen_nlte.tex](hydrogen_nlte.tex). Its equation appendix is generated from
the [schema-0.4 contract](../mceg/contracts/escape.hydrogen_nlte.yaml). It defines
the six-state atom, prescribed radiation, local solver, implicit derivatives
and branch-weighted H-alpha transfer, including the approximations that remain.

[model.pdf](model.pdf) and [model.tex](model.tex) retain the historical two-level
model and the initial NLTE proposal. Their bytes remain pinned by the original
formulation review; they are not updated to imply that its old tests cover the
new model.

Build with a TeX installation providing `latexmk` and `pdflatex`:

```bash
make -C doc pdf
make -C doc nlte-pdf
make -C doc coupled-pdf
make -C doc spectrum-pdf
make -C doc radiation-pdf
```

The PDFs are committed for reading without LaTeX. Auxiliary files stay in the
ignored `doc/build/` directory; `make -C doc clean` removes only that directory.
TeX and MCEG are external authoring tools, not Python package dependencies.

The original observer appendix is pinned historical evidence. Rebuild and
check it against its recorded contract, rather than overwriting it with later
definitions:

```bash
make -C doc spectrum-equations
make -C doc spectrum-pdf
python -m mceg equations doc/build/nlte_spectrum_contract.yaml \
  --output doc/nlte_spectrum_equations.tex --check
```

The last command checks generated bytes only. It does not validate the model.
To inspect all current definitions, export the current contract to a separate
file under `results/`; the radiation supplement quotes `radiation_closure@1`
and links its Binding explicitly. Previous document and evidence hashes remain
unchanged.
Equation Bindings identify Python functions and argument conventions; scoped
Assessments retain source/artifact identities. Decisions record chosen
approximations and the conditions for reconsidering them. Numerical evidence
comes from an actual execution receipt, independently of document generation.
The observer checks run with `python tools/verify_nlte_spectrum.py --layers 128
--record results/nlte-check-001`; the recorded 128-layer preset passes the
independent and joint refinements after the 32- and 64-layer candidates fail
the unchanged criterion. Its seven declared conditions are samples, not a
guarantee over a future inference domain. The example preserves the atmospheric
profile figure and observed spectrum; its optional NUTS smoke has not been run
for this update.
The final candidate combines the six numerical refinements in a joint control
in addition to testing each one independently.

For the coupled document, use `make -C doc coupled-equations` followed by
`make -C doc coupled-pdf`. The provider-owned `tools/verify_coupled.py` executes
the declared checks and retains each attempt. Its receipts never automatically
change the contract's scientific Assessments.

## Preserve the historical formulation

1. Update the equations, assumptions, implementation status and document revision
   in the TeX source. Preserve equation labels, or introduce a new label when a
   definition changes materially.
2. Rebuild the PDF and inspect equations, page breaks and references. Check
   `doc/build/model.log` for unresolved references and overflowing content.
3. Review the `population_formulation` Claim and `formulation` Evidence in the
   [MCEG contract](../mceg/contracts/escape.hydrogen_populations.yaml). Record new
   document hashes and a scoped assessment after review. Preserve older records
   in version history; do not relabel planned equations as implemented.
4. If code or physics changes, update the model inventory and run the relevant
   scientific checks. A new document does not renew historical test evidence.

In a separate MCEG environment, run from the repository root:

```bash
python -m mceg check mceg/contracts/escape.hydrogen_populations.yaml \
  --assessment formulation_review --root .
```

For the historical v0.2 contract, MCEG tracks evidence paths, prose locations and hashes. It does not parse TeX,
check equation labels, prove equivalence to code, or generate this PDF. The
historical formulation review covers only POP-1 through POP-5 and their limiting
identities. Other document sections and physical applicability need their own
reviews and evidence.

Historical contracts keep their original whole-file source hashes, even when
an unrelated function in that file changes. Replay those reviews against their
recorded checkouts as described in the [MCEG guide](../mceg/README.md); a current
source inventory update does not renew an older scientific Assessment.

## Current Cloudy repair investigation

[cloudy_investigation.pdf](cloudy_investigation.pdf) explains the causal rate-block
experiments, the PSM20/Coulomb repairs, unchanged comparison criteria and MCEG
repair loop. Its [TeX source](cloudy_investigation.tex) includes a generated
[Equation appendix](cloudy_investigation_equations.tex) from the new investigation
contract. The older documents remain historical formulations. Build with
`make -C doc cloudy-equations cloudy-pdf` from the repository root; equation
generation uses the separate MCEG authoring environment.
