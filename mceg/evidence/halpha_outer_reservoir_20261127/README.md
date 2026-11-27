# Extended dim outer H-alpha absorber

This campaign follows the phase-local cause investigation. It asks how an
outer n=2 reservoir could persist without an inner-like source function.
No production atmosphere, likelihood, data selection or observed residual
is changed. The [report](../../../doc/halpha_outer_reservoir.pdf) combines
three parallel investigations:

- [Atomic budgets](atomic/findings.md): restore receiving PI/RR terms,
  verify the saved five-level balance, and vary density, pumping and sinks
  with explicit fixed baths. Density can increase n2 while keeping n3/n2
  low, but radiation, charge, higher-level and thermal closure remain open.
- [Irradiation](radiation/README.md): identify the relevant photon bands,
  quantify shielding columns and finite-disk/stellar-line effects, and
  distinguish existing trapping from nonlocal Ly-alpha transfer.
- [Confinement](dynamics/notes.md): quantify continuity, pressure, magnetic
  and tidal scales. A slow dense sector needs actual force/boundary support.
  The inferred pressure field is conditional, not measured or sufficient.

`geometry_budget.py` gives a deliberately simplified area-times-source
counterexample. It is not a spectrum calculation: a larger dim shell can
emit more unresolved flux because its projected area grows. Both disk
attenuation and halo emission must be evaluated for any physical candidate.

The priority is a coupled dense-sector / radiation calculation, with
magnetic or external-pressure support as candidate dynamics. The density
factor near 20 is a local response probe, not an atmosphere measurement.
Single-channel trapping or photoionization removal, ground-H shielding and
a scalar finite-star correction do not supply the required reservoir in
their tested scopes. A self-consistent physical mechanism remains unknown.

The subdirectories retain replay commands and hashes of external archived
inputs; those inputs are not distributed by this campaign. The compact
receiving-rate NPZ is retained. `validation.json` records integrity and
finite numerical controls; it does not validate the atmosphere or fit.
MCEG adds a separate subject/context and assessments to the cause contract
without reassigning the earlier evidence or Decision basis.
