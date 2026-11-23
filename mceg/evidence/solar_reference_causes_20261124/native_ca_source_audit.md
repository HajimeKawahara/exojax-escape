# Native Ca/Ly-alpha source audit

Static source audit of the selected SolarCompositionModel.evaluate call path. No model execution, test rerun, thermal adequacy or physical-equivalence claim.

The native Ca/Ly-alpha OTS path already shares the actual four-level Ca absorption between H destruction probability, albedo, electron-escape remainder and destroyed-photon allocation. The missing legacy Ca opacity in the C23 reference is not present on this audited path. No native change to match legacy C23 is indicated.

| Stage | Source location |
|---|---|
| charge feedback | `src/escape/solar.py:164` |
| Ca opacity added once | `src/escape/solar.py:400` |
| common H operator | `src/escape/solar.py:416` |
| Fe not generic OTS | `src/escape/solar.py:430` |
| common metal receiver | `src/escape/solar.py:452` |
| thermal ledger | `src/escape/solar.py:530` |
| actual excited Ca opacity | `src/escape/metal_network.py:370` |
| Ca OTS receiver | `src/escape/metal_network.py:822` |
| Ca ionization | `src/escape/metal_network.py:830` |
| Ca photoelectron heat | `src/escape/metal_network.py:838` |
| Ca excitation loss | `src/escape/metal_network.py:867` |
| charge rate target | `src/escape/metal_network.py:897` |
| heat from events | `src/escape/metal_network.py:947` |
| same total absorption | `src/escape/hydrogen_escape_probability.py:459` |
| same albedo | `src/escape/hydrogen_escape_probability.py:498` |
| same beta input | `src/escape/hydrogen_escape_probability.py:532` |
| same OTS denominator | `src/escape/hydrogen_escape_probability.py:623` |
| beta from absorption | `src/escape/line_escape_probability.py:301` |
| electron escape remaining probability | `src/escape/line_escape_probability.py:346` |
| H loss and OTS source | `src/escape/line_escape_probability.py:365` |
| existing conditional photon and heat test | `tests/test_metal_network.py:214` |

The corresponding source hashes and equations are in `native_ca_source_audit.json`.

Limitations:

- The supplied Ca cross sections apply only to the Ly-alpha destruction/receiver prescription. They do not define a broadband excited-Ca photoionization continuum or complete inverse-Milne closure.
- This source audit establishes the connected bookkeeping, not the numerical size or adequacy of the resulting atmospheric feedback.
- The existing test checks conditional opacity, ionization and kinetic-energy consistency. It is not an atmosphere-level thermal or observational validation.
- Ca opacity also enters the point-frequency stellar Ly-alpha attenuation used for H pumping; without a broadband cross-section law this does not establish a complete integrated stellar Ca absorption budget.
