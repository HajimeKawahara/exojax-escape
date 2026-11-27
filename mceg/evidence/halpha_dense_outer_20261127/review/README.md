# Independent review of the dense radial precursor

This experiment can solve populations and thermal balance conditional on a
prescribed flow modification. It is a spherical radial precursor, not an
angular sector, a magnetic equilibrium, or a force-balanced wind inferred
from the data. The parent driver retains the existing production equations
and temporarily changes the wind result before radiation and energy evaluation.

Let `a2=kB T/(mu mp)`, `g=d(log v_ref)/dr` and `h=d(log q)/dr`, with
`q=exp(log(D) smoothstep((r/Rp-1.3)/0.425))`. Then

```
rho = q rho_ref                 v = v_ref/q
d(log v)/dr = g-h              d(log rho)/dr = -2/r-g+h
extra acceleration = (v^2-v_ref^2) g + (a2-v^2) h
```

The last expression is the added momentum defect relative to the reference
wind. The total required force must also retain that wind's finite numerical
momentum defect. At D=1 the modification is exactly zero; it does not erase
the reference interpolant's error. The accepted full-sphere reference has
sampled defects from -1.094 to +2.408 cm/s2 on its eight cells.

The existing thermal terms are
`Qadv=-1.5 rho v d(a2)/dr` and `Qexp=a2 v d(rho)/dr`.
At fixed T/mu, `rho v` is unchanged: advection is unchanged everywhere, and
expansion is unchanged wherever q is constant. In the ramp,
`Delta Qexp=rho_ref v_ref a2 h`. A slower velocity alone therefore does not
permit dividing both cooling terms by D. The ramp can produce compression
heating, which must retain its sign.

At the frozen full-sphere reference and D=20, cell 4 has q=sqrt(20), not 20.
Its expansion term changes from -1.110e-6 to +2.056e-7 erg/cm3/s. Added
accelerations in cells 4--7 are +660.4, -48.0, -114.8, -181.0 cm/s2.
The sign-changing force is a condition on any proposed support mechanism;
a gas/magnetic-pressure magnitude alone does not establish it. Mechanical
work density `rho v extra_acceleration` is recorded separately from heat.
The present internal-energy closure assumes no additional dissipative heating;
a proposed drag or magnetic process must specify its energy exchange.

## Radiation and observable scope

Changing density before `SolarCompositionModel.evaluate` recomputes conserved
species densities, collisions, photoionization/recombination, continuum
absorption/emission, line opacity, shielding and escape at every residual call.
No old n2/n3, line source or escape probability should be retained as fixed.
The complete hydrogen rates and receiving thermal corrections remain coupled.

Use the existing spherical H-alpha group at blend=1 for both D=1 and D>1,
so the density comparison does not simultaneously switch transfer closures.
The D=1 residual must reproduce the saved accepted full-sphere checkpoint.
The current group replaces all three H-alpha radiative/OTS channels and their
matching continuum and redistribution terms. Merely adding an intercell
intensity to the old trapping terms would count some photons twice.

Ly-alpha keeps its current Hummer--Kunasz/C23 closure; the new column and
continuum destruction are recomputed, but this does not become a full
spatial/frequency Ly-alpha solution. Both that closure and the spherical
H-alpha group are static. The slow velocity enters thermal flow and the
observer, not an implemented moving-line transfer calculation. Slower gas
therefore does not add a separate Sobolev trapping term here.

The explicit H-alpha group has thermal/natural profiles and CRD; it does not
include pressure wings. Its rate replacement also supersedes the isolated
H-alpha escape operator's empirical Stark increment. This existing reduction
deserves a new-density adequacy check and must not be described as full
pressure-broadened transfer. A denser shell's larger emitting halo remains in
the spherical observer; a restricted covering sector is not represented by q.
The global continuum and Ly-alpha remain their declared baseline geometry,
apart from the group's finite-window continuum replacement and existing IR
option. This is not a uniformly spherical multiwavelength radiation model.

The flow still omits advection of excitation/ionization energy and population
advection, as documented in `atmosphere_wind.py` and `SolarAtmosphereModel`.
Short individual n2 loss times do not prove all ionization/thermal modes are
fast. No new validity claim for those omitted terms follows from q rescaling.

## Controls and review

`flow_controls.py` uses the accepted full-sphere saved state; it does not run
another atmospheric root. It tests D=1,2,5,10,20,20.19, finite-differences the
actual local pressure and velocity, and compares with the algebraic force
increment plus reference defect. Local analytic reference extensions match
the supplied values and gradients, so this validates the rescaling identities,
not a new global wind resolution study.

All seven declared checks pass. Mass-flux recovery differs by at most 6.7e-16;
continuity and thermal identities meet 1e-12; the independent pressure/velocity
derivative agrees within 1.31e-8 in the declared force scale (limit 5e-7).
The source/input hashes and per-cell quantities are in `flow_controls.json`.
The slightly different existing kB values in wind and reported gas pressure
are explicitly distinguished rather than silently mixed.

The root driver and observer have been reviewed read-only. The initial driver
import location for `PLANET_RADIUS_CM` and its force diagnostic's kB were
corrected before execution. No production-source change was requested.
The observer retains seven phases, full native GP conditioning, the original
comparison baseline, shared RV/amplitude fits, exact complete target support,
and null unsupported bands. Its success does not establish the separate
physical-state qualification.

## Executed structural comparison

`structure.json` contains all 17 accepted density states. The final
`radial_structure.pdf`/`.png` compares D=1,2,5,10,20 and the free #1 shape.
At D=20, cell 5 (1.9375 Rp) has n2=4369.97 cm-3, 332.17 times D=1,
while its nominal-center source is 3.407 times D=1. Its temperature is
7807.11 K, slightly below D=1. The source ratio at 1.5125/1.9375 Rp changes
from 2.3103 at D=1 to 1.7424 at D=20; the interpolated free #1 ratio is
22.9956. The free shape has a different geometry and prescribed source
freedom. These ratios are structural comparisons, not uniquely inferred
observational requirements.

| Lower radius / Rp | D=1 N2 / cm-2 | D=20 N2 / cm-2 | Free #1 N2 / cm-2 |
|---|---:|---:|---:|
| 1.3 | 5.71537e12 | 1.68678e14 | 4.08469e13 |
| 1.725 | 8.41545e10 | 2.77060e13 | 1.18995e13 |
| 2.15 | 8.20673e9 | 2.47823e12 | 3.30855e12 |

The D=20 columns are 4.130, 2.328 and 0.749 times the free values at these
three cuts. Its static nominal-center radial tau=1 is 2.26095 Rp, close to
the free value 2.24642 Rp, but this does not equate emitting areas, sources
or observed flux. The free source transition (midpoint 1.45 Rp, scale
0.08 Rp) lies within the current 1.3--1.725 Rp cell; this stratification has
not been radially qualified.

The actual D=20 density at cell 5 is 26.175 times D=1, whereas the imposed
q there is 20. Even the q=1 inner cells have density ratios 9.85--18.22.
This is a coupled density/slow-flow family with a responding reference
wind, not an experiment holding the inner atmosphere fixed.

`stark_scope_final.json` uses the exact CPU06 D=20 fields. The maximum
outer ratio of the isolated midpoint Stark proxy to the own-branch
source-elimination factor is 0.004506. This is not a global escape fraction,
net-response bound or pressure-profile adequacy test: overlapping branches
can exchange photons and large radiative rates can cancel. Actual pressure
broadening remains untested. The separate `quadrature_final.json` records
fixed-state H-alpha transfer sensitivities; it does not solve a refined
population/thermal equilibrium.

The D=20 primary rate replay agrees within 2.23e-16 and the line-energy
identity within 1.54e-16 relative to emission. All four transport calls and
input pins pass their implementation checks. At the six non-LTE cells:

| Change | Maximum effective up/down relative change | Maximum summed n3 net relative change | Maximum partial heat change / saved thermal scale |
|---|---:|---:|---:|
| Angular order 32 to 64 | 9.89e-10 | 1.31e-8 | 5.64e-16 |
| Frequency order 8 to 16 | 8.89e-8 | 6.30e-7 | 4.47e-9 |
| Wing extent 30 to 60 | 1.57e-4 | 1.37e-2 | 3.98e-5 |

The wing change affects both the line domain and the finite-window
continuum-background replacement. At cell 5 its background H receiving
heat changes from -2.1663e-8 to -4.3954e-8 erg/cm3/s, while the line H
term changes from 3.0649e-9 to 3.0796e-9. The heat table includes only
redistribution plus H/free-free receiving terms; it is not the full thermal
residual. Large effective rates and small net differences have different
conditioning. These finite comparisons do not qualify a refined atmosphere,
observable accuracy, radial resolution or pressure broadening.
