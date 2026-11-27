# Density, flow and confinement of an outer excited-hydrogen reservoir

The useful dynamical candidate is a **slow, dense sector with a force balance**, possibly on closed magnetic field lines. A gas-density multiplier by itself is not a mechanism and does not set the n=2 multiplier or n3/n2. No new wind, thermal, population or observer fit is executed here.

The original free shape has 106.63 times more n=2 at 1.9375 Rp and 141.58 times more n=2 column above 1.725 Rp. Its local source at 1.9375 Rp is only 1.323 times the physical source. The structural clue is extended opacity at a comparable source, not a greatly colder gas. Increasing the emitting/absorbing area can increase off-limb emission; an extended reservoir is not automatically a darker observer spectrum.

## Executed scales and continuity

`scales.py` independently reconstructs mass density from the saved total gas pressure, temperature and mean molecular weight. It reproduces all eight cells' spherical mass flux, 1.0512711e13 g/s, to relative precision below 1e-12. This is the accepted +0.05 log-mass-loss snapshot, not the recipe's nominal 1e13 g/s. At 1.9375 Rp, T=7902 K, rho=2.476e-15 g/cm3, P=2.228e-3 dyn/cm2, vr=4.879 km/s and the pressure sound speed is 9.486 km/s. Hydrogen is approximately 85% ionized.

At unchanged T and composition, let D be the **gas-density** multiplier:

| D | P (dyn/cm2) | vr if the same spherical mass flux is kept (km/s) | Mass flux if vr is kept (g/s) | Surface equatorial dipole field for local beta=1 (G) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002228 | 4.879 | 1.051e13 | 1.721 |
| 3 | 0.006683 | 1.626 | 3.154e13 | 2.981 |
| 10 | 0.02228 | 0.4879 | 1.051e14 | 5.442 |
| 20.19 | 0.04498 | 0.2416 | 2.123e14 | 7.733 |
| 100 | 0.2228 | 0.04879 | 1.051e15 | 17.21 |

D=20.19 is included as the companion atomic calculation's illustrative fixed-radiation probe, **not an inferred density**. That calculation changes the local rates without solving the new optically thick radiation field or thermal/wind structure. Its dynamical bookkeeping would require a 66.4-hour cell crossing time at fixed mass flux, compared with 3.29 hours now. A closed region need not carry the prescribed global escaping mass flux at all.

The free shape's speed at this radius is 3.624 km/s. Replacing only the physical speed with that value, at equal mass flux and area, increases gas density by merely 1.346; the n=2 response still requires its atomic balance. This modest slowdown alone cannot be equated with the factor 107 in n=2. More generally, rho=mdot/(Omega_flow r2 vr). Concentrating a fixed mass flux into a smaller flow solid angle increases local density but reduces the area carrying it. Flow solid angle, the n=2 fraction and projected covering fraction are distinct variables. Lowering mass loss at unchanged speed lowers density.

The current tidal transonic solution selects vr through momentum balance and its regular sonic condition. A slowdown requires new pressure gradients, field geometry or an external boundary; it cannot be applied independently. Maintaining D=20.19 at the old speed would raise the minimum base-to-L1 potential-work scale from 9.08e25 to 1.83e27 erg/s. This is energy bookkeeping, not an energy-limited bound: the thermal budget and available heating must be recomputed. External pressure can suppress a dayside wind in suitable models ([Murray-Clay et al. 2009](https://arxiv.org/abs/0811.0006)).

## What magnetic confinement would and would not establish

For the saved gas, B_beta=1=sqrt(8 pi P)=0.2366 G locally. With an ideal centered equatorial dipole, B_surface=B_local(r/Rp)^3=1.721 G. Polar surface field is twice this equatorial value. Gas plus the current radial ram pressure gives 1.935 G instead. For D=20.19 with the slower continuity speed, the corresponding values are local 1.063 G and surface 7.735 G including ram pressure. These are conditional pressure scales, **not measured fields or sufficient confinement criteria**. Tension, gradients across and along the field, rotation, tides, open-field loading, reconnection and the external stellar environment decide whether a closed configuration exists.

Closed-field regions can retain denser gas than adjacent escaping streams in model hot Jupiters, including gas outside a hydrodynamic critical equipotential ([Trammell et al. 2011](https://arxiv.org/abs/1011.0017)). Radiation-MHD work also shows strong sensitivity to field topology, external fields and heating ([Owen & Adams 2014](https://arxiv.org/html/1408.3636v1)). Their field thresholds are not transplanted to this much denser saved state. No host-star or planetary field detection is assumed here.

Using the approximate charge-exchange momentum rate in Trammell et al., Appendix B, Table 2, the bulk H-proton coupling time is 0.105 s at 1.9375 Rp; the H thermal mean free path is 0.85 km, about 1.5e-5 of the cell width. Thus bulk neutral-ion coupling is not obviously the immediate obstacle at the saved ionization fraction. This does not trap a particular n=2 atom indefinitely or prove negligible long-term ambipolar leakage. If a new shielded reservoir becomes much less ionized, the rates and drift must be recalculated.

## Roche geometry is already partly included

The saved driver includes the outward 3 Omega_orb2 r tidal term. The missing candidate is its three-dimensional geometry and changed flow boundary, not adding that term again. Using the actual model mass 2.17 Mj, Rp=1.9 Rj and period 1.48 d gives the Hill/L1 approximation r_L1=2.4742 Rp. The free shape's static radial tau=1 radius, 2.2464 Rp, lies inside that star-planet-axis distance.

This does **not** put a spherical shell at 2.25 Rp inside the critical equipotential in every direction. In the same Hill approximation, its transverse and polar intercepts are 1.6495 and 1.5782 Rp; its radial intercept toward the current observer direction is 1.8145–1.8668 Rp across the target phases. These are geometrical potential-surface intercepts, not line-formation radii, volume-equivalent Roche radii or magnetic boundaries. No Eggleton radius is computed because this calculation does not introduce a new stellar-mass/separation pair. The saved sonic radius, 2.2434 Rp, is another distinct quantity. A slow reservoir at these heights needs a directional force balance, not a spherical hydrostatic assumption.

## Clumps and external compression: useful controls with competing effects

At fixed mean mass, optically thin microclumps with density contrast D and volume filling factor 1/D increase volume-integrated recombination by D if ionization and temperature stay fixed. Under a **fixed n=2 fraction**, they do not increase the mean linear n=2 column at all. If n=2 instead rises nonlinearly with density, both opacity and recombination-fed line emission change. This must be solved, not represented by multiplying opacity alone. Opaque clumps saturate: their attenuation is limited by projected covering, which is not their volume filling factor.

At unchanged T and composition, D-fold denser clumps are D-fold overpressured. Without another stress, isobaric D=20.19 would require about 391 K, showing why a warm atomic clump cannot simply be assumed to remain in pressure equilibrium. Shielding, ionization-dependent molecular weight, thermal stability and conduction would all matter; this is not a predicted clump temperature. Ram compression also creates heat. For illustration only, balancing the D=20.19 cell pressure with a purely proton stellar wind at 400 km/s requires about 1.68e7 cm-3 at the interaction, versus 1.05e6 cm-3 for the saved state. No such stellar-wind condition is measured or imposed.

Three-dimensional radiation hydrodynamics supports qualitatively different compressed streams and tails as external pressure changes ([McCann et al. 2019](https://arxiv.org/abs/1811.09276)). It does not establish cool, opaque clumps here. In another calculation external winds changed Lyman-alpha much more than H-alpha because the outer gas contributed little to H-alpha ([Mitani et al. 2022](https://arxiv.org/abs/2111.00471)). Transit-tail interpretations of KELT-9b require distant excited hydrogen and admit stellar alternatives; they do not establish coverage of this dayside emitter ([Lowson et al. 2023](https://arxiv.org/abs/2301.10406)).

## Priority and status

1. **Test a force-supported, slow dense sector together with its excitation.** Closed planetary field lines offer a concrete candidate with conditional fields of order several to tens of gauss at the surface for the illustrative densities. Nonmagnetic external-pressure confinement is another boundary choice. Neither is established by these pressure estimates.
2. **Test density and illumination jointly before choosing that force model.** The companion atomic response can show whether added gas keeps n3/n2 sufficiently small, but greater optical depth changes pumping, escape and cooling. The optical emitting area and all-phase observer spectrum must be included.
3. **Keep unresolved clumps/tails below a specified sector model.** Their sizes, covering, thermal stability and supply are unconstrained. A clumping factor alone cannot supply a dark absorber.

The minimal disfavored shortcuts are faster radial flow at unchanged supply (less density), lowering mass flux at unchanged speed, or assuming the free model's modest slowdown explains its large n=2 excess. No magnetic or stellar-wind hypothesis is excluded by a new full solution here; none has been demonstrated to improve the observed residual. Radiative and thermal feedback, full MHD stability and matched native-observer tests remain unperformed.

Replay: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src taskset -c 18-25 python mceg/evidence/halpha_outer_reservoir_20261127/dynamics/scales.py`. Inputs, constants, conditional arrays and checks are pinned in `scales.json`; primary-source scope is recorded in `literature.json`.
