# Radiation conditions for an extended excited-hydrogen reservoir

The specific comparison is the column above `1.725 Rp`: the physical state
has `N2=8.405e10 cm^-2`, whereas original free model #1 has
`N2=1.190e13 cm^-2`, a factor `141.6`. The free outer source is already
similar to the physical source; the target is not an arbitrary suppression
of every upper-layer source function. These diagnostic columns do not imply
that a spherical reservoir would improve the observed spectrum.

`quantify.py` evaluates coefficients and incident-boundary integrals at the
exact saved PR13 state. `review.json` stores quantities and source/input
hashes. No equilibrium, transfer field, hydrodynamics or observation is
re-solved. `controls.py` independently checks arithmetic, thresholds and
stellar-velocity-kernel refinement. Native stored base rates exclude later
receiving PI/RR packets; the separate atomic campaign reconstructs those
packets and checks the full selected-level balance.

The implementation audit covers `src/escape/solar.py:433` (incident boundary),
`hydrogen_escape_probability.py:264` (retained hydrogen continuum) and `:623`
(incident plus continuum-diffuse line pumping), `line_escape_probability.py:113`
(Hummer–Kunasz Lyα closure), `iron_lyalpha.py:1` (local PRD-width H/Fe exchange),
and `metal_network.py:410` (retained metal continua). Their hashes are in
`review.json`; this is not a claim of full frequency-dependent Lyα transfer.

## Which radiation can act separately?

For orientation, the relevant couplings are:

| Radiation | Principal role here | Competing consequence |
| --- | --- | --- |
| Lyα, 10.2 eV | Excite 1s→2p; scattering can retain/reallocate photons | Escape and true absorption terminate recycling; excitation is not a supply of new hydrogen nuclei |
| Balmer continuum, above about 3.4 eV | Photoionize n=2 and heat the gas | Suppressing it changes electron/proton density, cooling and recombination, not just n=2 lifetime |
| Hα, about 1.89 eV | Pump n=2→3 and couple emitter/absorber | Absorption repopulates n=3; re-emission and halo light cannot be omitted |
| Paschen continuum, above about 1.51 eV | Photoionize n=3 | It is a modest competitor to radiative depopulation in the current outer state |

In the unattenuated adopted PHX boundary, `88.1%/92.5%` of the 2s/2p
photoionization comes from 3.4–6 eV, while `93.9–98.0%` of the 3s/3p/3d
photoionization comes from 1.51–3.4 eV. Thus a wavelength-selective field
could reduce n=2 loss while leaving most direct n=3 ionization present.
This is spectral separation, not proof of an available physical filter.
The exact level thresholds differ slightly from the rounded band edges.

At cell 5, direct-boundary n=3 ionization is only `2.75%, 0.70%, 0.13%`
of the saved effective total bound-bound radiative outflow from 3s/3p/3d.
The saved base-field photoionization, which also includes diffuse/OTS
contributions, gives corresponding ratios about `3.99%, 1.04%, 0.20%`.
An ordinary percent-level optical/IR change therefore cannot be assumed
to halve n=3. These comparisons are fixed-state scales, and the radiative
outflow includes transitions to higher levels as well as downward decay.

## Simple continuum shielding is insufficient

The current radial optical depth above `1.725 Rp` at 3.41 eV is
`1.156e-6`; its H(n=2) contribution is `1.155e-6`. At the free-model target
column, retaining the physical 2s/2p mix, it is only `1.635e-4`.
Even at this favorable near-threshold energy, a pure n=2 screen attenuating
the beam by a factor 141.6 requires `N2=3.605e17 cm^-2`, about 30,000 times
the target column. At 4 eV the requirement rises to `5.645e17 cm^-2`.
This is only a necessary monochromatic attenuation estimate: a 141-fold
population increase does not follow unless production stays fixed and
all competing losses are negligible.

Ground-state H has zero bound-free opacity below its 13.6 eV threshold.
The same outer column is optically thick at 13.7 eV (`tau=7.16`) yet
transparent to the Balmer-ionizing photons. An EUV shield is therefore
not a Balmer-continuum shield. The retained metal continua are zero at
3.41, 4 and 5 eV; the remaining retained molecular/free-free contribution
at 3.41 eV is only `1.40e-10` in radial optical depth. At 1.89 eV the total
retained continuum optical depth is `6.61e-9`.

This does not exclude an additional dense screen or complete excited-metal
line blanketing. It states what that proposal must demonstrate: broad
coverage of the dominant 3.4–6 eV photon band, survival of the absorbing
species, consistent re-emission/heating, and sufficient Lyα or recombination
supply behind the screen. The reduced current metal network includes
ground-ion continua and selected excited continua, not every excited-metal
edge or a complete overlapping near-UV line forest. A generic grey screen
would also attenuate useful pumping and the n=3-ionizing continuum.

Balmer absorption is already an energy and population channel in this
model. García Muñoz & Schneider showed that very small Balmer-continuum
optical depths can still dominate heating around KELT-9 because of its
large NUV/EUV flux ratio; their atmosphere includes explicit Lyα transport
but simplifies other bound-bound transfer. Their result motivates checking
the coupled energy balance, not adding a second Balmer-heating term.
[Primary paper](https://arxiv.org/html/1910.00267v1)

## Nonlocal Lyα is a closure test, not a missing reaction

The current model uses Hummer–Kunasz/C23-style Lyα escape/destruction,
continuum-shielded pumping, angular-momentum-changing collisions, two-photon loss and local
H/Fe exchange. It already traps Lyα strongly: at cell 5 the escape
probability is `5.82e-6`, and its effective spontaneous escape rate is
about `3646 s^-1`. It does not solve the full angle/frequency-dependent
transport of inner-shell Lyα emission into outer shells. The optional
resolved spherical Hα group changes a different transition and does not
establish adequacy of this Lyα closure.

In a scattering-dominated limit, n2/n1 depends on the Lyα occupation,
whereas n3/n2 can depend on Hα occupation. This offers a physical route
to different excitation ratios at the two transitions. Huang et al.
explicitly transported stellar and locally generated Lyα for HD 189733b,
and found strong 2s/2p mixing and radiative control of n=2. They also
include Balmer photoionization as a photon-destruction pathway. These
are useful mechanisms, not a numerical prediction for KELT-9b.
[Primary paper, Sections III and V](https://arxiv.org/html/1711.05428v1)

The global ionization budget is restrictive. The current cell-5 2s/2p
photoionization coefficients are about `1.83e4/1.34e4 s^-1`. Their inverse
values are `55/74 microseconds` for individual 2s/2p atoms at the saved base
field. These are individual excited-atom
photoionization times, not the total hydrogen ionization time or a coupled
population-relaxation time; other losses and replenishment coexist. Maintaining
roughly `1400 cm^-3` of n=2 at that field would produce order `2e7`
ionizations per cm3 per second. Lyα redistributes excitation but does not
replace the proton/electron recombination needed to balance those
ionizations. A nonlocal-pumping proposal therefore needs a denser reservoir,
different ionization/thermal balance, or genuinely reduced NUV illumination.
The atomic campaign checks this constraint with recovered receiving rates.

PRD changes where Lyα photons escape and hence can change outer n=2.
In solar simulations, Leenaarts et al. found that full Voigt CRD in the
Lyman lines gave lower Hα opacity than their PRD treatment; Doppler/truncated
approximations behaved differently. They also found 3D Hα scattering
important. This supports a matched nonlocal/redistribution comparison;
it does not justify deleting line wings or importing a solar correction
factor into this planetary escape-probability model.
[Primary paper, Sections IV–V](https://arxiv.org/html/1202.1926v1)

Three-dimensional KELT-9b models have also included excited-state kinetics
and a trapping/diffusion treatment, with anisotropic radiation and tidal
geometry. Their transit comparison is evidence that these processes merit
coupled modeling, not evidence for the particular emission-phase screen
required here.
[Shaikhislamov et al. 2025](https://doi.org/10.1051/0004-6361/202453309)

## Direction, stellar line shape and covering geometry

The adopted incident flux ratio 0.1 corresponds to `Rstar/a=0.316` and a
stellar angular radius `18.43 deg`. At fixed normal irradiance, replacing
the beam-equivalent dilution 0.025 with the exact uniform-disk value
0.025658 changes the angle-averaged intensity by only 2.63%. The potentially
larger changes involve direction-dependent attenuation, stellar surface
spectra/velocities and partial shadows, which that scalar correction omits.

The boundary currently samples an unbroadened PHX spectrum at the line
frequencies. Illustrative uniform-disk velocity averaging at maximum
projected speeds 77.4, 110 and 160 km/s increases the profile-weighted Hα
flux by approximately `1.39, 1.49, 1.60` relative to that unbroadened input.
This is an executed kernel experiment, not a reconstruction of the
planet-view stellar spectrum. It shows that restoring velocity averaging
need not lower Hα pumping; for this supplied line it fills the deep core.
The 77.4 km/s case is comparable to the transverse orbital-speed scale
`244.8(Rstar/a)`, without asserting its exact stellar-disk velocity pattern.
Using the observed Earth-view rotational profile as the planet-view input
would likewise require a viewing-geometry calculation.

Gravity darkening changes the incident spectral shape along the polar
orbit. Ahlers et al. estimate approximately 10% UV and 1–2% visible orbital
variation. Such modulation may interact with a nonlinear atmosphere, but
alone is far below the required column enhancement. The resolved surface
Hα profile and planet-view Doppler distribution have not been inferred
from those broad-band percentages.
[Primary paper, Section IV.2](https://arxiv.org/html/2004.14812v1)

A bright inner sphere of radius 1.3 Rp subtends dilution factors
`0.244, 0.129, 0.0825, 0.0577` at the four upper-cell centers. Directional
obscuration could reduce this inner Hα supply while diffuse Lyα arrives
by other paths. Those solid angles omit local emission and scattering;
the previous fully spherical Hα population calculation increased the
upper source, so geometry alone is not a demonstrated suppressor.

A planetary shadow is another way to remove direct NUV, but the selected
phase sees the star-facing hemisphere at a roughly 38–45 degree phase
angle. Under a spherical bright-emitter bound, an absorber at 1.94 Rp
covering the 1.3 Rp emitter lies within about 87 degrees of the star-facing
axis. An anti-stellar shadow is consequently not automatically the
foreground screen that hides the bright dayside. Tail/sector/other-screen
geometries need explicit covering and radiation calculations.

Finally, increased n2 at fixed n3/n2 increases Hα emissivity as well as
opacity. A thick spherical shell gains emitting projected area and can
fill the unresolved core through halo emission. A restricted foreground
sector could reduce that competition, but absorbed photons still feed n3
and must escape, convert, or transfer energy consistently. No cold,
non-emitting absorber has been introduced here.

## Priority

The most specific radiation follow-up is a photon- and energy-conserving
Lyα/Hα transport comparison in a denser, spatially restricted outer sector,
with the same Balmer/Paschen photoionization and recombination channels.
Compare the existing Lyα escape closure with spatial/frequency transport,
and track the complete ionization budget and halo emission. Simple
ground-H shielding, n2 self-shielding at the target column, a scalar
finite-disk correction, or an arbitrarily deeper stellar Hα line do not
provide the missing reservoir by themselves.

## Validation

All six declared checks in `controls.json` pass: unchanged inputs/sources,
ground/n=2 thresholds, independent column-opacity arithmetic, partitioned
photoionization integrals, and stellar-kernel refinement. The arithmetic
relative errors are below `2e-15`; doubling the velocity and angular
resolution changes the stellar broadening ratios by at most `3.4e-8`
(declared limit `0.002`). The column, finite-disk, shadow and inverse-rate
estimates remain conditional diagnostics; these checks do not validate a
new atmosphere or predict an observed improvement.
