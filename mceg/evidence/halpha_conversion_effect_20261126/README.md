# Frozen-state H-alpha/Ly-beta/O I recycling

This executed conditional calculation uses the **current PR12/13 eight-cell
atmosphere** at the refined mass-loss point. Temperature, all 18 H populations,
and radial velocity are checked for exact equality between `fixed_inputs.npz`
and `complete_physical.pkl`. Electron density and all H transition rates come
from that same physical return. The older E8 rate ledger is not combined with
this state. Inputs, source files, and raw results are hashed in `review.json`.

The question is whether an H-alpha photon, after absorption in the principal
receiving cell at 1.5125 Rp, is efficiently removed by conversion through
Ly-beta. The three initial upper-state weights are set by H-alpha emitted in
the inner four cells and absorbed in that receiving cell. This is a tagged
photon response at fixed populations, not an additional spontaneous source
to add to the current radiation field and not a new atmosphere solution.

## Calculation

The transient states are H(3s,3p,3d) and the three upper O I states of its
1025-A multiplet, in each of eight cells. A second label remembers whether the
last absorbed photon was H-alpha or UV. Actual Einstein coefficients compete
with the saved density-dependent H orbital mixing, collisions to other states,
photoionization, and electron ionization. Emitted H-alpha and the seven H/O
UV components are formally transported through the same spherical shells.
The UV coefficients reproduce the current saved H/O opacity and emissivity
to approximately 8e-15 relative error before stimulated emission is removed.

For each emitting cell and line, positive absorption coefficients partition
one photon flight among receiving cell/line, true continuum absorption, the
absorbing inner core, and the outer boundary. Those fractions define a
transient transition matrix Q and terminal matrix D. The final fates are

`B = solve(I - Q, D)`.

An analogous reward matrix counts gross emissions and band conversions.
The saved H-alpha coefficients are converted from net extinction to positive
lower-population absorption. This makes a positive tagged-photon process;
stimulated photon multiplication is not modeled. The largest stimulated
correction is 3.02% in an inner cell and 0.463% in the principal receiving cell.

H/O UV resonance absorption and re-emission are followed in both directions.
O I emission to other lower levels leaves the tracked network. H collisions
to lower levels, excursions to higher shells, and ionization are separate
terminal outcomes. **These are conditional exits, not all irreversible energy
destruction**: subsequent cascades or recombination can return photons in the
full atmosphere. O I electron collisions are omitted; their total rate is at
most 0.894% of its radiative decay rate in this state. O I photoionization and
the full non-resonant cascade field are also outside this calculation.

## Results and interpretation

For one H-alpha photon absorbed at 1.5125 Rp, the baseline outer H-alpha
escape probability is 0.790087. Removing the real 3p-to-1s Einstein branch as
an isolation control changes it to 0.790621: only **0.0534 percentage point**.
Removing O I resonance opacity gives 0.790483; removing continuum absorption
gives 0.790696; doubling continuum absorption gives 0.789781.

In the baseline, UV outer escape, UV continuum absorption, and O I
non-resonant radiation together account for approximately **0.134%** of the
input photons. An excitation history contains on average 40.7 Ly-beta
emissions. Gross H-alpha-to-UV and UV-to-H-alpha conversions are approximately
14.7707 and 14.6703, respectively. Their difference, approximately 0.10037,
also includes atoms exiting the tracked network while carrying the UV label;
it is **not** the final UV escape probability. The code separately counts
those atomic exits and checks the band-conversion identity.

The larger conditional losses are H excursions to other shells (0.11436),
ionization (0.06500), and collisions to lower states (0.01509). The first two
cannot be called permanent H-alpha destruction without following the
subsequent recombination/cascade response. Removing all those atomic exits
raises outer H-alpha survival to 0.93152. These rates already exist in the
native model; this experiment does not discover a missing destruction term.

The native rate ledger provides an independent scale check. At the same
receiving cell, raw spontaneous Ly-beta production is 5.70865e8 photons
cm^-3 s^-1, whereas its effective **net** emission is only 1.16579e4.
The native H-alpha-only net n=3 gain is -1.52594e7 cm^-3 s^-1. Gross branching
therefore cannot be interpreted as an equally large extra sink. The H/O
overlap, continuum absorption, and collisional channels are already present
in the native SE closure. The new diagnostic follows repeated spatial and
inter-line recycling explicitly.

## Controls and remaining scope

The review records a joint angular/frequency/UV-wing refinement, a separate
doubled H-alpha bandwidth control, exact source/absorption photon balances,
and the expected omitted continuum-scattering encounters along the full
no-scattering histories. The latter bounds the fraction of those conditional
histories affected by an omitted scattering event; it is not a frequency
redistribution calculation. No scattering event is counted as true heating.

The joint refinement changes principal outer H-alpha survival by 1.17e-6
relative. Doubling the H-alpha bandwidth changes it by 0.1272%, and changes
the isolated Ly-beta on/off effect itself by 0.3830% relative (2.05e-6 in
absolute survival probability). Thus the small differential effect survives
this finite-bandwidth check. Photon accounting errors are below 3.7e-14.
Expected omitted scattering encounters are 1.90e-4 per input history. The
band-color identity closes to 4.5e-15 in the baseline.

The Ly-beta continuum uses the actual native line-center absorption at each
cell, held constant over the group. The factor-of-two probe measures
sensitivity, not convergence of that approximation. Transfer uses static
complete redistribution and uniform receiving-cell redistribution, so it
does not resolve the subcell absorption layer or partial redistribution.
The absorbing core, finite frequency range, and conditional terminal choices
remain explicit. Photon-number conservation is checked; equal photon counts
do not imply equal radiative energies. H-alpha-to-Ly-beta conversion exchanges
energy with the frozen n=2/n=1 reservoir, and cascade/ionization exits require
the full atomic energy ledger for thermal balance.

None of these survival probabilities predicts a proportional source-function
change or an observed fit improvement. In particular, suppressing part of an
added nonlocal radiation field does not establish improvement relative to
the native escape/shielding closure, which lacks the complete emitted
intercell H-alpha field.

Reproduce on the local audited inputs:

```sh
PYTHONPATH=src:examples OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 7 python examples/halpha_conversion_effect.py
PYTHONPATH=src:examples OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 7 python -m pytest -q tests/test_halpha_conversion_effect.py
```

The tests check normalized probabilities, transparent escape, accumulation of
small true destruction over repeated flights, rejection of an unclosed or
perfectly trapped network, and the full H/O state's conservative recycling
limit with gross forward/reverse band counts.
