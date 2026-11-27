# Maintaining an outer excited-hydrogen reservoir

The useful atomic direction is to increase the supply of excited hydrogen
without thermalizing its n=3/n=2 ratio. The saved atmosphere already has a low
excitation ratio. At 1.9375 Rp, n2=13.14 cm^-3 and n3/n2=0.00567, versus an LTE
ratio of 0.14046 at 7902 K. Its three H-alpha excitation temperatures are
3614–3786 K. Original free model #1 requires n2=1401.48 cm^-3, 106.63 times
larger, but its local source is only 1.32 times larger. Absolute n3 need not
decrease; keeping its ratio low while both populations grow is the constraint.

## The recovered native budget

`budget.py` restores the metal OTS and signed H/O receiving packets to the
five n=2/3 levels using the native cross sections and Milne inverse. The saved
base rate object precedes these additions. After recovery, the largest
individual level net/turnover residual in cells 4–6 is 5.2e-13. Native collision
coefficients, including density-dependent proton mixing, reproduce to 6.9e-16.
At cell 5 the receiving correction increases 2s/2p photoionization by 4.80%/4.12%.
IR photons are below these levels' photoionization thresholds. Native chemical
and charge-exchange additions act only on the ground state; no direct n=2/3
terms from that implementation are omitted here.

At 1.9375 Rp, with 2s–2p mixing canceled from the n2 shell budget:

| Channel | Fraction of n2 gross production | Fraction of n2 gross loss |
|---|---:|---:|
| Effective H-alpha exchange with n3 | 84.26% | 71.83% |
| Radiative exchange with higher levels | 10.65% | 14.35% |
| Base recombination / photoionization | 4.09% | 11.20% |
| Ground radiative exchange, principally Ly-alpha | 0.91% | 2.10% |
| Ground collisional exchange | 0.093% | 0.006% |
| Metal receiving recombination / photoionization | negligible | 0.486% |

These large H-alpha terms mostly recycle population. Combining n2+n3 removes
that internal cycle: higher-level radiative exchange supplies a **net
1.113e5 cm^-3 s^-1**, while base bound-free exchange removes 8.608e4 and ground
radiative exchange removes 1.821e4. A fixed five-state matrix decomposition
attributes 73.8% of the maintained n2 to the imposed higher-level radiative
bath, 22.1% to direct recombination, and 3.78% to ground radiative pumping.
This does not identify the earlier origin of the higher-level populations;
they themselves respond to excitation, recombination, and recycling.

The radiative matrix is the actual **effective SE operator**, not microscopic
absorption and emission counts. Native trapping eliminates the local resonant
self-cycle, appearing as a reduced downward A(escape+destruction+electron
escape) term plus shielded pumping. At this radius Ly-alpha escape is
5.82e-6 and effective downward loss is 3667 s^-1, including Fe exchange, versus
the raw A=6.26e8 s^-1. Adding an independent trapped-line pump would double
count that eliminated cycle.

## Conditional mechanisms and their limits

The five-level local solve retains the saved temperature, radiation/escape,
and ground/higher-level baths. Density experiments scale ne, protons and
external populations together and recompute screened collisions.

| Radius/Rp | Required n2 factor | Density factor matching that n2 | Local source factor |
|---|---:|---:|---:|
| 1.5125 | 4.72 | 3.15 | 0.976 |
| 1.9375 | 106.63 | 20.19 | 0.978 |
| 2.3625 | 323.15 | 36.56 | 0.981 |

Thus a denser outer structure could, at fixed radiation, increase recombination
without immediately forcing a bright LTE source. At 1.9375 Rp, D=10 gives n2
times 30.16; D=20.19 gives n3=7.47 cm^-3, about 100 times its original value,
while n3/n2 stays 0.00533. Direct recombination then accounts for 84.1% of the
fixed-bath maintained n2. The density factors are not inferred atmospheric
solutions: higher levels scale only with D here, whereas their recombination
and three-body sources can scale as D^2 and D^3 and require a new atomic solve.

Collisional thermalization is not the immediate obstacle in this conditional
experiment. The population-weighted n3-to-n2 collision rate is 150 s^-1,
versus an effective radiative rate of 1.95e7 s^-1. Individual allowed branches
would need density increases of 3.2e4–2.7e5 to equate these *fixed-radiation*
rates. These are local rate comparisons, not critical densities for the
coupled atmosphere.

Optical-depth feedback matters sooner. D=20.19 raises the local nominal-center
opacity by 106.66, taking the original cell optical depth about 0.041 to about
4.35. As a controlled sensitivity, scaling **both** effective H-alpha upward
and downward rates by 0.25 leaves n2=1376 but raises the local source to 1.34
times baseline; scaling them by 0.1 leaves n2=1332 and raises the source to
1.99 times baseline. This is not a recomputed escape law or transfer solution.
It demonstrates why the density result must be checked with line pumping,
escape, and emission from the enlarged outer volume.

At fixed baths, removing Ly-alpha escape entirely raises n2 by only 7.65%.
Removing every n2/n3 photoionization sink raises n2 by only 69.2%. Neither
simple trapping nor this deliberately optimistic sink-removal limit supplies
the required factor 107. A different nonlocal radiation or population
structure could change the baths; these bounds do not rule that out.

Increasing only the effective Ly-alpha pump requires a factor 3424 to reach
the target. The resulting n2 photoionization is 2.139e7 cm^-3 s^-1, compared
with **all 18 levels' base** radiative recombination of 3.726e5 and three-body
recombination of 2.667e4. Baseline charge-exchange return is 1.254e5 but its
forward ionization is 1.257e5, nearly canceling. The pump-only solution cannot
establish ground/proton closure. Receiving corrections outside n2/n3, notably
high-shell IR stimulated recombination, were not recovered, so these numbers
are a supply-scale warning rather than a complete ionization-budget proof.
There is also a complete differential check for the defined pump-only probe:
its five changed levels add 2.171e7 cm^-3 s^-1 of net proton production.
Unchanged receiving terms of the other levels cancel in this difference.
Keeping every other atomic/chemical bath fixed would therefore violate the
original ionization balance, irrespective of those unrecovered base terms.

## Relation to published mechanisms

García Muñoz & Schneider connect excited hydrogen, strong stellar near-UV
photoionization, recombination/cascades, and atmospheric expansion for KELT-9b.
Those atomic channels already exist here. Their coupling to density and flow,
rather than adding a duplicate sink, is the relevant hypothesis.
[Primary paper, sections II–III](https://arxiv.org/html/1910.00267v1).

Huang et al. show how stellar, collisional, and recombination-generated
Ly-alpha photons set n2 in a WASP-121b calculation, with destruction and
orbital mixing competing against recycling. Their partial redistribution
produces less trapped intensity than complete redistribution in their test;
improved Ly-alpha transfer is not automatically an increased pump. Their
different star/planet and their case-B cascade approximation do not provide
a quantitative missing source for this state.
[Primary paper, section II.7](https://arxiv.org/html/2304.07352v1#S2.SS7).

The priority is therefore a coupled density/flow/irradiation structure with
its higher-level and photon budgets re-solved. This calculation establishes
atomic compatibility under controlled baths, not force balance, thermal
balance, charge closure, a new observed-spectrum fit, or transport of a
long-lived n2 population. Excited atoms are continuously regenerated; their
fast local cycling does not determine the ionization history of the gas.

## Replay

```sh
taskset -c 2-9 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=True \
  XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1 \
  MPLCONFIGDIR=/tmp/escape-halpha-outer-atomic-mpl \
  PYTHONPATH=src:examples:/tmp/exojax-spectrum-upstream/src \
  python mceg/evidence/halpha_outer_reservoir_20261127/atomic/budget.py
```

`review.json` contains all three cell budgets and controls; the compact NPZ
contains recovered receiving coefficients. Source/input hashes are checked
before and after execution. The five-state matrix and receiving integration
also received independent read-only review by the geometry agent.
