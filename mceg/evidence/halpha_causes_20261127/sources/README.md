# Frozen source and opacity responses

The same twelve complete exposures at phase 0.37614958–0.39322256 require
more blue signal and less core/red signal. The baseline B/C/R residuals are
`(+590.52, −585.45, −215.84)` ppm on primary pixels and
`(+402.49, −449.22, −223.31)` ppm on complementary pixels. All three bands
use the same exposures; night 1 has no complete blue support in this interval.
These are projected native log-contrast means, not absolute planetary fluxes.

`review.json` retains the full conditional results, source/input hashes,
supported phase rows, direct planet and GP responses, numerical controls and
local population interpretation. `source_responses.png`/`.pdf` show primary
fixed-RV/amplitude changes. Raw spectra and fine transfer components remain
in `/tmp/escape-halpha-cause-sources/`, with a hash in the receipt.

## Results

At fixed original RV/amplitude, with the GP mean reconditioned on all data:

| Frozen change | Primary ΔQ | Target residual B/C/R, ppm |
| --- | ---: | --- |
| Upper source ×0.5 | −17.31 | +482 / −290 / −186 |
| Upper opacity ×0.497, source fixed | +40.02 | +634 / −753 / −164 |
| Upper opacity ×2.014, source fixed | −7.18 | +596 / −567 / −215 |
| Inner source ×1.5 | +66.22 | +433 / −337 / −671 |
| Inner ×1.5 and upper ×0.5 sources | +31.85 | +325 / −41 / −641 |
| Upper opacity ×2.014, emissivity fixed | +1.19 | +482 / −259 / −160 |
| Previously solved coupled H-alpha state | +74.56 | +747 / −945 / −74 |

The all-data two-source fit gives inner/upper multipliers `1.1022/0.5461`
on primary pixels and `1.0503/0.5257` on complementary pixels. Its ΔQ is
`−21.44/−22.73`; target residuals remain `(+460, −266, −281)` and
`(+284, −145, −240)` ppm. This fit holds the original global RV/amplitude.
Both source coefficients are interior to the declared nonnegative `[0,3]`
bounds. It is a frozen spectral-shape fit, not an equilibrium solution.

Separately refitting the same two global RV/amplitude nuisance parameters
for each fixed template gives:

| Template | Primary ΔQ | Primary B/C/R, ppm | Alternate ΔQ | Alternate B/C/R, ppm |
| --- | ---: | --- | ---: | --- |
| Upper source ×0.5 | −23.73 | +444 / −221 / −294 | −19.59 | +280 / −121 / −245 |
| Inner ×1.5, upper ×0.5 | −27.28 | +416 / −262 / −264 | −19.01 | +260 / −168 / −213 |
| Upper opacity ×2.014, emissivity fixed | −13.15 | +464 / −168 / −330 | −11.81 | +277 / −63 / −273 |
| Original free model #1 | −21.83 | +184 / −165 / −184 | +0.52 | +36 / −82 / −117 |
| Previously shape-refitted free #1 | −37.75 | +168 / −371 / −113 | −2.46 | +33 / −282 / −42 |

Each ΔQ in this table uses its own selection's likewise refitted baseline.
The free models have additional prescribed shape freedom. The original
free shape improves the selected blue/core pair substantially; further
global shape fitting improves primary Q while worsening this local core.
Local agreement and full-data agreement are distinct diagnostics.

Matching only the primary B/C pair with the two source directions requires
approximately `3.828/2.391` inner/upper multipliers. Applying those finite
changes leaves B/C residuals `(+0.46, −3.06)` ppm, but the red residual is
`−2866` ppm and Q worsens by `3205`. The alternate selection similarly
leaves a `−1861` ppm red residual and worsens Q by `1294`. This demonstrates
why a selected two-band match is insufficient.

## Interpretation and implementation

The split is at `1.3 Rp`: cells 0–3 are inner and cells 4–7 upper. Source
changes multiply line emissivity at fixed extinction. Opacity changes
multiply both line coefficients, retaining the source function. Continuum,
boundary, temperatures, densities and velocities stay fixed. The emergent
intensity is linear in each emissivity component at these fixed opacities.
Changing opacity still changes transmission of the continuum and boundary.

For source multiplier `s`, opacity multiplier `t`, and
`b = n_u g_l/(n_l g_u)`, the implied populations obey
`n_u'/n_u = s t` and `n_l'/n_l = t [1 + (s−1)b]`.
In the main upper cell, source halving corresponds to `n3` halving and
about a 0.22–0.23% reduction in each branch's inferred `n2` at fixed opacity.
Its excitation temperature changes from about `4041–4077 K` to
`3584–3613 K`; these are population-ratio temperatures, not kinetic cooling.
Shared lower sublevels and the prescribed inner LTE cells constrain whether
the independent coefficient perturbations can be physically realized.
No new population, thermal, mass or momentum balance is solved.

`compare_free_shape.py` and `free_shape.json` compare the archived free #1
structure without a new fit. The original free model prescribes an `n2`
exponential scale of `0.35 Rp`, a departure sigmoid centered at `1.45 Rp`
with scale `0.08 Rp`, and 64 layers of width `0.03125 Rp`. Its upper kinetic
temperature is `14182 K`, compared with `8152 K` in physical cell 4; its
thermal width there is 1.31 times larger, not narrower.

The original free source at `1.5125 Rp` is 16.4 times the physical local
source, but drops to about 1.32 times it at `1.9375 Rp` while retaining much
more `n2`. The static center optical depth above `1.725 Rp` is `4.883`
versus `0.04555`; the corresponding radial tau-one locations are `2.246`
versus `1.592 Rp`. Thus bright and dim material occupy different radii and
columns. A uniform source multiplier above 1.3 Rp does not reproduce that
structure. These are descriptive rest-center comparisons: free #1 has
plane columns, one unresolved line, independent kinetic/wind temperatures,
and no atmospheric continuum or off-limb emission. The physical observer
has spherical rays and three branches. No single causal change is isolated.

## Numerical scope

Source-only cases use fine rays at all seven phase nodes. Opacity cases
use fine differences at both nodes bracketing all target exposures, and
coarse differences elsewhere. The saved fine baseline is retained exactly.
Baseline transfer reproduction differs by at most `0.00655 ppm`.
Coarse-to-fine target responses differ by at most `1.164 ppm` across all
probes; extending source refinement from target brackets to every phase
changes target means by at most `0.146 ppm` and Q by at most `0.117`.
These controls do not establish convergence of the eight-cell physical
atmosphere or its sharp excitation transition.

The native masks, common-reference projection, phase interpolation and GP
covariance are unchanged. The GP mean is recomputed on all retained data
for every candidate. Unsupported rows are null. Q changes are conditional
descriptions of reused data, not calibrated significances or model evidence.

Run the main script with `PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src`,
CPU JAX/x64 and single-thread BLAS. It verifies archived input hashes and
native phase/map alignment, then reuses or renders the required fine caches.
The companion comparison uses the same environment and requires no network
or atmospheric solve.
