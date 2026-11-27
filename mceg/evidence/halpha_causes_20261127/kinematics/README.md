# Frozen velocity and longitude diagnostics

The target consists of the same 12 August-8 exposures at phase 0.37614958–0.39322256 in the sealed, complete blue/core/red maps. The original PR13 atmosphere, all native observations, masks, full-data reference, GP hyperparameters, and original shared RV/amplitude define the primary comparison. Each changed spectrum reconditions the GP mean using all data. Separate RV/amplitude profiles retain the same GP covariance and compare against the correspondingly profiled baseline for each pixel interleave.

`review.json` keeps 13 representative cases, both pixel interleaves, individual order responses, all supported phase bands, and a short table for every tested case. Unsupported bands and the absent second-night target are null. `full_review` and `raw` identify the complete external arrays and receipt by hash. `controls.json` records numerical refinement; `kinematic_directions.pdf` and `.png` show fixed-nuisance response directions and their changes outside the target. Positive `Q_improvement` means lower Q; it is a conditional descriptive score, not a significance.

Longitude maps redistribute positive local source and emitting-boundary terms at fixed extinction, using the actual curved disk/halo kernels and their existing opaque-core geometry. Their dayside longitude mean is one. A Gaussian patch has width 15 degrees and a specified peak/background contrast; its center uses the implemented longitude convention. No occulting screen or new eclipse geometry is introduced. The smooth all-data map fits two longitude coefficients together with the original two nuisance parameters using all native data.

Upper radial probes add 5.5 or 13 km/s to the actual radial vector in cells above 1.3 Rp. The observer transfer recomputes each ray's projected velocity. These frozen density/population states do not conserve a new wind mass flux and are not dynamical solutions. The separate upper-LOS probes shift line absorption and emission uniformly in the observer direction; they are useful alignment controls, not a specified three-dimensional circulation. A common planetary shift is already degenerate with the fitted shared RV.

The observer shape effects here differ from the earlier moving-medium pumping experiment. Neither a favorable shape change nor a change in absorbed photon rate establishes a new statistical-equilibrium source function. The source-only experiments also do not provide a physical response to variable illumination: solving populations/thermal balance would still be necessary. The current direction ranking cannot establish why one short phase interval differs, particularly with its previously measured time-block uncertainty.

## Findings

The target baseline residuals are blue +590.52, core -585.45, red -215.84 ppm for the primary pixels. Values below are changes in planet plus reconditioned GP at the original RV/amplitude; the desired blue/core direction is positive/negative.

| Frozen direction | Blue change (ppm) | Core change (ppm) | Red change (ppm) | Profiled Q improvement, primary / alternate |
| --- | ---: | ---: | ---: | ---: |
| Longitude patch +30 deg, 4:1 peak/background | +218.66 | -26.72 | -145.08 | -13.58 / -22.49 |
| Longitude patch +30 deg, 10:1 peak/background | +370.44 | -45.23 | -245.86 | -40.97 / -51.93 |
| Uniform upper LOS +5 km/s | +80.53 | +88.35 | -219.93 | +20.40 / -7.83 |
| Actual upper radial increment +5.5 km/s | -150.58 | +29.25 | +156.25 | -27.22 / -15.33 |
| Actual upper radial increment +13 km/s | -419.47 | +296.34 | +245.97 | -72.83 / -59.08 |

The positive brightness patch enhances blue substantially but only weakly deepens the core, changes other phases, and worsens the full-data score. The smooth all-data longitude fit selects only 1.50:1 source contrast, gives primary Q improvement 0.46 and alternate -0.97, and leaves the core residual large. Its 0.46 score change is comparable to the phase-grid sensitivity (0.34); it is not a preference for that geometry. The favorable primary profiled score for the upper-LOS control does not persist on alternate pixels and leaves a large core residual. The actual radial increments move blue and core in the opposite direction to the required change.

Ray refinement changes the +13 radial and +5 LOS target bands by at most 0.51 and 0.84 ppm. The strongest longitude patch changes by at most 9.71 ppm under ray refinement, 0.84 ppm under 49/97-node longitude refinement, and 2.66 ppm under phase refinement. The full fine LOS control yields primary profiled Q improvement 20.31. These checks support the qualitative direction ranking, not sub-unit score claims. Nuisance profiles use two local optimizer starts and the longitude fit uses three; successful convergence does not prove a global optimum.

## Replay

Run from the repository root with the archived inputs listed in the receipt present. Raw spectra stay outside Git. The independent radial and LOS render jobs can be distributed over cores 18–25; each should use one BLAS thread.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/halpha-cause-mpl
export PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src
out=mceg/evidence/halpha_causes_20261127/kinematics
raw=/tmp/escape-halpha-cause-kinematics
for phase in 0 1 2 3 4 5 6; do
  for speed in 5.5 13; do
    taskset -c 18 python examples/halpha_cause_kinematics.py --output "$out" --raw-output "$raw" --render-radial-kms "$speed" --phase-index "$phase" --grid coarse
  done
  taskset -c 18 python examples/halpha_cause_kinematics.py --output "$out" --raw-output "$raw" --render-radial-kms 13 --phase-index "$phase" --grid fine
  taskset -c 18 python "$out/refine.py" --render-los-phase "$phase"
done
for grid in coarse fine; do
  taskset -c 18 python examples/halpha_cause_kinematics.py --output "$out" --raw-output "$raw" --render-radial-kms 0 --phase-index 3 --grid "$grid"
done
taskset -c 18-25 python examples/halpha_cause_kinematics.py --output "$out" --raw-output "$raw"
taskset -c 18-25 python "$out/finalize.py"
taskset -c 18-25 python "$out/refine.py"
```

The source kernels and finite LOS cache are existing computed profiles, not analytic Gaussian replacements. Ray, longitude-node, and phase refinement are distinct controls. The phase control uses identical coarse rays on both phase grids; it does not establish atmospheric radial convergence. The independently recomputed zero-increment spectrum differs from the archive by 0.00655 ppm, while recomputed branch coefficients agree to about 2e-16 relative. Thus numerical reproduction is stated to 0.01 ppm rather than bitwise equality.
