# Current conditional H-alpha fits

These figures replace the historical five-cell spectrum at the top of PR #12.
They use a qualified eight-cell atmospheric state and actual phase/master/mask
processing. Atmospheric parameters and populations are fixed. The plane curve
fits zonal velocity and a velocity offset; curved-ray curves fit only the offset
at zonal velocity 11.6 km/s. Numerical refinement of the curved-ray best points
is pending. The earlier 1 ppm single-phase diagnostic failure remains recorded.

The observed file's 87 samples and supplied errors are retained. There is no
amplitude scaling or fitted baseline. The residual statistic assumes diagonal
supplied errors only; it is not a validated likelihood or parameter inference.

With NumPy and Matplotlib installed, reproduce the plots using:

```sh
python plot_fit.py plot.json --output plots
python plot_profiles.py profiles.npz --output plots
```

This compact export reproduces figures and their statistics, not a fresh
coupled-atmosphere solve. Full verification archives and the original stellar
spectrum remain private and local. `provenance.json` records source and review
hashes and the precise fitted/fixed parameters. Code is MIT licensed; bundled
observed values and their plotted use are CC BY 4.0 with attribution in NOTICE.txt.
