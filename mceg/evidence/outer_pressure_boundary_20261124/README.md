# Outer pressure-boundary figure

The blue and orange curves pass through the same base and five atmospheric-cell pressure values. Blue holds T/mu flat beyond the last cell center. Orange adds a pressure-only extrapolation at the 3 Rp radiative-transfer edge, then becomes constant. The last cell therefore no longer has a forced zero pressure slope or zero advective heating.

This is a conditional wind/flow control at fixed temperatures and composition, not a new thermal-equilibrium atmosphere or an observational fit. The hollow orange point represents pressure alone; it does not assign a temperature or chemical composition to an extra atmospheric cell.

Run `python plot.py` with NumPy and Matplotlib to redraw the figure from `plot_arrays.npz`. The source calculations are identified in `provenance.json`. No stellar spectra, radiation arrays, model archives, executables or compiler data are included.
