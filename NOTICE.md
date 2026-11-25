# Phase-combined GP spectrum for exojax-escape PR #13

Figure for https://github.com/HajimeKawahara/exojax-escape/pull/13.
The same saved conditional NUTS posterior is combined over all 104 selected
science exposures, both nights and both orders, using the original even pixels.
The 8 km/s bins span the full supported fitting interval. Each night/order
retains its own common-spectrum projection; this is not an absolute emission
spectrum or a prediction over the full original HDF5 spectral orders.

Observations derive from Zhang et al. (2026),
https://arxiv.org/abs/2606.23899v1, and the associated Zenodo archive,
https://doi.org/10.5281/zenodo.20707255, licensed CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.
The figure is a modified analysis using log contrasts, native-pixel selection,
orbital Doppler coordinates, inverse-variance combination and posterior bands.

Recipe: `examples/plot_combined_gp_halpha.py` at source commit
`21638588e75ba0d8bf92161a0e932d94f65c3ae1` in exojax-escape.
The original sampling implementation and equations are unchanged.

This tag contains only this notice and the explanatory PNG. Original stellar
input, HDF5, full chains and prediction arrays remain local.
