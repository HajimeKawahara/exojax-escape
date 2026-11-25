# Eclipse GP figure evidence for exojax-escape PR #13

These figures accompany https://github.com/HajimeKawahara/exojax-escape/pull/13.
They show conditional fixed-atmosphere NUTS fits, marginalized time-GP
uncertainty and finite observational/synthetic controls. Sampler convergence
does not establish complete noise calibration or a validated atmosphere.

Observed residuals derive from Zhang et al. (2026),
https://arxiv.org/abs/2606.23899v1, and the associated Zenodo archive,
https://doi.org/10.5281/zenodo.20707255 (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/).
The figures apply native-pixel selection, logarithmic time contrasts,
conditional fits and display binning. They are modified analyses, not the
authors' original figures or a reproduction of their full reduction.

The plot recipes are `examples/gp_halpha.py` and `examples/gp_controls.py`
in exojax-escape at commit `f649bea4e6cf930624db65c38fa6e1e7740c1b4c`.
Equations and scope are documented in `doc/time_gp.pdf`;
compact numerical evidence is in `mceg/evidence/time_gp_20261125/review.json`.

This evidence tag contains only the three explanatory PNGs and this notice.
Original HDF5 data, the input stellar spectrum, Cloudy outputs, full chains
and full prediction arrays are not included.
