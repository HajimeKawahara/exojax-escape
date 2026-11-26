# Frozen H-alpha velocity coupling

`review.json` records an executed formal-transfer experiment on the saved PR12/13
eight-cell state. It does not certify a coupled SE/thermal solution or a fit to
the data. `controls.json` records static-solver parity and numerical sensitivity.
The large per-source arrays remain in `/tmp/escape-halpha-velocity-effect/review/`;
their SHA-256 digest is in `review.json`.

```sh
taskset -c 6 env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python examples/halpha_velocity_effect.py \
  --input-dir /tmp/escape-e8-mdot-sphere/fine \
  --recipe /tmp/escape-corrected-observer-full-f/recipe.json \
  --output /tmp/escape-halpha-velocity-effect/review \
  --angle-order 16 --segments 8 --refine-case upper_plus_13
taskset -c 6 env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  pytest -q tests/test_halpha_velocity_effect.py
```

The formal solution uses the same impact-parameter panels and shell-volume
weights as the existing spherical solver. On each ray subsegment, each local
Voigt branch is shifted by the projected radial velocity. Its shifted profile
is used in emission, propagation opacity, and the receiving atom's absorption
integral. Thus a relative velocity changes the pumping integral itself.

All cells retain their original temperature, populations, integrated line
coefficients, and continuum. The inner surface and outer diffuse boundary are
dark. Line photons from each source cell and continuum photons are propagated
separately through the same total extinction. Local radiation is distinguished
from remote pumping; this is not a repeated-scattering closure. The reported
line absorption includes stimulated emission in its net extinction coefficient.

The prescribed upper-velocity probes add 5.5 or 13 km/s of outward radial speed
to the cells centered at or above 1.3 Rp. These are distinct from adding a
uniform line-of-sight shift to the observed upper-layer spectrum. The Doppler
treatment is nonrelativistic: profile-center shifts are retained, while
intensity and angle transformation factors of order v/c are omitted. The
continuum remains fixed over the narrow line interval. The doubled-wing check
holds continuum endpoint values beyond the saved interval; it does not add a
new continuum calculation.

The net radiative rate subtracts the frozen rest-frame spontaneous-emission
integral from total stimulated-corrected absorption. Its cancellation makes
relative changes larger than those in the individual terms. It must not be
interpreted as a percentage change in the equilibrium source function.

At the primary receiving layer, 1.5125 Rp:

| Velocity prescription | Inner-source absorption / static | Net absorption minus emission (cm^-3 s^-1) |
| --- | ---: | ---: |
| Static | 1.000000 | 5.05406e7 |
| Native radial flow | 1.000014 | 5.04965e7 |
| Native plus 5.5 km/s in upper cells | 1.000167 | 4.98103e7 |
| Native plus 13 km/s in upper cells | 0.981713 | 4.46102e7 |

The strongest tested differential velocity reduces lower-origin pumping by
1.83% and the net radiative excitation by 11.73%. These are frozen rate changes,
not source-function changes. Native flow increases lower-origin illumination
of the farther outer cells; the actual optically thick transfer does not behave
like the overlap of two isolated Gaussian profiles.

The numerical accuracy conclusion is limited to the primary upper layer and
the reported pumping integrals. At +13 km/s, doubling the wing width changes
the primary net rate by 0.0756%, but the net residual in dense lower cell 1
(1.0917 Rp) changes by **104.6%** because of cancellation between large
absorption and emission terms. The corresponding changes in lower cells 0,
2, and 3 are 13.9%, 1.66%, and 1.77%. The finite-window calculation therefore
does **not** establish convergence of net rates throughout the atmosphere.
Static-solver parity and geometric normalization also do not establish a
global photon/energy budget for the moving calculation, which is not reported.
