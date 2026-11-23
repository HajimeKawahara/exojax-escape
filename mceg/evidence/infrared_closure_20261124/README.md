# Infrared closure continuity control

Run `python plot.py` with NumPy and Matplotlib to reproduce the figure from
the included derived arrays. No stellar spectrum or atmospheric boundary
array is redistributed.

The test follows a single saved Newton direction in the eight-cell atmosphere.
At the crossing, the inherited electron escape prescription switches between
net line opacity and absorption-only opacity. The three-line control replaces
H 14→10, 14→11 and 15→8 rates with profile-resolved formal transfer using the
same continuum and an explicit broad continuum scattering reservoir.

The right panel recomputes differences of all 18 H residuals in each of the six
upper cells. The original jump remains approximately 0.01707 as the input
states approach one another; the conditional transfer differences decrease
proportionally to their separation. The left panel shows the outer n=14 row.
The lower two prescribed LTE cells are excluded from this comparison.

This establishes a local cause and a continuous candidate. It does not solve
the full charge, radiation, wind or thermal problem, establish physical
scattering adequacy, or certify the other line operators.

The underlying line/continuum approximation is discussed in Netzer, Elitzur
and Ferland (1985), ApJ 299, 752–762, [doi:10.1086/163741](https://doi.org/10.1086/163741).
The source receipt and executed script hashes are in `provenance.json`;
the original downloaded paper is not included.
