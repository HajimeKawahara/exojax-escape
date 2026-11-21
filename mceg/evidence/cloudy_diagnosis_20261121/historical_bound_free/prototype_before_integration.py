"""Exploratory small-n port of Cloudy's Brocklehurst bound-free recurrence."""
from math import factorial
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

def cross_section(n, ell, relative_energy):
    energy = jnp.asarray(relative_energy)
    t2 = jnp.maximum(energy - 1, 0.)
    small = t2 < 1e-6
    safe_t = jnp.sqrt(jnp.where(small, 1., t2))
    atan_over_t = jnp.where(small, 1 - t2 / 3 + t2**2 / 5 - t2**3 / 7,
                           jnp.arctan(safe_t) / safe_t)
    k2 = t2 / n**2
    safe_k = jnp.sqrt(jnp.maximum(k2, 1e-30))
    g0 = np.sqrt(np.pi / 2) * 8 * n * (4 * n)**n / factorial(2 * n - 1)
    gk = (g0 * jnp.exp(-2 * n * atan_over_t)
          / jnp.sqrt(-jnp.expm1(-2 * jnp.pi / safe_k))
          / (1 + t2)**(n + 2))
    plus = {n - 1: gk}
    if n > 1:
        plus[n - 2] = (2 * n - 1) * (1 + t2) * n * gk
    for q in range(n - 3, -1, -1):
        plus[q] = ((4 * n**2 - 4 * (q + 2)**2 + (q + 2) * (2 * q + 3) * (1 + t2))
                   * plus[q + 1] - 4 * n**2 * (n**2 - (q + 2)**2)
                   * (1 + (q + 3)**2 * k2) * plus[q + 2])
    minus = {n - 1: (1 + t2) / (2 * n) * gk}
    if n > 2:
        minus[n - 2] = (2 * n - 1) * (4 + (n - 1) * (1 + t2)) * minus[n - 1]
    for q in range(n - 3, 0, -1):
        minus[q] = ((4 * n**2 - 4 * (q + 1)**2 + (q + 1) * (2 * q + 3) * (1 + t2))
                    * minus[q + 1] - 4 * n**2 * (n**2 - (q + 2)**2)
                    * (1 + (q + 1)**2 * k2) * minus[q + 2])
    total = jnp.zeros_like(energy)
    for lp, recurrence in [(ell + 1, plus)] + ([(ell - 1, minus)] if ell > 0 else []):
        product = jnp.ones_like(energy)
        for s in range(1, lp + 1):
            product = product * (1 + s**2 * k2)
        radial2 = (factorial(n + ell) / factorial(n - ell - 1) * product
                   * (2 * n)**(2 * (ell - n)) * recurrence[ell]**2)
        total = total + max(ell, lp) / (2 * ell + 1) * (1 + t2) * radial2
    sigma = (4 / 3 * jnp.pi * (1 / 137.035999084) * (5.29177210903e-9)**2
             * n**2 * total)
    return jnp.where(energy >= 1, sigma, 0.)

if __name__ == '__main__':
    rows = []
    for line in Path('/tmp/cloudy-reference-run/c23.01/data/hydro_pcs.dat').open():
        if line.startswith('#') or len(line.split()) != 4:
            continue
        values = line.split()
        if not values[0].isdigit() or not values[1].isdigit():
            continue
        if int(values[0]) > 5:
            break
        rows.append([float(x) for x in values])
    rows = np.asarray(rows)
    report = []
    for n in range(1, 6):
        for ell in range(n):
            data = rows[(rows[:, 0] == n) & (rows[:, 1] == ell)]
            # The production frequency window is <=100eV; include a wider margin.
            data = data[np.exp(data[:, 2]) < 1000]
            result = np.asarray(cross_section(n, ell, np.exp(data[:, 2])))
            error = np.abs(result / np.exp(data[:, 3]) - 1)
            report.append({'n': n, 'l': ell, 'max_relative_table_error': float(error.max()),
                           'threshold_cm2': float(cross_section(n, ell, 1.))})
    Path('/tmp/cloudy-bf-investigation/recurrence_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
