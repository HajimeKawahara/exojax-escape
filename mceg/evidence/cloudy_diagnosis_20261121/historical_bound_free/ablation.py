"""One paired bound-free intervention, retaining all baseline collision/radiative data."""
from pathlib import Path
import hashlib
import json
import jax.numpy as jnp
import numpy as np
from prototype import cross_section
from escape.hydrogen_atom import make_hydrogen_atom, atom_rate_coefficients, atom_milne_population
from escape.hydrogen_rates import PLANCK_CGS, LIGHT_SPEED_CGS, EV_CGS, planck_occupation
from escape.hydrogen_nlte import solve_statistical_equilibrium, _bound_ratios

root = Path('/tmp/exojax-escape-coupled')
fixture = json.loads((root/'tests/data/cloudy_hydrogen_thin_reference.json').read_text())
report = {'protocol_revision': 'f286d125d5a96b9c9ef3e3021ad409a95719510b',
          'intervention': 'Paired hydrogenic l-resolved BF photoionization/Milne replacement; all other blocks fixed.',
          'prototype_sha256': hashlib.sha256(Path(__file__).with_name('prototype.py').read_bytes()).hexdigest(),
          'results': []}
for model in fixture['models']:
    atom = make_hydrogen_atom(model['nmax'])
    edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS, 100*EV_CGS/PLANCK_CGS])
    x, w = np.polynomial.legendre.leggauss(48)
    widths = np.diff(np.log(edges))
    nu = edges[:-1,None] * np.exp(widths[:,None] * (x+1)/2)
    weights = jnp.asarray((nu*widths[:,None]*w/2).ravel())
    nu = jnp.asarray(nu.ravel())
    eta = planck_occupation(nu,10000.,.025)
    rates = atom_rate_coefficients(atom,9000.,planck_occupation(atom.line_frequency_hz,10000.,.025),eta,nu,weights)
    sigma = []
    for i, n in enumerate(np.asarray(atom.principal_n,dtype=int)):
        relative = PLANCK_CGS*nu/atom.ionization_energy_erg[i]
        if i < 6:
            ell = [0,0,1,0,1,2][i]
            sigma.append(cross_section(int(n),ell,relative))
        else:
            sigma.append(sum((2*ell+1)/n**2 * cross_section(int(n),ell,relative) for ell in range(n)))
    kernel = 8*jnp.pi/LIGHT_SPEED_CGS**2 * nu**2 * jnp.stack(sigma)
    gamma = jnp.sum(kernel*eta*weights,axis=1)
    alpha = jnp.sum(kernel*atom_milne_population(atom,9000.,nu)*(1+eta)*weights,axis=1)
    modified = rates._replace(photoionization_s=gamma, radiative_recombination_cm3_s=alpha)
    ne = model['proton_cm3_from_nucleus_conservation']
    for label, rate in [('baseline',rates), ('hydrogenic_bf',modified)]:
        answer = solve_statistical_equilibrium(1e12,rate)
        assert answer.converged
        fixed = ne*np.asarray(_bound_ratios(ne,rate))
        row = {'nmax': model['nmax'], 'case': label, 'bound_cm3':np.asarray(answer.bound_cm3).tolist(),
               'ionized_fraction':float(answer.ionized_fraction), 'fixed_ne_bound_cm3':fixed.tolist(),
               'n2_error':float(jnp.sum(answer.bound_cm3[1:3])/model['n2_cm3']-1),
               'n3_error':float(jnp.sum(answer.bound_cm3[3:6])/model['n3_cm3']-1),
               'photoionization_s':np.asarray(rate.photoionization_s).tolist(),
               'radiative_recombination_cm3_s':np.asarray(rate.radiative_recombination_cm3_s).tolist()}
        report['results'].append(row)
        print({k:v for k,v in row.items() if not isinstance(v,list)},flush=True)
Path('/tmp/cloudy-bf-investigation/ablation.json').write_text(json.dumps(report,indent=2)+'\n')
