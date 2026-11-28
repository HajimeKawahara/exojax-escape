"""Optional checks of native records and selected provider correspondences.

Expressions are read from the generated native snapshot. These tests do not
supply a second maintained scientific definition or renew any Assessment.
"""
from functools import lru_cache
from hashlib import sha256
import json
from math import lgamma, pi
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
sp = pytest.importorskip("sympy")
pytest.importorskip("mceg.schema_v06")

from mceg import load_contract, mathematical_model_digest, to_sympy
from mceg.check import check_contract

from escape import coupled_hydrogen, hydrogen_atom, hydrogen_rates, parker
from escape.hydrogen_bound_free import _orbital_cross_sections
from escape.hydrogen_continuum import continuum_coefficients
from escape.radiation import moment_geometry, solve_moment_transfer
from escape.reactive_flow import spherical_shell_volume

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
CONTRACT = load_contract(DIRECTORY / "contract.yaml")
CONSTANTS = dict(h=hydrogen_rates.PLANCK_CGS, k=hydrogen_rates.BOLTZMANN_CGS,
                 me=hydrogen_rates.ELECTRON_MASS_CGS, c=hydrogen_rates.LIGHT_SPEED_CGS)


@lru_cache
def equation(suffix):
    matches = [e for e in CONTRACT.equations if e.id.endswith(suffix)]
    assert len(matches) == 1
    return to_sympy(matches[0].sympy)


@lru_cache
def right_side(suffix):
    expression = equation(suffix).rhs
    symbols = sorted(expression.free_symbols, key=str)
    return tuple(str(s) for s in symbols), sp.lambdify(symbols, expression, "numpy")


def rhs(suffix, **values):
    names, function = right_side(suffix)
    return function(*(values[name] for name in names))


def evaluated(expression, values):
    replacements = {atom: values[str(atom)]
                    for atom in expression.atoms(sp.Symbol, sp.Function, sp.Derivative)
                    if str(atom) in values}
    return float(expression.xreplace(replacements))


def test_native_pins_source_and_local_assessments():
    lock = json.loads((DIRECTORY / "imports.lock.json").read_text())
    for item in lock["imports"]:
        pin = item["model"]
        assert mathematical_model_digest(CONTRACT, pin["id"], pin["revision"]) == pin["sha256"]
    assert len(CONTRACT.mathematical_models) == 7
    assert len(CONTRACT.equations) == 39
    assert all(a.id.startswith("escape.planetlib.") and a.status == "unknown"
               for a in CONTRACT.assessments)
    for assessment in CONTRACT.assessments:
        report = check_contract(DIRECTORY / "contract.yaml", {}, assessment.id, root=ROOT)
        assert not [c for c in report["checks"] if c["status"] == "FAIL"]
        assert next(c for c in report["checks"] if c["name"] == "selected_files")["status"] == "PASS"
    for subject in CONTRACT.subjects:
        for path, digest in subject.selected_files.items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("temperature,mass,mu", [(8000., .7, .6), (12000., 2., 1.)])
def test_parker_cgs_radius_and_velocity_derivative(temperature, mass, mu):
    a = np.sqrt(parker.kB * temperature / (mu * parker.PROTON_MASS_CGS))
    gm = parker.G * parker.MJ * mass
    rs = float(parker.sonic_radius_cm(temperature, mass, mu))
    np.testing.assert_allclose(rs, rhs("parker.sonic_radius", gm=gm, a=a), rtol=2e-15)
    velocity = lambda radius: 1e5 * parker.parker_velocity(radius, temperature, mass, mu)
    slope = float(jax.grad(velocity)(rs))
    np.testing.assert_allclose(slope, rhs("parker.accelerating_slope", a=a, r_s=rs), rtol=1e-13)
    relation = equation("parker.reduced_wind")
    for radius in [0.7 * rs, 1.3 * rs]:
        substitutions = {"Derivative(v(r), r)": float(jax.grad(velocity)(radius)),
                         "v(r)": float(velocity(radius)), "r": radius, "gm": gm, "a": a}
        np.testing.assert_allclose(evaluated(relation.lhs, substitutions),
                                   evaluated(relation.rhs, substitutions), rtol=1e-12)
    # A missing km/s conversion is a meaningful negative control.
    assert not np.isclose(float(parker.parker_velocity(rs, temperature, mass, mu)), a)


def test_saha_milne_rates_and_continuum_with_common_kernels():
    atom = hydrogen_atom.make_hydrogen_atom(5)
    temperature = 9500.
    frequency = np.array([3.3e15, 4.8e15, 8e15])
    weights = np.array([2e14, 3e14, 4e14])
    saha = rhs("balance.saha_factor", **CONSTANTS, T=temperature,
               g=np.asarray(atom.statistical_weights), binding=np.asarray(atom.ionization_energy_erg))
    np.testing.assert_allclose(hydrogen_atom.atom_saha_factors(atom, temperature), saha, rtol=5e-15)
    inverse = rhs("balance.milne_factor", **CONSTANTS, T=temperature, S=saha[:, None], f=frequency)
    np.testing.assert_allclose(hydrogen_atom.atom_milne_population(atom, temperature, frequency), inverse, rtol=5e-15)
    occupation = rhs("balance.planck_occupation", **CONSTANTS, T=temperature, f=frequency)
    sigma = np.asarray(hydrogen_atom.atom_cross_sections(atom, frequency))
    gamma, alpha = hydrogen_atom.atom_bound_free_rates(atom, temperature, occupation, frequency, weights)
    expected_gamma = rhs("balance.photoionization_integrand", **CONSTANTS, f=frequency, sigma=sigma, eta=occupation)
    expected_alpha = rhs("balance.recombination_integrand", **CONSTANTS, f=frequency, sigma=sigma, eta=occupation, b=inverse)
    np.testing.assert_allclose(gamma, expected_gamma @ weights, rtol=5e-15)
    np.testing.assert_allclose(alpha, expected_alpha @ weights, rtol=5e-15)
    ne, np_ = 2e8, 3e8
    bound = saha * ne * np_ * np.linspace(.5, 2., len(saha))
    chi, emission = continuum_coefficients(atom, temperature, bound, ne, np_, frequency,
                                         include_free_free=False, include_hminus=False)
    expected_chi = rhs("balance.bound_free_extinction", sigma=sigma, b=inverse, ne=ne, np=np_, ni=bound[:, None])
    expected_j = rhs("balance.bound_free_emission", **CONSTANTS, f=frequency, sigma=sigma, b=inverse, ne=ne, np=np_)
    np.testing.assert_allclose(chi, expected_chi.sum(axis=0), rtol=1e-13)
    np.testing.assert_allclose(emission, expected_j.sum(axis=0), rtol=5e-15)


def test_line_coefficients_and_material_heating_in_actual_coupling(monkeypatch):
    captured = {}
    def capture(edges, chi, emission, inner, outer, **kwargs):
        captured.update(chi=np.asarray(chi), emission=np.asarray(emission))
        return solve_moment_transfer(edges, chi, emission, inner, outer, **kwargs)
    monkeypatch.setattr(coupled_hydrogen, "solve_moment_transfer", capture)
    model = coupled_hydrogen.CoupledHydrogenModel(
        [1e10, 1.001e10, 1.002e10], nmax=3, quadrature_order=2,
        include_free_free=False, include_hminus=False, include_orbital_mixing=False,
    )
    temperature, density = jnp.array([9000., 10000.]), jnp.array([1e5, 2e5])
    fractions = jnp.broadcast_to(jnp.array([.4, 1e-9, 1e-9, 1e-10, 1e-10, 1e-10]), (2, 6))
    boundary = model.boundary_blackbody(10000.)
    state = model.radiation_state(density, temperature, fractions, boundary, boundary, coupling=1.)
    assert state.converged
    atom, grid = model.atom, model.grid
    bound = np.asarray(density[:, None] * fractions)
    electron = np.asarray(density * (1 - fractions.sum(axis=1)))
    expected_chi, expected_j = [], []
    for cell in range(2):
        phi, _ = coupled_hydrogen.atom_line_profiles(atom, temperature[cell], grid)
        values = dict(CONSTANTS, A=np.asarray(atom.einstein_a_s)[:, None],
                      nu0=np.asarray(atom.line_frequency_hz)[:, None], phi=np.asarray(phi),
                      nl=bound[cell, atom.line_lower, None], nu_population=bound[cell, atom.line_upper, None],
                      gl=np.asarray(atom.statistical_weights)[atom.line_lower, None],
                      gu=np.asarray(atom.statistical_weights)[atom.line_upper, None])
        chi_cont, j_cont = continuum_coefficients(atom, temperature[cell], bound[cell], electron[cell],
                                                 electron[cell], grid.frequency_hz, include_free_free=False)
        expected_chi.append(rhs("line.extinction", **values).sum(axis=0) + chi_cont)
        expected_j.append(rhs("line.emission", **values).sum(axis=0) + j_cont)
    np.testing.assert_allclose(captured["chi"], expected_chi, rtol=2e-14)
    np.testing.assert_allclose(captured["emission"], expected_j, rtol=2e-14)
    relation = equation("energy.material_heating")
    for cell in range(2):
        source = float(((captured["chi"][cell] * state.mean_intensity[cell]
                         - captured["emission"][cell]) * grid.weight_hz).sum())
        expected = evaluated(relation.rhs, {"S(r)": source, "C(r)": float(state.two_photon_cooling[cell])})
        np.testing.assert_allclose(state.heating_erg_cm3_s[cell], expected, rtol=1e-13)


def test_finite_volume_radiation_uses_catalog_shell_and_flux_conventions():
    edges = np.geomspace(1e10, 2e10, 5)
    chi = np.linspace(.1, 2., 8).reshape(4, 2) / 1e10
    emission = chi * np.array([2., 3.])
    result = solve_moment_transfer(edges, chi, emission, jnp.array([1., 2.]), jnp.array([.2, .3]))
    assert result.converged
    weights = np.array([2., 3.])
    flux_relation = equation("energy.flux_moment")
    flux = np.array([evaluated(flux_relation.rhs, {"H(r)": h})
                     for h in np.asarray(result.flux_moment_edges) @ weights])
    volume = rhs("energy.shell_volume", r_left=edges[:-1], r_right=edges[1:])
    np.testing.assert_allclose(spherical_shell_volume(edges), volume, rtol=2e-15)
    np.testing.assert_allclose(moment_geometry(edges)[1], volume, rtol=2e-15)
    relation = equation("energy.cell_radiation")
    for i in range(4):
        source = float((chi[i] * result.mean_intensity[i] - emission[i]) @ weights)
        values = dict(r_left=edges[i], r_right=edges[i+1],
                      F_left=flux[i], F_right=flux[i+1], S_cell=source, V=volume[i])
        np.testing.assert_allclose(evaluated(relation.lhs, values), evaluated(relation.rhs, values), rtol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 20])
def test_catalog_coulomb_recurrence_reconstructs_provider_cross_sections(n):
    # The recurrence/seeds come only from the catalog. The final radial
    # normalization below is outside that partial model and is tested here as
    # a provider-specific projection, using ordinary floating-point arithmetic.
    energy = np.array([1., 1.01, 2., 10.])
    momentum = np.sqrt((energy - 1) / n**2)
    plus = {n-1: np.full_like(energy, rhs("coulomb.plus_top_seed"))}
    minus = {n-1: np.full_like(energy, rhs("coulomb.minus_top_seed", n=n))}
    if n > 1:
        plus[n-2] = np.full_like(energy, rhs("coulomb.plus_next_seed", n=n))
    if n > 2:
        minus[n-2] = rhs("coulomb.minus_next_seed", n=n, E=energy, Hm1=minus[n-1])
    for branch, values, stop in [(2, plus, -1), (1, minus, 0)]:
        for q in range(n-3, stop, -1):
            args = dict(n=n, q=q, E=energy, momentum=momentum, branch=branch)
            values[q] = rhs("coulomb.new_step", Anew=rhs("coulomb.new_a", **args),
                            Bnew=rhs("coulomb.new_b", **args), H1=values[q+1], H2=values[q+2])
    t = np.sqrt(energy - 1)
    ratio = np.divide(np.arctan(t), t, out=np.ones_like(t), where=t != 0)
    normalization = (.5*np.log(pi/2) + np.log(8*n) + n*np.log(4*n) - lgamma(2*n)
                     - 2*n*ratio - (n+2)*np.log(energy))
    normalization -= .5*np.log(-np.expm1(-2*pi/np.maximum(momentum, 1e-30)))
    expected = []
    for ell in range(n):
        total = np.zeros_like(energy)
        channels = [(ell+1, plus, n-1-ell)]
        if ell:
            channels.append((ell-1, minus, n-ell))
        for lp, values, power in channels:
            log_radial = (lgamma(n+ell+1)-lgamma(n-ell) + sum(np.log1p(step**2*momentum**2) for step in range(1,lp+1))
                          + 2*(ell-n)*np.log(2*n) + 2*(normalization+power*np.log(energy)+np.log(np.abs(values[ell]))))
            total += max(ell,lp)/(2*ell+1)*energy*np.exp(log_radial)
        expected.append(4/3*pi*(1/137.035999084)*(5.29177210903e-9)**2*n**2*total)
    np.testing.assert_allclose(_orbital_cross_sections(n, energy), expected, rtol=2e-13)


def test_material_cell_balance_with_prescribed_heating():
    from escape.hydrogen_nlte import RateCoefficients
    from escape.reactive_flow import solve_reactive_flow

    heating = 1e-7
    def sources(radius, density, temperature, fractions, params):
        del radius, density, temperature, fractions, params
        zero = jnp.zeros(1)
        rates = RateCoefficients(jnp.zeros((1, 1)), jnp.zeros((1, 1)), zero, zero, zero, zero)
        return rates, jnp.asarray(heating)

    edges = jnp.linspace(1e10, 1.01e10, 6)
    answer = solve_reactive_flow(edges, 1e8, 9000., 3e6, jnp.array([.4]),
                                jnp.array([0.]), 2.178709417e-11, sources, None, 1e21,
                                iterations=24)
    assert answer.valid and answer.converged
    relation = equation("energy.cell_material")
    for i in range(len(edges)-1):
        values = dict(mdot=float(answer.mass_loss_g_s),
                      B_left=float(answer.bernoulli_erg_g[i]), B_right=float(answer.bernoulli_erg_g[i+1]),
                      V=float(rhs("energy.shell_volume", r_left=edges[i], r_right=edges[i+1])),
                      S_cell=heating/(4*pi), C_cell=0.)
        np.testing.assert_allclose(evaluated(relation.lhs, values), evaluated(relation.rhs, values), rtol=1e-8)
