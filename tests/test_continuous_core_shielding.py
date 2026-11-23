"""Bounded positive-core connection; not a global smooth-shielding claim."""

import jax
import jax.numpy as jnp
import numpy as np

from escape.line_escape_probability import federman_shield

jax.config.update("jax_enable_x64", True)


def test_connected_positive_branches_and_source_deviation_bound():
    damping = jnp.geomspace(1e-9, 1e3, 37)
    for join in (2., 10., 100.):
        left = federman_shield(join*(1-1e-11),damping,average_zone=False,
                              shielding_model="continuous_core")
        right = federman_shield(join*(1+1e-11),damping,average_zone=False,
                               shielding_model="continuous_core")
        np.testing.assert_allclose(left,right,rtol=4e-11)
    tau = jnp.r_[jnp.linspace(-29.9,-1e-8,31),0.,jnp.geomspace(1e-10,1e14,2001)]
    source = federman_shield(tau[:,None],damping[None,:],average_zone=False)
    variant = federman_shield(tau[:,None],damping[None,:],average_zone=False,
                             shielding_model="continuous_core")
    difference = np.asarray(variant/source-1)
    assert difference.min() >= -.017317330-1e-9
    assert difference.max() <= 3e-16
    np.testing.assert_array_equal(variant[tau<2],source[tau<2])
    # Explicitly retain the separate source jump at zero for negative columns.
    assert federman_shield(-1e-12,1.,average_zone=False,
                          shielding_model="continuous_core") > 1.
    assert federman_shield(0.,1.,average_zone=False,
                          shielding_model="continuous_core") == 1.


def test_connected_core_derivative_away_from_remaining_kinks():
    def values(logtau):
        return federman_shield(jnp.exp(logtau),1e-3,average_zone=False,
                              shielding_model="continuous_core")
    for tau in (.5,3.,30.,300.,1e7):
        point,step=np.log(tau),1e-5
        derivative=jax.grad(values)(point)
        finite=(values(point+step)-values(point-step))/(2*step)
        np.testing.assert_allclose(derivative,finite,rtol=3e-7,atol=1e-14)


def test_hydrogen_selector_changes_only_the_supplied_pump_operator():
    from escape.hydrogen_escape_probability import HydrogenEscapeProbabilityModel

    options=dict(nmax=3,quadrature_order=2,angle_order=2,line_sampling="average")
    source=HydrogenEscapeProbabilityModel([0.,1e4,3e4],**options)
    variant=HydrogenEscapeProbabilityModel([0.,1e4,3e4],
        shielding_model="continuous_core",**options)
    bound=jnp.array([[7e11,1e5,3e5,1e3,3e3,5e3],
                     [8e11,2e5,6e5,2e3,6e3,1e4]])
    temperature=jnp.array([8000.,9000.]);proton=jnp.array([3e11,2e11])
    electron=jnp.array([3.1e11,2.1e11])
    incident=source.boundary_blackbody(10000.,.025)
    def operators(model):
        return model.population_state(temperature,bound,proton,electron,incident)[
            "line_operators"]
    before=jax.jit(lambda:operators(source))()
    after=jax.jit(lambda:operators(variant))()
    np.testing.assert_array_equal(before.escape_probability,after.escape_probability)
    np.testing.assert_array_equal(before.destruction_probability,after.destruction_probability)
    relative=np.asarray(after.pump_up_s/before.pump_up_s-1)
    assert relative.min() < -1e-4
    assert relative.min() >= -.017317331
    assert relative.max() <= 3e-16
