"""Prototype serial batching for scalar E1, preserving analytic derivatives."""
import json,time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from jax.scipy.special import exp1
from jax.custom_batching import sequential_vmap
import numpy as np

@jax.custom_jvp
@sequential_vmap
def sequential_e1(x):
    return exp1(x)

@sequential_e1.defjvp
def sequential_e1_jvp(primals,tangents):
    (x,),(tangent,)=primals,tangents
    return sequential_e1(x),-j.exp(-x)/x*tangent

if __name__=='__main__':
    points=j.asarray([.01,1.,100.])
    results={}
    for name,fn,arg in [
        ('ordinary_vmap',jax.vmap(exp1),points),
        ('sequential_vmap',jax.vmap(sequential_e1),points),
        ('ordinary_nested',jax.vmap(jax.vmap(exp1)),j.tile(points,(12,1))),
        ('sequential_nested',jax.vmap(jax.vmap(sequential_e1)),j.tile(points,(12,1))),
    ]:
        print('start',name,flush=True)
        compiled=jax.jit(fn);start=time.perf_counter();v=compiled(arg).block_until_ready();cold=time.perf_counter()-start
        start=time.perf_counter()
        for _ in range(30):compiled(arg).block_until_ready()
        results[name]=dict(cold_s=cold,warm_s=(time.perf_counter()-start)/30,value=np.asarray(v).tolist());print(name,results[name]['cold_s'],results[name]['warm_s'],flush=True)
    np.testing.assert_allclose(results['sequential_vmap']['value'],results['ordinary_vmap']['value'],rtol=1e-14)
    np.testing.assert_allclose(results['sequential_nested']['value'],results['ordinary_nested']['value'],rtol=1e-14)
    x=j.tile(points,(3,1));fn=lambda a:j.sum(jax.vmap(jax.vmap(sequential_e1))(a))
    d1=jax.jit(jax.grad(fn))(x);d2=jax.jit(jax.jacfwd(jax.grad(fn)))(x)
    np.testing.assert_allclose(d1,-j.exp(-x)/x,rtol=1e-14)
    diagonal=np.diag(np.asarray(d2).reshape(9,9));np.testing.assert_allclose(diagonal,np.asarray(j.exp(-x)*(1/x+1/x**2)).ravel(),rtol=1e-14)
    results['nested_first_second_AD']='passed'
    print(json.dumps(results,indent=2),flush=True)
