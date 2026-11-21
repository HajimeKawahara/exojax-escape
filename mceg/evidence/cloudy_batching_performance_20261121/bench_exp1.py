"""Small, bounded E1 warm-execution comparison; no production changes."""
import time,json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from jax.scipy.special import exp1

results={}
for name,func,arg in [
 ('scalar_small',exp1,j.asarray(.01)),
 ('scalar_large',exp1,j.asarray(100.)),
 ('vmap_three',jax.vmap(exp1),j.asarray([.01,1.,100.])),
 ('array_three',exp1,j.asarray([.01,1.,100.])),
]:
 print('start',name,flush=True)
 compiled=jax.jit(func);start=time.perf_counter();out=compiled(arg);out.block_until_ready();cold=time.perf_counter()-start
 for _ in range(3):compiled(arg).block_until_ready()
 start=time.perf_counter()
 for _ in range(100):compiled(arg).block_until_ready()
 warm=(time.perf_counter()-start)/100
 results[name]=dict(cold_s=cold,warm_s=warm,value=out.tolist());print(name,results[name],flush=True)
print(json.dumps(results,indent=2),flush=True)
