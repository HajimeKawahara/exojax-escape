"""Apply the prototype only in this process, leaving production files intact."""
import time,importlib.util
from bench_seq_exp1 import sequential_e1
import escape.hydrogen_mixing as mixing
mixing._exponential_integral_e1=sequential_e1
spec=importlib.util.spec_from_file_location('mix_tests','/tmp/exojax-escape-coupled/tests/test_hydrogen_mixing.py');test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
start=time.perf_counter();test.test_canonical_population_gradients_include_screening_and_charge_feedback();print('canonical AD regression passed; seconds',time.perf_counter()-start,flush=True)
