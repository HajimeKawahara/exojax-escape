from pathlib import Path
import subprocess, json, hashlib
root=Path('/tmp/cloudy-causal-rates')
original=Path('/tmp/exojax-escape-coupled/mceg/evidence/cloudy_reference_20261121/thin_nmax5_fine3.in').read_text()
cases={'baseline5':'','no_lmix5':'database H-like collisions l-mixing off\n','no_induced2ph5':'set induced two photon off\n'}
for name,extra in cases.items():
 p=root/name;p.mkdir(exist_ok=True)
 text=original.replace('thin_nmax5_fine3',name)+extra
 (p/(name+'.in')).write_text(text)
 r=subprocess.run([str(root/'source/cloudy.exe'),'-r',name],cwd=p,capture_output=True,text=True)
 (p/'process_stdout.txt').write_text(r.stdout);(p/'process_stderr.txt').write_text(r.stderr)
 print(name,r.returncode,flush=True)
