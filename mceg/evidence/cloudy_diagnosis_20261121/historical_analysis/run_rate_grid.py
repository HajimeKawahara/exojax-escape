from pathlib import Path
import subprocess,json
root=Path('/tmp/cloudy-causal-rates')
original=(root/'baseline5/baseline5.in').read_text()
rows=[]
for temperature in [6000,9000,14000]:
 for log_density in [8,10,12,14]:
  name=f'rates_T{temperature}_n{log_density}';p=root/name;p.mkdir(exist_ok=True)
  text=original.replace('baseline5',name).replace('9000 K linear',f'{temperature} K linear').replace('hden 12',f'hden {log_density}')
  text='\n'.join(l for l in text.splitlines() if not l.startswith(('save continuum','save optical depths')))+'\n'
  (p/(name+'.in')).write_text(text)
  r=subprocess.run([str(root/'source/cloudy.exe'),'-r',name],cwd=p,capture_output=True,text=True)
  (p/'process_stdout.txt').write_text(r.stdout);(p/'process_stderr.txt').write_text(r.stderr)
  print(name,r.returncode,flush=True)
