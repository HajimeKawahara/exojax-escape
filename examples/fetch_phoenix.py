"""Fetch the exact public PHOENIX input used by this boundary-field experiment.

Requires only the standard library. Keep source spectra outside version control;
the inspected public archive requests citation but gives no explicit data license.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

BASE='https://ftp.gwdg.de/pub/misc/phoenix/v2.0/HiResFITS/'
FILES={
    'wavelength.fits': ('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',
        '9befb317a6d37bf8cf031e9ef1ba8a942f2f13786c953348f437aaffadf60f73'),
    'phoenix10000.fits': ('PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte10000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
        '95091fc4c83252d8c1c6b79445b8a7d9ab8853335476e9ceb98ee3714de2a961'),
}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('results/stellar_input'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
for filename,(relative,expected) in FILES.items():
    target=args.output/filename
    if not target.exists():
        temporary=target.with_suffix('.download')
        with urlopen(BASE+relative,timeout=120) as source, temporary.open('wb') as destination:
            while chunk:=source.read(1024*1024):destination.write(chunk)
        if sha256(temporary.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Unexpected checksum for {filename}; inspect changed upstream data.')
        temporary.replace(target)
    if sha256(target.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Existing {filename} has unexpected content; not overwritten.')
    print(f'Verified {target}')
