from pathlib import Path
from hashlib import sha256
import json
import sys
import tarfile
import zipfile

dist = Path(sys.argv[1])
tarpath = next(dist.glob('*.tar.gz'))
wheelpath = next(dist.glob('*.whl'))
expected = {
    str(p) for base in ['mceg', 'tools', 'tests', 'doc']
    for p in Path(base).rglob('*')
    if p.is_file() and '__pycache__' not in p.parts and 'build' not in p.parts
    and p.suffix not in ('.pyc', '.pyo')
}
with tarfile.open(tarpath) as archive:
    members = {m.name.split('/', 1)[1]: m for m in archive.getmembers()
               if m.isfile() and '/' in m.name}
    assert not expected - set(members), sorted(expected - set(members))
    # Archive order avoids repeatedly decompressing a large gzip stream.
    for name, member in members.items():
        if name in expected:
            assert sha256(archive.extractfile(member).read()).digest() == sha256(Path(name).read_bytes()).digest(), name
with zipfile.ZipFile(wheelpath) as archive:
    names = archive.namelist()
    metadata = archive.read(next(n for n in names if n.endswith('METADATA'))).decode()
    dependencies = [line for line in metadata.splitlines() if line.startswith('Requires-Dist:')]
    assert not any('mceg' in line.lower() or 'cloudy' in line.lower() for line in dependencies)
    assert not any(n.startswith(('mceg/', 'tools/', 'tests/')) for n in names)
    sources = [p for p in Path('src/escape').rglob('*') if p.is_file() and p.suffix in ('.py', '.txt')]
    for path in sources:
        assert archive.read(str(path.relative_to('src'))) == path.read_bytes(), path
report = {
    'status': 'passed',
    'scope': 'Final source/evidence snapshot before this packaging receipt was written; distributions remain outside the repository.',
    'sdist_verified_files': len(expected), 'wheel_verified_source_files': len(sources),
    'runtime_requires_dist': dependencies,
    'distributions': {p.name: {'sha256': sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                      for p in [tarpath, wheelpath]},
}
destination = Path('mceg/evidence/distribution_check_20261121')
destination.mkdir(exist_ok=False)
(destination / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
(destination / 'verify.py').write_bytes(Path(__file__).read_bytes())
(destination / 'build.log').write_bytes(Path('/tmp/escape-dist-review-20261121.log').read_bytes())
print(json.dumps(report, indent=2))
