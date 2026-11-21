"""Recompute current comparison, reference controls and execution equivalence."""
from pathlib import Path
from datetime import datetime, timezone
import json, math, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_cloudy_cases import evaluate_batch, evaluate_reference_refinement, sha256
fine = ROOT / 'mceg/evidence/cloudy_repair_cases_20261121_attempt03'
previous = ROOT / 'mceg/evidence/cloudy_repair_cases_20261121_attempt02'
coarse = ROOT / 'mceg/evidence/cloudy_repair_cases_20261121_attempt01'
output = Path(__file__).parent / 'audit.json'
if output.exists():
    raise FileExistsError(output)
comparison = evaluate_batch(fine)
refinement = evaluate_reference_refinement(fine, coarse)
rows = []
for case in json.loads((fine / 'cases.json').read_text())['cases']:
    old_path = previous / case['id'] / 'receipt.json'
    new_path = fine / case['id'] / 'receipt.json'
    old, new = (json.loads(path.read_text()) for path in (old_path, new_path))
    assert old['configuration'] == new['configuration']
    errors = {}
    for name in ('bound_cm3', 'n2_cm3', 'n3_cm3', 'electron_cm3'):
        a, b = old['escape'][name], new['escape'][name]
        a, b = (a, b) if isinstance(a, list) else ([a], [b])
        assert len(a) == len(b) and all(math.isfinite(x) and x > 0 for x in a + b)
        errors[name] = max(abs(x / y - 1) for x, y in zip(b, a))
    rows.append(dict(case=case['id'], old_receipt_sha256=sha256(old_path),
                     new_receipt_sha256=sha256(new_path), relative_changes=errors))
report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
              scope='Fresh optimized-source comparison against verified retained Cloudy; no new reference calculation or empirical adjustment.',
              comparison=comparison, reference_refinement=refinement,
              previous_fine_batch_sha256=sha256(previous / 'batch.json'),
              current_fine_batch_sha256=sha256(fine / 'batch.json'),
              equivalence=rows, maximum_population_change=max(v for row in rows for v in row['relative_changes'].values()),
              audit_source_sha256={name: sha256(ROOT / name) for name in ['tools/verify_cloudy_cases.py', 'tools/cloudy_reference.py']})
output.write_text(json.dumps(report, indent=2) + '\n')
print(comparison['status'], refinement['status'], report['maximum_population_change'])
