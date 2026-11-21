from pathlib import Path
from datetime import datetime, timezone
import json, sys
root=Path('/tmp/exojax-escape-coupled');sys.path.insert(0,str(root/'tools'))
from verify_cloudy_cases import evaluate_batch,evaluate_reference_refinement,sha256
fine=root/'mceg/evidence/cloudy_repair_cases_20261121_attempt02'
coarse=root/'mceg/evidence/cloudy_repair_cases_20261121_attempt01'
out=root/'mceg/evidence/cloudy_campaign_audit_20261121';out.mkdir(exist_ok=False)
report={'recorded_utc':datetime.now(timezone.utc).isoformat(),
'command':[sys.executable,str(Path(__file__))],
'qualification':'The fine campaign retained completed per-case runs and current-source comparisons. Its original batch status is failed because its in-memory auditor used plain-file existence checks for subsequently gzip-compressed artifacts. The wrapper was edited during the campaign; its exact loaded bytes were not retained. This new audit retains the original batch and receipts unchanged, verifies uncompressed hashes through the final gzip-aware auditor, and recomputes every population criterion. No Cloudy or escape calculation is rerun here.',
'comparison':evaluate_batch(fine),'reference_refinement':evaluate_reference_refinement(fine,coarse),
'fine_batch':'mceg/evidence/cloudy_repair_cases_20261121_attempt02/batch.json','fine_batch_sha256':sha256(fine/'batch.json'),
'coarse_batch':'mceg/evidence/cloudy_repair_cases_20261121_attempt01/batch.json','coarse_batch_sha256':sha256(coarse/'batch.json'),
'auditor_sha256':{path:sha256(root/path) for path in ['tools/verify_cloudy_cases.py','tools/cloudy_reference.py']}}
for path in report['auditor_sha256']:(out/Path(path).name).write_bytes((root/path).read_bytes())
(out/'audit_driver.py').write_bytes(Path(__file__).read_bytes())
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['comparison']['status'],report['comparison']['reasons'])
print(report['reference_refinement']['status'],report['reference_refinement']['reasons'])
print(max(v for row in report['comparison']['relative_errors'].values() for v in row.values()))
print(max(v for row in report['reference_refinement']['relative_changes'].values() for v in row.values()))
