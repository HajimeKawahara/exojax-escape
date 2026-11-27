# H-alpha data-analysis audit

The [report](../../../doc/halpha_data_audit.pdf) separates actual data-support and
covariance limitations from implementation checks. No production likelihood,
mask, atmosphere or historical assessment is changed. Phase cuts and noise
variants are diagnostic sensitivities, not a selected replacement fit.

Receipts retain source/input hashes and local array locations. Public residual
files and the published mean spectrum match the saved Zenodo metadata checksums.
The physical render, stellar template and GP chains are the same saved inputs as
the preceding conditional comparison; they must be restored separately to replay
the calculations. Scripts default to the recorded local paths. The third public
night is not part of this audit.

Run from the repository root with the recorded ExoJAX source dependency available:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export PYTHONPATH=src:/tmp/exojax-spectrum-upstream/src
python examples/halpha_operator_audit.py
python examples/halpha_data_consistency.py --output mceg/evidence/halpha_data_audit_20261127/consistency --raw-output /tmp/escape-halpha-data-consistency
python examples/halpha_noise_audit.py --output mceg/evidence/halpha_data_audit_20261127/noise
python examples/halpha_archive_audit.py \
  --metadata /tmp/escape-observation-processing/zenodo_record.json \
  --files /tmp/escape-observation-processing/KELT9b_2024-08-08_line_w6565.h5 \
    /tmp/escape-observation-processing/KELT9b_2024-08-20_line_w6565.h5 \
    /tmp/exojax-escape-nlte-spectrum/data/zenodo_20707255/KELT9b_w6565_avg_p0.30-0.45.dat \
  --output mceg/evidence/halpha_data_audit_20261127/archive.json
```

The operator has explicit input-path arguments. The other diagnostic scripts
retain the campaign paths used in their receipts. Figures and compact numerical
summaries are committed; full per-exposure arrays are hashed local artifacts.
The existing 19 GP, contrast and observation-control tests pass. Successful
execution does not establish the missing extraction covariance or unique
physical attribution. Build the report with `make -C doc data-audit-pdf`.
