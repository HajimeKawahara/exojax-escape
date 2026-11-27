"""Inspect published residual-file provenance without reconstructing extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def run(args):
    metadata = json.loads(args.metadata.read_text())
    published = {item["key"]: item for item in metadata["files"]}
    results = []
    for path in args.files:
        entry = published[path.name]
        algorithm, expected = entry["checksum"].split(":")
        row = dict(file=path.name, local_path=str(path.resolve()),
                   sha256=digest(path), published_checksum=entry["checksum"],
                   checksum_matches=digest(path, algorithm) == expected,
                   size_matches=path.stat().st_size == entry["size"])
        if path.suffix == ".h5":
            with h5py.File(path) as handle:
                phase, flux = handle["phase"][:], handle["flux"][:]
                valid = np.isfinite(flux)
                count = valid.sum(axis=0)
                mean = np.nansum(flux, axis=0) / np.maximum(count, 1)
                row.update(
                    datasets={key: list(value.shape) for key, value in handle.items()},
                    file_attribute_keys=list(handle.attrs),
                    dataset_attribute_keys={key: list(value.attrs) for key, value in handle.items()},
                    exposures=len(phase), phase_range=[float(phase.min()), float(phase.max())],
                    phase_strictly_increasing=bool(np.all(np.diff(phase) > 0)),
                    phase_interval_seconds_at_adopted_1p48d=[float(x) for x in
                        np.quantile(np.diff(phase) * 1.48 * 86400, [0, .5, 1])],
                    science_exposures=int(((phase >= .30) & (phase < .435)).sum()),
                    eclipse_exposures=int(((phase >= .47) & (phase <= .5)).sum()),
                    finite_flux_fraction=float(valid.mean()),
                    median_absolute_arithmetic_time_mean_ppm=float(np.median(abs(mean[count > 0])) * 1e6),
                    max_absolute_arithmetic_time_mean_ppm=float(np.max(abs(mean[count > 0])) * 1e6),
                )
        results.append(row)
    report = dict(
        schema="escape.halpha_archive_audit.v1",
        source_sha256={str(Path(__file__).resolve()): digest(__file__)},
        metadata=dict(path=str(args.metadata.resolve()), sha256=digest(args.metadata),
                      record_id=metadata["id"], doi=metadata["doi"],
                      created=metadata["created"], modified=metadata["modified"]),
        files=results,
        all_published_checksums_and_sizes_match=all(x["checksum_matches"] and x["size_matches"] for x in results),
        primary_sources=["https://arxiv.org/html/2606.23899v1", "https://doi.org/10.5281/zenodo.20707255"],
        scope=[
            "These are post-extraction, telluric-corrected, stellar-frame residuals divided by a nightly master, according to the paper.",
            "The paper describes 150-second exposures and errors based on temporal residual scatter per wavelength channel.",
            "The supplied error arrays vary by exposure and are not exactly the unscaled full-time standard deviation; their complete scaling history is unavailable.",
            "No timestamps, exposure duration, explicit air/vacuum convention, extraction covariance or master weights are stored in these HDF5 datasets/attributes.",
            "The paper lists 85 first-night exposures, while the residual archive has 84. No cause is established by this audit.",
            "Nonzero arithmetic time means alone do not demonstrate a reduction error: upstream weighting and normalization are unavailable.",
            "A time-independent multiplicative master is removed by the current common-log-reference projection, applied to both model and data.",
            "The third published H-alpha night was not retrieved for this audit; its early-phase null is not independently re-evaluated here.",
            "Archive integrity does not validate the telluric, stellar-pulsation, blaze or wavelength-calibration corrections."
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_published_checksums_and_sizes_match"]:
        raise SystemExit("Published checksum or file-size mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
