#!/usr/bin/env python3
"""Run and retain a controlled Cloudy hydrogen comparison outside the package.

Cloudy is supplied explicitly. This tool neither installs it nor changes its
source, and execution success is distinct from agreement with escape.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_artifact(path):
    path = Path(path)
    return path.is_file() or Path(str(path) + ".gz").is_file()


def artifact_bytes(path):
    """Read original artifact bytes, allowing lossless archived gzip storage."""
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    return gzip.decompress(Path(str(path) + ".gz").read_bytes())


def artifact_sha256(path):
    return hashlib.sha256(artifact_bytes(path)).hexdigest()


def store_artifact(path, data):
    """Keep large output bytes compressed; receipt hashes always refer to raw bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(data) > 100_000:
        Path(str(path) + ".gz").write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_bytes(data)


def reuse_cloudy(reference, destination, config, commands):
    """Reuse only verified Cloudy output; never reuse old escape support labels."""
    reference = Path(reference).resolve()
    receipt_bytes = artifact_bytes(reference / "receipt.json")
    prior = json.loads(receipt_bytes)
    if (prior.get("status") == "running" or prior.get("exit_code") != 0
            or "cloudy" not in prior):
        raise ValueError("The reference has no completed Cloudy execution.")
    if prior["configuration"] != config or prior.get("extra_cloudy_commands", []) != list(commands):
        raise ValueError("The reference's Cloudy configuration or commands differ.")
    target = destination / "reference"
    for name, expected in prior["artifacts"].items():
        path = reference / name
        if not path.resolve().is_relative_to(reference):
            raise ValueError("Reference artifact is outside its recorded directory.")
        data = artifact_bytes(path)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Reference artifact changed: {name}.")
        store_artifact(target / name, data)
    store_artifact(target / "receipt.json", receipt_bytes)
    # An already replayed receipt keeps its primary Cloudy files one level deeper.
    model_directory = (target / prior.get("cloudy_output_directory", ".")).resolve()
    if not model_directory.is_relative_to(target.resolve()):
        raise ValueError("Cloudy output directory is outside the retained reference.")
    actual = parse_cloudy(model_directory, config["nmax"])
    if actual != prior["cloudy"]:
        raise ValueError("Parsed Cloudy results differ from the retained reference receipt.")
    return prior, actual, model_directory.relative_to(destination).as_posix()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hydrogen_input(config, commands=()):
    """Return a pure-H, fixed-T, one-zone C23 input with explicit boundaries."""
    nmax = config["nmax"]
    if not isinstance(nmax, int) or isinstance(nmax, bool) or nmax < 4:
        raise ValueError("Cloudy needs resolved n<=3 plus at least one collapsed shell.")
    for name, value in config.items():
        if name != "nmax" and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    if any("\n" in command or "\r" in command for command in commands):
        raise ValueError("Supply each extra Cloudy command as one line.")
    lines = [
        "# Controlled external-model comparison; not a planetary atmosphere.",
        f"title Pure hydrogen thin reference nmax={nmax}",
        f'blackbody {config["radiation_temperature_k"]:.17g} K dilution {config["dilution"]:.17g}',
        f'hden {math.log10(config["density_cm3"]):.17g}',
        f'constant temperature {config["temperature_k"]:.17g} K linear',
        'init file "honly.ini"', "no molecules",
        "database H-like levels resolved 3 element hydrogen",
        f"database H-like levels collapsed {nmax - 3} element hydrogen",
        "database H-like topoff off", "database H-like levels print",
        f'set dr {math.log10(config["thickness_cm"]):.17g}',
        "stop zone 1", "iterate to convergence", "print h-like population hydrogen",
        'save species densities "model.levels" "H[:]" last',
        'save species densities "model.electrons" "e-" last',
        'save species densities "model.protons" "H+" last',
        'save hydrogen conditions "model.conditions" last',
        'save line optical depths "model.line_tau" last',
        'normalize to "H  1" 6562.80A',
        "set eden convergence 0.000001",
        f'set continuum resolution {config["resolution"]:.17g}',
        *commands,
    ]
    return "\n".join(lines) + "\n"


def one_row(path):
    """Read one final saved Cloudy row; reject missing or multi-zone output."""
    rows = [line.split() for line in artifact_bytes(path).decode().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 1:
        raise ValueError(f"Expected one final zone in {path.name}; found {len(rows)}.")
    values = [float(value) for value in rows[0]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Nonfinite saved values in {path.name}.")
    return values


def parse_cloudy(directory, nmax, prefix="model"):
    """Normalize retained output, keeping measured charge separate from closure."""
    directory = Path(directory)
    output = artifact_bytes(directory / (prefix + ".out")).decode(errors="replace")
    if "Cloudy exited OK" not in output:
        raise ValueError("Cloudy did not report successful completion.")
    saved = one_row(directory / (prefix + ".levels"))
    size = 6 + nmax - 3
    if len(saved) < size + 1 or any(value != 0 for value in saved[size + 1:]):
        raise ValueError("Saved levels do not match the declared finite atom.")
    bound = saved[1:size + 1]
    conditions = one_row(directory / (prefix + ".conditions"))
    if (len(conditions) < 6 or any(value < 0 for value in bound)
            or conditions[1] <= 0 or conditions[2] <= 0 or conditions[3] < 0):
        raise ValueError("Invalid level populations or hydrogen conditions.")
    result = {
        "nmax": nmax, "depth_cm": saved[0], "bound_cm3": bound,
        "n2_cm3": sum(bound[1:3]), "n3_cm3": sum(bound[3:6]),
        "temperature_k": conditions[1], "density_cm3": conditions[2],
        "electron_cm3": conditions[3], "proton_cm3": conditions[2] * conditions[5],
        "charge_source": "hydrogen conditions (rounded output)",
        "proton_cm3_from_nucleus_conservation": conditions[2] - sum(bound),
    }
    charge_files = [directory / (prefix + suffix) for suffix in (".electrons", ".protons")]
    if any(has_artifact(path) for path in charge_files):
        if not all(has_artifact(path) for path in charge_files):
            raise ValueError("Only part of the charge-density output is present.")
        for name, path in zip(("electron_cm3", "proton_cm3"), charge_files):
            row = one_row(path)
            if len(row) != 2 or row[1] < 0:
                raise ValueError(f"Invalid charge output in {path.name}.")
            result[name] = row[1]
        result["charge_source"] = "saved e- and H+ species densities"
    return result


def escape_reference(config, *, bound_free_model="hydrogenic", include_orbital_mixing=True):
    """Evaluate the current finite atom on an edge-split continuum quadrature."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from escape.hydrogen_atom import make_hydrogen_atom, solve_atom_populations
    from escape.hydrogen_rates import EV_CGS, PLANCK_CGS, planck_occupation

    jax.config.update("jax_enable_x64", True)
    atom = make_hydrogen_atom(config["nmax"])
    nodes, weights = np.polynomial.legendre.leggauss(48)
    edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                            100 * EV_CGS / PLANCK_CGS])
    widths = np.diff(np.log(edges))
    frequency = edges[:-1, None] * np.exp(widths[:, None] * (nodes + 1) / 2)
    weight = frequency * widths[:, None] * weights / 2
    frequency = jnp.asarray(frequency.ravel())
    radiation = config["radiation_temperature_k"], config["dilution"]
    answer = solve_atom_populations(
        config["density_cm3"], atom, config["temperature_k"],
        planck_occupation(atom.line_frequency_hz, *radiation),
        planck_occupation(frequency, *radiation), frequency, jnp.asarray(weight.ravel()),
        bound_free_model=bound_free_model, include_orbital_mixing=include_orbital_mixing,
    )
    if not bool(answer.converged):
        raise ValueError("escape did not converge at the requested reference point.")
    bound = np.asarray(answer.bound_cm3)
    return {"bound_cm3": bound.tolist(), "n2_cm3": float(bound[1:3].sum()),
            "n3_cm3": float(bound[3:6].sum()), "electron_cm3": float(answer.electron_cm3)}


def run(executable, destination, config, *, commands=(), protocol=None,
        compare_escape=False, relative_tolerance=.05, timeout=120.,
        escape_bound_free="hydrogenic", escape_orbital_mixing=True,
        reference_directory=None):
    """Retain a fresh attempt, including failures; never overwrite an old run."""
    input_text = hydrogen_input(config, commands)
    if not 0 < relative_tolerance < 1:
        raise ValueError("The relative tolerance must be between zero and one.")
    if executable is None and reference_directory is None:
        raise ValueError("Supply either a Cloudy executable or a retained reference.")
    executable = Path(executable).resolve() if executable is not None else None
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    if reference_directory is None:
        (destination / "model.in").write_text(input_text)
    (destination / "driver.py").write_bytes(Path(__file__).read_bytes())
    source_paths = [Path(__file__)]
    if compare_escape:
        source_paths += sorted((ROOT / "src/escape").glob("*.py"))
    source_before = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    receipt = {
        "receipt_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "extra_cloudy_commands": list(commands),
        "command": [str(executable), "-r", "model"] if reference_directory is None else None,
        "execution_mode": "external_command" if reference_directory is None else "retained_reference",
        "cwd": str(destination),
        "source_sha256_before": source_before, "status": "running",
        "escape_model": {"evaluated": compare_escape, "bound_free": escape_bound_free,
                         "orbital_mixing": escape_orbital_mixing},
        "repository_revision": version.stdout.strip() if version.returncode == 0 else None,
        "python": sys.version, "python_executable": sys.executable,
        "cloudy_data_path": os.environ.get("CLOUDY_DATA_PATH"),
        "acceptance": {"relative_tolerance": relative_tolerance,
                       "observables": ["n2_cm3", "n3_cm3"],
                       "scope": "External-model agreement at this input, not general physical validity."},
        "limitations": ["The executable hash is recorded; its source/data origin is not inferred.",
                        "Extra commands may deliberately change the comparison physics.",
                        "No scientific Assessment is changed by this tool."],
    }
    if protocol is not None:
        protocol = Path(protocol)
        (destination / "protocol.json").write_bytes(protocol.read_bytes())
        receipt["protocol"] = {"supplied_path": str(protocol.resolve()), "sha256": sha256(protocol)}
    write_json(destination / "receipt.json", receipt)
    try:
        if reference_directory is None:
            receipt["executable_sha256"] = sha256(executable)
            with (destination / "stdout.log").open("w") as stdout, (destination / "stderr.log").open("w") as stderr:
                process = subprocess.run(receipt["command"], cwd=destination,
                                         stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            receipt["exit_code"] = process.returncode
            if process.returncode != 0:
                raise ValueError(f"Cloudy exited with code {process.returncode}.")
            receipt["cloudy"] = parse_cloudy(destination, config["nmax"])
            receipt["cloudy_output_directory"] = "."
        else:
            prior, receipt["cloudy"], directory = reuse_cloudy(reference_directory, destination, config, commands)
            receipt.update(executable_sha256=prior["executable_sha256"], exit_code=0,
                           cloudy_output_directory=directory,
                           reference_receipt_sha256=artifact_sha256(Path(reference_directory) / "receipt.json"),
                           reference_escape_reused=False)
        receipt["status"] = "completed"
        if compare_escape:
            receipt["package_versions"] = {name: importlib.metadata.version(name)
                                           for name in ("jax", "jaxlib", "numpy")}
            receipt["escape"] = escape_reference(config, bound_free_model=escape_bound_free,
                                                   include_orbital_mixing=escape_orbital_mixing)
            if any(receipt["cloudy"][name] <= 0 for name in receipt["acceptance"]["observables"]):
                raise ValueError("Relative comparison requires positive reference populations.")
            errors = {name: abs(receipt["escape"][name] / receipt["cloudy"][name] - 1)
                      for name in receipt["acceptance"]["observables"]}
            if not all(math.isfinite(value) for value in errors.values()):
                raise ValueError("Nonfinite population comparison.")
            receipt["relative_errors"] = errors
            receipt["agreement"] = "passed" if all(value <= relative_tolerance for value in errors.values()) else "failed"
    except (OSError, subprocess.TimeoutExpired, ValueError, ImportError, KeyError) as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    after_paths = [Path(__file__)] + (sorted((ROOT / "src/escape").glob("*.py")) if compare_escape else [])
    receipt["source_sha256_after"] = {str(path.relative_to(ROOT)): sha256(path) for path in after_paths}
    receipt["changed_sources"] = sorted(name for name in set(source_before) | set(receipt["source_sha256_after"])
                                         if source_before.get(name) != receipt["source_sha256_after"].get(name))
    receipt["review_ready"] = receipt["status"] == "completed" and not receipt["changed_sources"]
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    # Storage changes are lossless and happen after parsing, without changing raw hashes.
    for path in list(destination.rglob("*")):
        if path.is_file() and path != destination / "receipt.json" and path.suffix != ".gz" and path.stat().st_size > 100_000:
            store_artifact(path, path.read_bytes())
            path.unlink()
    receipt["artifacts"], receipt["artifact_storage"] = {}, {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path == destination / "receipt.json":
            continue
        relative = path.relative_to(destination).as_posix()
        name = relative[:-3] if path.suffix == ".gz" else relative
        receipt["artifacts"][name] = artifact_sha256(destination / name)
        receipt["artifact_storage"][name] = {"path": relative, "stored_sha256": sha256(path)}
    write_json(destination / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--executable", type=Path)
    source.add_argument("--reference", type=Path, help="Reuse verified Cloudy bytes from a prior attempt; do not rerun Cloudy.")
    parser.add_argument("--record", type=Path, required=True, help="New output directory; never overwritten.")
    parser.add_argument("--protocol", type=Path, help="Predeclared investigation plan retained with the run.")
    parser.add_argument("--command", action="append", default=[], help="Additional Cloudy command; repeatable.")
    parser.add_argument("--compare-escape", action="store_true")
    parser.add_argument("--escape-bound-free", choices=("hydrogenic", "kramers"), default="hydrogenic")
    parser.add_argument("--escape-no-orbital-mixing", action="store_true")
    parser.add_argument("--relative-tolerance", type=float, default=.05)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--nmax", type=int, default=5)
    parser.add_argument("--temperature-k", type=float, default=9000.)
    parser.add_argument("--density-cm3", type=float, default=1e12)
    parser.add_argument("--radiation-temperature-k", type=float, default=10000.)
    parser.add_argument("--dilution", type=float, default=.025)
    parser.add_argument("--thickness-cm", type=float, default=1e-4)
    parser.add_argument("--resolution", type=float, default=.05)
    args = parser.parse_args()
    config = {name: getattr(args, name) for name in ("nmax", "temperature_k", "density_cm3",
              "radiation_temperature_k", "dilution", "thickness_cm", "resolution")}
    try:
        receipt = run(args.executable, args.record, config, commands=args.command,
                      protocol=args.protocol, compare_escape=args.compare_escape,
                      relative_tolerance=args.relative_tolerance, timeout=args.timeout,
                      escape_bound_free=args.escape_bound_free,
                      escape_orbital_mixing=not args.escape_no_orbital_mixing,
                      reference_directory=args.reference)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({key: receipt[key] for key in ("status", "agreement", "relative_errors", "error", "changed_sources")
                      if key in receipt}, indent=2))
    return int(receipt["status"] != "completed" or receipt.get("agreement") == "failed"
               or bool(receipt["changed_sources"]))


if __name__ == "__main__":
    sys.exit(main())
