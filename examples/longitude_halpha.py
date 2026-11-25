"""Render a fixed-state longitude kernel from explicit saved atmospheric inputs.

Run separately for each phase and spatial grid. The plan JSON names the NPZ
``fixed_inputs`` and ``formation_arrays``, a ``recipe`` with ``edges_rp``,
``phases``, ``grid`` and ``zonal_kms``. No atmospheric equilibrium is solved.
This diagnostic retains the qualified observer's continuum and Doppler rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import voigt_profile

from escape.longitude_map import rebin_longitude_kernel, source_longitude_kernel


def render(plan_path, phase_index, output):
    start = time.monotonic()
    plan = json.loads(Path(plan_path).read_text())
    inputs = [Path(plan_path)] + [
        Path(plan[key]) for key in ("fixed_inputs", "formation_arrays", "recipe")
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    for path in inputs[1:]:
        expected = plan.get("files", {}).get(str(path))
        if expected is not None and hashes[str(path)] != expected:
            raise ValueError(f"Pinned input changed: {path}")
    fixed = dict(np.load(plan["fixed_inputs"]))
    atom = dict(np.load(plan["formation_arrays"]))
    edges = np.asarray(json.loads(Path(plan["recipe"]).read_text())["edges_rp"])
    wn = fixed["wavenumber_cm"]
    light = 299792.458
    radius = 1.9 * 7.1492e9
    rotation = 2 * np.pi * radius / (1.48 * 86400) / 1e5

    def planck(temperature):
        return (
            3.741771790075259e-5
            * wn**3
            / np.expm1(1.4387773538277202 * wn / temperature)
            / np.pi
        )

    def coefficient(cell, velocity):
        profile = voigt_profile(
            wn[None]
            - atom["branch_centers"][:, None]
            + atom["branch_centers"][:, None] * velocity / light,
            atom["branch_sigma"][cell, :, None],
            atom["branch_gamma"][:, None],
        )
        chi = (
            np.sum(atom["branch_alpha"][cell, :, None] * profile, axis=0)
            + fixed["continuum_extinction"][cell]
        )
        eta = (
            np.sum(atom["branch_eta"][cell, :, None] * profile, axis=0)
            + fixed["continuum_emission_hz"][cell] * light * 1e5
        )
        if not (
            np.isfinite(chi).all()
            and np.isfinite(eta).all()
            and np.all(chi >= 0)
            and np.all(eta >= 0)
        ):
            raise ValueError("Finite nonnegative transfer coefficients are required.")
        return chi, eta

    nodes = np.linspace(-np.pi / 2, np.pi / 2, 97)
    kernel = source_longitude_kernel(
        plan["phases"][phase_index],
        edges,
        nodes,
        coefficient,
        planck(4500),
        fixed["radial_velocity_kms"],
        radius,
        rotation,
        plan["zonal_kms"],
        plan["grid"],
    )
    sigma = light / 98000 / np.sqrt(8 * np.log(2))
    half = int(np.ceil(6 * sigma / 0.2))
    lsf = np.exp(-0.5 * (np.arange(-half, half + 1) * 0.2 / sigma) ** 2)
    lsf /= lsf.sum()
    kernel = np.array(
        [
            0.08**2 * np.convolve(row, lsf, mode="same") / (np.pi * planck(10000))
            for row in kernel
        ]
    )[:, ::-1]
    coarse = rebin_longitude_kernel(kernel, nodes, nodes[::2])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        phase=plan["phases"][phase_index],
        velocity_kms=(light * (15233.146 / wn - 1))[::-1],
        longitude=nodes,
        kernel=kernel,
        coarse_kernel=coarse,
    )
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in hashes.items()
    )
    if not unchanged:
        raise RuntimeError("An input changed during rendering.")
    review = dict(
        elapsed_s=time.monotonic() - start,
        input_sha256=hashes,
        grid=plan["grid"],
        phase=plan["phases"][phase_index],
        unchanged=unchanged,
        scope="Fixed-opacity local source redistribution; no new NLTE equilibrium.",
    )
    output.with_suffix(".json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("phase_index", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.plan, args.phase_index, args.output)
