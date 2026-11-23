"""Rebuild the compact selected-metal tables from a local Cloudy C23.01 tree.

Run: python tools/extract_metal_data.py /path/to/c23.01
This developer tool never downloads data. Runtime uses only packaged arrays.
Readers are altered adaptations of the documented C23 table formats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np

from escape.metal_atom import Atom
from escape.metal_rates import PhotoFit

jax.config.update("jax_enable_x64", True)

WAVENUMBER_EV = 6.62607015e-27 * 2.99792458e10 / 1.602176634e-12


def _rows(path):
    for line in path.read_text().splitlines():
        row = line.split()
        if row and (row[0] == "-1" or row[0].startswith("*")):
            return
        if row and not row[0].startswith("#"):
            yield row, line


def load_atom(data_root, species, levels, collider="ELECTRON"):
    """Load one declared C23.01 data selection; never fetch or copy the data.

    These files are the sources selected by the matching Cloudy diagnostic.
    Only electron collision tables are loaded here. Other colliders, ionization,
    recombination and charge exchange remain explicitly separate obligations.
    """
    roots = {
        "Fe II": ("stout/fe/fe_2/fe_2_Smyth19", 2),
        "Ca II": ("chianti/ca/ca_2/ca_2", 2),
        "Mg II": ("chianti/mg/mg_2/mg_2", 2),
        "Si II": ("stout/si/si_2/si_2", 2),
        "C I": ("stout/c/c_1/c_1", 1),
        "Al II": ("chianti/al/al_2/al_2", 2),
        "Mn II": ("stout/mn/mn_2/mn_2", 2),
        "C II": ("stout/c/c_2/c_2", 2),
        "N I": ("stout/n/n_1/n_1", 1),
        "O I": ("stout/o/o_1/o_1", 1),
        "Fe III": ("stout/fe/fe_3/fe_3", 3),
    }
    stem, ion = roots[species]
    prefix = Path(data_root) / stem
    chianti = stem.startswith("chianti")
    suffixes = (".elvlc", ".wgfa", ".splups") if chianti else (".nrg", ".tp", ".coll")
    files = [Path(str(prefix) + suffix) for suffix in suffixes]
    raw_levels = []
    for row, line in _rows(files[0]):
        if not row[0].isdigit() or (not chianti and row[:3] == ["17", "09", "05"]):
            continue
        identifier = int(row[0])
        if chianti:
            energy, weight = float(line[40:55]), float(line[37:40])
            if energy <= 0 and identifier != 1:
                continue  # the selected reference uses experimental energies
        else:
            energy, weight = float(row[1]), float(row[2])
        raw_levels.append((identifier, energy, weight))
    raw_levels = sorted(raw_levels, key=lambda row: row[1])[:levels]
    if len(raw_levels) != levels:
        raise ValueError("Insufficient retained levels.")
    index = {row[0]: i for i, row in enumerate(raw_levels)}
    energies = np.array([row[1] for row in raw_levels])
    weights = np.array([row[2] for row in raw_levels])
    spontaneous = np.zeros((levels, levels))
    radiative_wavenumber = (energies[:, None]-energies[None, :]).astype(np.float32).astype(float)
    for row, _ in _rows(files[1]):
        if chianti:
            if len(row) < 5 or not row[0].isdigit():
                continue
            lower, upper, value = int(row[0]), int(row[1]), float(row[4])
        else:
            if row[0] != "A":
                continue
            lower, upper, value = int(row[1]), int(row[2]), float(row[3])
        if lower in index and upper in index:
            low, high = sorted((index[lower], index[upper]))
            spontaneous[high, low] += value
            if chianti:
                wavelength = np.float32(float(row[2]))
                if wavelength <= 0:
                    wavelength = np.float32(1e8 / (energies[high] - energies[low]))
                radiative_wavenumber[high, low] = np.float32(1e8 / float(wavelength))
    entries, temperature = [], None
    for row, _ in _rows(files[2]):
        if chianti:
            if len(row) < 12 or not row[0].isdigit():
                continue
            lower, upper, kind = map(int, row[2:5])
            energy, scale = float(row[6]), float(row[7])
            values = np.array(list(map(float, row[8:])))
            if kind < 6:
                values = np.maximum(values, 0)
            coordinate = np.linspace(0, 1, len(values))
        else:
            if row[0] == "TEMP":
                temperature = np.array(list(map(float, row[1:])))
                continue
            if row[0] not in ("CS", "RATE") or row[1] != collider:
                continue
            lower, upper = map(int, row[2:4])
            kind, energy, scale = (0 if row[0] == "CS" else -1), 1.0, 1.0
            values, coordinate = np.array(list(map(float, row[4:]))), temperature.copy()
            if values.shape != coordinate.shape:
                raise ValueError(
                    "Collision table has mismatched temperature/value count."
                )
        if lower in index and upper in index:
            low, high = sorted((index[lower], index[upper]))
            entries.append((low, high, kind, energy, scale, coordinate, values))
    width = max((len(row[5]) for row in entries), default=2)
    xx, yy = np.empty((len(entries), width)), np.empty((len(entries), width))
    for i, entry in enumerate(entries):
        n = len(entry[5])
        xx[i, :n], yy[i, :n] = entry[5], entry[6]
        xx[i, n:] = entry[5][-1] + np.arange(1, width - n + 1)
        yy[i, n:] = entry[6][-1]
    atom = Atom(
        *map(
            jnp.asarray,
            (
                energies,
                weights,
                spontaneous,
                [row[0] for row in entries],
                [row[1] for row in entries],
                [row[2] for row in entries],
                [row[3] for row in entries],
                [row[4] for row in entries],
                xx,
                yy,
                [len(row[5]) for row in entries],
                ion,
                radiative_wavenumber,
            ),
        )
    )
    return atom, dict(
        species=species,
        levels=levels,
        electron_table_pairs=len(entries),
        possible_pairs=levels * (levels - 1) // 2,
        files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )


def _data(path):
    for line in Path(path).read_text().splitlines():
        row = line.split("#", 1)[0].split()
        if row:
            yield row


def load_photoionization(data_root, atomic_number, charge):
    """Read one ion's seven shell fits, with C23 realnum rounding preserved."""
    z, ne = int(atomic_number), int(atomic_number - charge)
    if not 1 <= ne <= z <= 30:
        raise ValueError("Fits require 1 <= bound electrons <= Z <= 30.")
    rows = iter(_data(Path(data_root) / "phfit.dat"))
    assert next(rows) == ["20061204"]
    angular = np.array(next(rows), dtype=int)
    ninner = int(next(rows)[ne - 1])
    nouter = int(next(rows)[ne - 1])
    if z == ne and z > 18 or z == ne + 1 and z in (20, 21, 22, 25, 26):
        nouter = 7
    fit95 = np.ones((7, 6))
    exists = np.zeros(7, dtype=bool)
    for row in rows:
        shell, electrons, element = map(int, row[:3])
        if shell < 0:
            break
        if electrons == ne - 1 and element == z - 1:
            fit95[shell] = np.array(row[3:], dtype=np.float32).astype(float)
            exists[shell] = True
    fit96 = np.ones(7)
    for row in rows:
        electrons, element = map(int, row[:2])
        if electrons < 0:
            break
        if electrons == ne - 1 and element == z - 1:
            fit96 = np.array(row[2:], dtype=np.float32).astype(float)
    potential = next(
        float(row[2])
        for row in _data(Path(data_root) / "ionization_potentials.dat")
        if len(row) > 2 and int(row[0]) == z and int(row[1]) == charge
    )
    threshold = fit95[:, 0].copy()
    threshold[nouter - 1] = potential * WAVENUMBER_EV
    valid = exists & (np.arange(1, 8) <= nouter)
    if z in (19, 20) and ne >= 19:
        valid[5] = False
    if z in (15, 17, 19) or z > 20 and z != 26:
        inner_threshold = 0.0
    elif ne < 3:
        inner_threshold = 1e30
    else:
        inner_threshold = (
            threshold[nouter - 1] if ninner == nouter else fit95[ninner - 1, 0]
        )
    return PhotoFit(
        *map(
            jnp.asarray,
            (angular, fit95, fit96, threshold, valid, ninner, nouter, inner_threshold),
        )
    )


def read_fits(data):
    """Read C23 ground-parent fits with recorded data identities."""
    data = Path(data)
    rr, dr_c, dr_e = {}, {}, {}
    target = rr
    for name in ("badnell_rr.dat", "badnell_dr.dat"):
        target = rr if name == "badnell_rr.dat" else dr_c
        for line in (data / name).read_text().splitlines():
            fields = line.split()
            if fields[:4] == ["Z", "N", "M", "W"]:
                if fields[4] == "E1":
                    target = dr_e
                continue
            try:
                z, n, m, w = map(int, fields[:4])
                values = [float(v) for v in fields[4:]]
            except (ValueError, IndexError):
                continue
            if m == 1:
                target[z, n] = values
    # ADfA coefficients are stored as C++ realnum (float32), then evaluated
    # in double. Preserve that rounding for a meaningful source comparison.
    iterator = iter((data / "rad_rec.dat").read_text().split())
    next(iterator)
    old, new, iron = {}, {}, {}
    for table, count in ((old, 2), (new, 4)):
        while True:
            i, j = int(next(iterator)), int(next(iterator))
            if (i, j) == (-1, -1):
                break
            table[i, j] = np.array(
                [float(next(iterator)) for _ in range(count)], dtype=np.float32
            ).astype(float)
    while True:
        i = int(next(iterator))
        if i == -1:
            break
        iron[i] = np.array(
            [float(next(iterator)) for _ in range(3)], dtype=np.float32
        ).astype(float)
    ct_ion = np.loadtxt(data / "ctiondata.dat", skiprows=1).reshape(30, 4, 8)
    ct_rec = np.loadtxt(data / "ctrecombdata.dat", skiprows=1).reshape(30, 4, 7)
    return dict(
        rr=rr,
        dr_c=dr_c,
        dr_e=dr_e,
        old_rr=old,
        new_rr=new,
        iron_rr=iron,
        ct_ion=ct_ion,
        ct_rec=ct_rec,
        sha256={
            name: hashlib.sha256((data / name).read_bytes()).hexdigest()
            for name in (
                "badnell_rr.dat",
                "badnell_dr.dat",
                "rad_rec.dat",
                "ctiondata.dat",
                "ctrecombdata.dat",
            )
        },
    )


SPECIES = (
    ("C I", 6, 15),
    ("Mg II", 12, 15),
    ("Al II", 13, 15),
    ("Si II", 14, 15),
    ("Ca II", 20, 15),
    ("Mn II", 25, 45),
    ("Fe II", 26, 716),
    ("C II", 6, 15),
    ("N I", 7, 50),
    ("O I", 8, 29),
    ("Fe III", 26, 70),
)


def _cota_tables(source):
    text = (source / "atmdat_3body.cpp").read_text()
    blocks = re.findall(
        r"_rs\d+\[\] = \{(.*?)\};\s*for\(.*?\)\s*(\w+)\[_i\] = .*?RC_INI", text, re.S
    )
    tables = {}
    for block, name in blocks:
        values = []
        for count, value in re.findall(r"\{(\d+)L?,\s*([-+\d.eE]+)\}", block):
            values.extend([float(value)] * int(count))
        values = np.asarray(values)
        if name.startswith("ba") or name.startswith("bb"):
            values = values.astype(np.float32).astype(float)
            target = "b" + name[-1]
            tables[target] = np.r_[tables.get(target, []), values]
        else:
            tables[name] = values
    if len(tables) != 12 or len(tables["b0"]) != 83:
        raise ValueError("Unexpected C23 Cota table layout.")
    return tables


def _ground_weights(z, charge):
    # C23 atmdat_outer_shell.cpp, with its neutral/singly-ionized overrides.
    weights = [
        2,
        1,
        2,
        1,
        2,
        1,
        4,
        5,
        4,
        1,
        2,
        1,
        2,
        1,
        4,
        5,
        4,
        1,
        4,
        5,
        4,
        1,
        6,
        9,
        10,
        9,
        6,
        1,
        2,
        1,
    ]
    neutral = [2, 1, 4, 5, 4, 7, 6, 9, 10, 9, 2, 1]
    electrons = z - charge
    ground, next_ion = (
        weights[electrons - 1],
        weights[electrons - 2] if electrons > 1 else 1,
    )
    special = {20: 2, 21: 3, 22: 4, 25: 7, 26: 10, 30: 2}
    if electrons > 18 and charge == 0:
        ground = neutral[z - 19]
        next_ion = special.get(z, next_ion)
    elif electrons > 18 and charge == 1:
        ground = special.get(z, ground)
    return ground, next_ion


def extract(tree, destination):
    data, source = tree / "data", tree / "source"
    destination.mkdir(parents=True, exist_ok=True)
    arrays, provenance, sources = {}, [], {}
    for label, z, count in SPECIES:
        atom, meta = load_atom(data, label, count)
        arrays.update(
            {
                label.replace(" ", "_") + "__" + key: np.asarray(value)
                for key, value in zip(Atom._fields, atom)
            }
        )
        meta["files"] = {
            str(Path(path).relative_to(tree)): digest
            for path, digest in meta["files"].items()
        }
        if not next(iter(meta["files"])).startswith("data/chianti"):
            for collider in ("H", "PROTON", "HE", "HE+", "H2", "H2ORTHO", "H2PARA"):
                collisions, _ = load_atom(data, label, count, collider=collider)
                if len(collisions.lower):
                    if np.any(np.asarray(collisions.collision_kind) != -1):
                        raise ValueError(
                            "Non-electron collision strengths are unsupported by C23."
                        )
                    for name in (
                        "lower",
                        "upper",
                        "collision_x",
                        "collision_y",
                        "collision_count",
                    ):
                        arrays[
                            label.replace(" ", "_") + "__" + collider + "__" + name
                        ] = np.asarray(getattr(collisions, name))
        provenance.append(meta)
        sources.update(meta["files"])
    np.savez_compressed(destination / "metal_atoms.npz", **arrays)
    arrays = {}
    fits = read_fits(data)
    selected = {2} | {z for _, z, _ in SPECIES}
    for name in ("rr", "dr_c", "dr_e", "old_rr", "new_rr", "iron_rr"):
        for key, value in fits[name].items():
            if name in ("rr", "dr_c", "dr_e"):
                keep = key[0] in selected and key[0] - key[1] in (1, 2, 3)
            elif name.endswith("_rr") and name != "iron_rr":
                keep = key[1] + 1 in selected and key[1] + 1 - key[0] in (1, 2, 3)
            else:
                keep = key in (23, 24, 25)
            if keep:
                index = key if isinstance(key, tuple) else (key,)
                arrays[name + "__" + "__".join(map(str, index))] = np.asarray(value)
    arrays.update({name: fits[name] for name in ("ct_ion", "ct_rec")})
    for z in selected:
        for q in range(min(z, 3)):
            fit = load_photoionization(data, z, q)
            arrays.update(
                {
                    f"photo__{z}__{q}__{name}": np.asarray(value)
                    for name, value in zip(PhotoFit._fields, fit)
                }
            )
            arrays[f"ground_weight__{z}__{q}"] = _ground_weights(z, q)
    coll = {}
    for row in _data(data / "coll_ion.dat"):
        if len(row) == 7 and int(row[0]) >= 0:
            electrons, z = int(row[0]) + 1, int(row[1]) + 1
            if z in selected and z - electrons in range(3):
                coll[z, z - electrons] = list(map(float, row[2:]))
    text = (source / "atmdat_adfa.cpp").read_text()
    block = text.split("static const double DereRatio[30][30]", 1)[1].split("};", 1)[0]
    ratios = [
        list(map(float, row.split(","))) for row in re.findall(r"\{([^{}]+)\}", block)
    ]
    for (z, q), values in coll.items():
        arrays[f"coll_ion__{z}__{q}"] = np.r_[values, ratios[z - 1][q]]
    arrays.update(
        {"cota__" + name: value for name, value in _cota_tables(source).items()}
    )
    np.savez_compressed(destination / "metal_rates.npz", **arrays)
    source_files = [data / name for name in fits["sha256"]]
    source_files += [
        data / name
        for name in ("phfit.dat", "ionization_potentials.dat", "coll_ion.dat")
    ]
    source_files += [
        source / name
        for name in (
            "atmdat_3body.cpp", "atmdat_adfa.cpp", "atmdat_outer_shell.cpp",
            "atmdat_chianti.cpp", "transition.h",
        )
    ]
    sources.update(
        {
            str(path.relative_to(tree)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        }
    )
    metadata = {
        "reference_version": "Cloudy C23.01",
        "atoms": provenance,
        "runtime_default_levels": {"Fe II": 80},
        "ion_stages": "q=0,1,2; RR/DR/CT/Cota coefficients also q=3 for boundary diagnostics",
        "sources_sha256": sources,
        "rounding": (
            "ADfA realnum tables and Cota b0/b1 retain original float32 rounding. "
            "Radiative EnergyWN uses the final C23 float32 assignment; CHIANTI "
            "first rounds its wgfa wavelength to float32. Thermal level energies "
            "are kept separately from these radiative centers."
        ),
        "references": [
            "Cloudy C23.01; Chatzikos et al. 2023, RMxAA 59, 327",
            "STOUT: Lykins et al. 2015, ApJ 807, 118",
            "CHIANTI 10.0.1: Del Zanna et al. 2021, ApJ 909, 38",
            "Photoionization: Verner & Yakovlev 1995, A&AS 109, 125; Verner et al. 1996, ApJ 465, 487",
            "Radiative recombination: Badnell 2006, ApJS 167, 334; Verner & Ferland 1996, ApJS 103, 467",
            "Collisional ionization: Voronov 1997, ADNDT 65, 1; Dere 2007, A&A 466, 771",
        ],
        "distribution": "Altered numerical selection from the freely distributed C23.01 STOUT/CHIANTI data; retain ../_licenses/Cloudy.txt and CHIANTI_acknowledgment.txt.",
        "source_citations": {},
    }
    for item in provenance:
        citations = {}
        for relative in item["files"]:
            lines = (tree / relative).read_text().splitlines()
            # Retain original reference comments, rather than inventing citations.
            citations[relative] = [
                line for line in lines if line.lstrip().startswith(("#", "%"))
            ]
        metadata["source_citations"][item["species"]] = citations
    (destination / "metal_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    license_path = destination.parent / "_licenses/Cloudy.txt"
    license_path.parent.mkdir(exist_ok=True)
    if (
        license_path.exists()
        and license_path.read_text() != (tree / "license.txt").read_text()
    ):
        raise ValueError(
            "Existing Cloudy license differs from the source distribution."
        )
    if not license_path.exists():
        license_path.write_text((tree / "license.txt").read_text())
    (destination / "CHIANTI_acknowledgment.txt").write_text(
        "This package includes a selected numerical subset of CHIANTI atomic data distributed with Cloudy C23.01.\n"
        "Please acknowledge CHIANTI and the original atomic calculations when publishing results.\n"
        "https://www.chiantidatabase.org/ ; https://www.chiantidatabase.org/cug.html\n\n"
        + (data / "chianti/README_CHIANTI.md").read_text()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloudy_tree", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/escape/data",
    )
    arguments = parser.parse_args()
    extract(arguments.cloudy_tree, arguments.output)
