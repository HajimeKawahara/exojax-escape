#!/usr/bin/env python3
"""Export an offline Cloudy column and optionally relax its thermal/wind state.

This is an independent reference experiment, not escape's forward model or the
unpublished Zhang et al. configuration. Input NPZ arrays are radius_cm (outward),
temperature_k, n_h_cm3 and velocity_cm_s. Cloudy illuminates from the top inward.
The optional iteration uses sunbather's thermal-advection/expansion equation;
population advection is absent. All raw runs remain in the requested directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from cloudy_reference import write_json

KB = 1.380649e-16
MH = 1.67262192369e-24
RYD_HZ = 3.2898419602508e15


def state_metadata(nmax):
    """Describe the packed populations; l=-1 denotes a summed principal shell."""
    shell = np.arange(4, nmax+1)
    return {
        "principal_n": np.r_[1, 2, 2, 3, 3, 3, shell],
        "angular_l": np.r_[0, 0, 1, 0, 1, 2, np.full(len(shell), -1)],
        "statistical_weights": np.r_[2, 2, 6, 2, 6, 10, 2*shell**2],
    }


def sha256(path):
    """Hash large reference files without loading instrumented output into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_profiles(profiles):
    """Reject nonpositive, mismatched or reversed physical input arrays."""
    names = ("radius_cm", "temperature_k", "n_h_cm3", "velocity_cm_s")
    values = {name: np.asarray(profiles[name], dtype=float) for name in names}
    radius = values["radius_cm"]
    if (radius.ndim != 1 or radius.size < 4 or np.any(np.diff(radius) <= 0)
            or any(value.shape != radius.shape or not np.all(np.isfinite(value))
                   or np.any(value <= 0) for value in values.values())):
        raise ValueError("Profiles require matching positive finite arrays and outward radii.")
    return values


def depth_law(radius, values, name):
    """Render Cloudy's logarithmic table, starting at the illuminated top."""
    width = radius[-1] - radius[0]
    depth = np.unique(np.r_[1e-35, np.geomspace(1e-2, width, 256),
                            (radius[-1] - radius[:-1])[::-1]])
    sampled = np.interp(radius[-1] - depth, radius, values)
    rows = np.column_stack((np.log10(depth), np.log10(sampled)))
    rows = np.vstack((rows, [rows[-1, 0] + .1, rows[-1, 1]]))
    return [f"{name} table depth", *[f"{x:.12g} {y:.12g}" for x, y in rows],
            f"end of {name}"]


def write_sed(directory, source, flux_scale):
    """Retain the supplied surface spectrum without spectral decimation/tails.

    The scale is F_incident/F_surface, not a mean-intensity dilution. Cloudy's
    TABLE SED has no continuum outside the supplied table unless requested;
    no tail extrapolation is requested here. This coverage is recorded.
    """
    data = np.load(source)
    frequency = np.asarray(data["frequency_hz"])
    flux = np.asarray(data["surface_flux_nu"])
    if (frequency.ndim != 1 or flux.shape != frequency.shape
            or not np.all(np.isfinite(frequency)) or np.any(np.diff(frequency) <= 0)
            or np.any(frequency <= 0) or not np.all(np.isfinite(flux))
            or np.any(flux <= 0) or not np.isfinite(flux_scale) or flux_scale <= 0):
        raise ValueError("SED requires finite increasing frequencies and positive F_nu.")
    destination = directory / "SED"
    destination.mkdir()
    with (destination / "incident.sed").open("w") as stream:
        stream.write(f"{frequency[0]:.16g} {flux[0]:.16g} units Hz\n")
        np.savetxt(stream, np.column_stack((frequency[1:], flux[1:])), fmt="%.16g")
    index = int(np.argmin(abs(frequency - RYD_HZ)))
    normalization = np.log10(frequency[index] * flux[index] * flux_scale)
    return [f'table SED "incident.sed"',
            f"nuF(nu) {normalization:.16g} at {frequency[index]/RYD_HZ:.16g} Ryd"]


def column_input(profiles, spectrum_commands, *, composition, nmax=15,
                 double_tau=True, resolution=.1, molecules=False,
                 hydrogen_atom="native-matched", max_zone_cm=None):
    """Use explicit chemistry and the public sunbather top-illuminated geometry."""
    if composition not in ("pure-h", "h-he", "solar") or nmax < 4:
        raise ValueError("Use explicit pure-h, H+He or Cloudy-default solar composition, nmax >= 4.")
    radius = profiles["radius_cm"]
    lines = ["title Offline atmosphere reference; author configuration unknown",
             *spectrum_commands]
    if composition in ("pure-h", "h-he"):
        lines.append('init file "honly.ini"')
        if composition == "h-he":
            lines += ["element helium on", "element helium abundance -1 log"]
    else:
        lines.append('abundances "default.abn"')
    if not molecules:
        lines.append("no molecules")
    if hydrogen_atom == "native-matched":
        lines += ["database H-like levels resolved 3 element hydrogen",
                  f"database H-like levels collapsed {nmax-3} element hydrogen",
                  "database H-like topoff off"]
    elif hydrogen_atom != "cloudy-default":
        raise ValueError("Unknown hydrogen atom configuration.")
    lines += ["stop temperature off",
              "set nend 3000", "iterate to convergence", "print last iteration",
              "set WeakHeatCool 0.0001",
              f"set continuum resolution {resolution:.16g}",
              f"stop thickness {np.log10(radius[-1]-radius[0]):.16g}",
              *depth_law(radius, profiles["n_h_cm3"], "dlaw"),
              *depth_law(radius, profiles["temperature_k"], "tlaw")]
    if double_tau:
        lines.append("double optical depths")
    if max_zone_cm is not None:
        lines.append(f"set drmax {max_zone_cm:.16g} linear")
    lines += ['save overview "model.overview" last',
              'save heating "model.heating" last',
              'save cooling "model.cooling" last',
              'save pressure "model.pressure" last',
              'save radius "model.radius" last',
              'save special "model.mass" last',
              'save hydrogen conditions "model.conditions" last',
              'save hydrogen Lya "model.lya" last',
              'save species densities "model.levels" "H[:]" last',
              'save species densities "model.electrons" "e-" last',
              'save line optical depths "model.tau" last']
    return "\n".join(lines) + "\n"


def numeric_rows(path, ncolumn):
    """Read fixed leading columns, retaining Cloudy's varying process diagnostics."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = [float(value) for value in line.split()[:ncolumn]]
        except ValueError:
            continue
        if len(row) == ncolumn:
            rows.append(row)
    result = np.asarray(rows)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"Missing/nonfinite numeric output: {path}")
    return result


def parse_column(directory, radius_top, nmax, resolved_levels=3):
    """Read material properties directly; derive mu from Cloudy's gas pressure."""
    with (directory / "model.out").open() as stream:
        completed = any("Cloudy exited OK" in line for line in stream)
    if not completed:
        raise ValueError("Cloudy did not complete successfully; inspect retained model.out.")
    if "Ionization not converged" in (directory / "model.heating").read_text():
        raise ValueError("Cloudy reports unconverged ionization; this is not a usable reference.")
    conditions = numeric_rows(directory / "model.conditions", 10)
    heating = numeric_rows(directory / "model.heating", 5)
    pressure = numeric_rows(directory / "model.pressure", 6)
    mass = numeric_rows(directory / "model.mass", 7)
    # SAVE H[:] may include zero-filled inactive slots. Reject nonzero levels
    # beyond the declared atom rather than silently truncating a changed input.
    nstate = resolved_levels*(resolved_levels+1)//2 + nmax-resolved_levels
    level_path = directory / "model.levels"
    ncolumn = next(len(line.split()) for line in level_path.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#"))
    all_levels = numeric_rows(level_path, ncolumn)
    if ncolumn < nstate+1 or np.any(all_levels[:, nstate+1:] != 0):
        raise ValueError("Saved populations disagree with the declared hydrogen atom.")
    levels = all_levels[:, :nstate+1]
    # Keep 1s,2s,2p,3s,3p,3d; sum higher resolved l states into principal shells.
    packed = [levels[:, 1:7]]
    for principal in range(4, resolved_levels+1):
        offset = 1+principal*(principal-1)//2
        packed.append(levels[:, offset:offset+principal].sum(axis=1, keepdims=True))
    packed.append(levels[:, 1+resolved_levels*(resolved_levels+1)//2:])
    bound = np.column_stack(packed)
    overview = numeric_rows(directory / "model.overview", 11)
    if any(len(item) != len(conditions) for item in (heating, pressure, mass, levels, overview)):
        raise ValueError("Cloudy saved arrays have inconsistent zone counts.")
    if any(not np.allclose(item[:, 0], conditions[:, 0], rtol=2e-4)
           for item in (heating, pressure, levels, overview)):
        raise ValueError("Cloudy saved arrays do not describe the same depths.")
    rho, temp, pgas = mass[:, 5], heating[:, 1], pressure[:, 5]
    profiles = {
        "radius_cm": radius_top-conditions[:, 0], "temperature_k": temp,
        "density_g_cm3": rho, "n_h_cm3": conditions[:, 2],
        "electron_cm3": conditions[:, 3], "ionized_fraction": conditions[:, 5],
        "hminus_fraction": conditions[:, 9], "gas_pressure_dyn_cm2": pgas,
        "helium_neutral_fraction": overview[:, 8],
        "helium_singly_ionized_fraction": overview[:, 9],
        "helium_doubly_ionized_fraction": overview[:, 10],
        "mean_molecular_weight": rho * KB * temp / (MH * pgas),
        "heating_erg_cm3_s": heating[:, 2], "cooling_erg_cm3_s": heating[:, 3],
        "n2_cm3": levels[:, 2:4].sum(axis=1),
        "n3_cm3": levels[:, 4:7].sum(axis=1),
        "bound_cm3": bound,
    }
    if (directory / "model.radius").exists():
        zones = numeric_rows(directory / "model.radius", 4)
        if len(zones) != len(conditions):
            raise ValueError("Cloudy radius output does not match the thermochemical zones.")
        profiles["zone_width_cm"] = zones[:, 3]
    if np.any(rho <= 0) or np.any(pgas <= 0) or np.any(np.diff(profiles["radius_cm"]) >= 0):
        raise ValueError("Invalid physical state or truncated/repeated Cloudy iteration output.")
    return {key: value[::-1] for key, value in profiles.items()}


def thermal_balance(profile, velocity_cm_s):
    """Sunbather Eq. 3: thermal advection + expansion + radiation = zero."""
    radius, rho, temp, mu = (profile[key] for key in (
        "radius_cm", "density_g_cm3", "temperature_k", "mean_molecular_weight"))
    rho = profile.get("wind_density_g_cm3", rho)
    specific_pressure = KB * temp / (MH * mu)
    advection = -rho * velocity_cm_s * np.gradient(1.5 * specific_pressure, radius)
    expansion = specific_pressure * velocity_cm_s * np.gradient(rho, radius)
    heating = profile["heating_erg_cm3_s"] + np.maximum(advection, 0) + np.maximum(expansion, 0)
    cooling = profile["cooling_erg_cm3_s"] + np.maximum(-advection, 0) + np.maximum(-expansion, 0)
    return advection, expansion, heating / cooling


def dense_lower_patch(profiles, result, start, base_temperature_k):
    """Join prescribed lower gas to Cloudy; this is not an author-supplied law.

    Temperature is linear in radius from the photosphere to the first valid
    Cloudy node. Hydrogen ionization bridges dense Saha equilibrium to Cloudy;
    the boundary non-H electron abundance is held fixed. Bound excitation is
    LTE below the join. Populations are NOT blended to hide a failed LTE join.
    """
    if not start:
        return
    radius = profiles["radius_cm"]
    coordinate = (radius[:start+1]-radius[0])/(radius[start]-radius[0])
    temperature = base_temperature_k + coordinate*(result["temperature_k"][start]-base_temperature_k)
    nh = profiles["n_h_cm3"][:start+1]
    mass_per_h = result["density_g_cm3"][start]/result["n_h_cm3"][start]
    electron_boundary = result["electron_cm3"][start]/result["n_h_cm3"][start]
    background = max(0., electron_boundary-result["ionized_fraction"][start])
    nuclei = mass_per_h/(MH*result["mean_molecular_weight"][start])-electron_boundary
    # Saha's ground-state H partition: 2*g_proton/g_1s = 1.
    phase_space = (2*np.pi*9.1093837e-28*KB*temperature/(6.62607015e-27)**2)**1.5
    saha = phase_space*np.exp(-13.5984346*1.602176634e-12/(KB*temperature))/nh
    b = background+saha
    ionized_lte = 2*saha/(b+np.sqrt(b*b+4*saha))
    ionized = (1-coordinate**4)*ionized_lte + coordinate**4*result["ionized_fraction"][start]
    electron = nh*(background+ionized)
    mu = mass_per_h/(MH*(nuclei+background+ionized))
    nbound = result["bound_cm3"].shape[1]
    principal = np.r_[1,2,2,3,3,3,np.arange(4,nbound-2)]
    degeneracy = np.r_[2,2,6,2,6,10,2*np.arange(4,nbound-2)**2]
    weights = degeneracy[None,:]*np.exp(-13.5984346*1.602176634e-12*(1-1/principal[None,:]**2)/(KB*temperature[:,None]))
    bound = nh[:,None]*(1-ionized[:,None])*weights/weights.sum(axis=1,keepdims=True)
    patch = {
        "temperature_k": temperature, "density_g_cm3": nh*mass_per_h, "n_h_cm3": nh,
        "mean_molecular_weight": mu, "electron_cm3": electron, "ionized_fraction": ionized,
        "gas_pressure_dyn_cm2": nh*(nuclei+background+ionized)*KB*temperature,
        "bound_cm3": bound, "n2_cm3": bound[:,1:3].sum(axis=1), "n3_cm3": bound[:,3:6].sum(axis=1),
        "hminus_fraction": (1-ionized)*electron/(4*phase_space)*np.exp(.754195*1.602176634e-12/(KB*temperature)),
    }
    for key,value in patch.items():
        result[key][:start] = value[:start]
    profiles["temperature_k"][:start] = temperature[:start]
    for key in ("heating_erg_cm3_s", "cooling_erg_cm3_s"):
        result[key][:start] = np.nan  # prescribed region; no claimed thermal balance


def updated_wind(args, profiles, result):
    """Reuse the native transonic solver, with Cloudy's composition feedback."""
    import jax
    from escape.atmosphere_wind import wind_from_profiles
    jax.config.update("jax_enable_x64", True)
    mu = np.interp(profiles["radius_cm"], result["radius_cm"], result["mean_molecular_weight"])
    wind = wind_from_profiles(
        profiles["radius_cm"], profiles["temperature_k"], mu,
        args.mass_loss_g_s, args.planet_mass_mj,
        base_radius_cm=profiles["radius_cm"][0],
        outer_radius_cm=args.outer_wind_radius_cm,
        tidal_coefficient_s2=args.tidal_coefficient_s2,
    )
    mass_per_h = np.median(result["density_g_cm3"] / result["n_h_cm3"])
    updated = dict(profiles, velocity_cm_s=np.asarray(wind.velocity_kms) * 1e5,
                   n_h_cm3=np.asarray(wind.density_g_cm3) / mass_per_h)
    return validate_profiles(updated)


def relax_wind(current, proposal, weight):
    """Damp velocity logarithmically, preserving the proposed mass flux exactly."""
    if not 0 < weight <= 1:
        raise ValueError("Wind relaxation must be in (0, 1].")
    velocity = np.exp((1-weight)*np.log(current["velocity_cm_s"])
                      + weight*np.log(proposal["velocity_cm_s"]))
    density = proposal["n_h_cm3"]*proposal["velocity_cm_s"]/velocity
    return dict(proposal, velocity_cm_s=velocity, n_h_cm3=density)


def sunbather_temperature(profiles, result, directory, iteration, *, freeze_base=True,
                         relaxation=1.):
    """Call the actual optional reference algorithm; do not vendor its code."""
    import pandas as pd
    from sunbather import solveT
    radius, temp = profiles["radius_cm"], profiles["temperature_k"]
    velocity = profiles["velocity_cm_s"]
    rho = result.get("wind_density_g_cm3", result["density_g_cm3"])
    mu = result["mean_molecular_weight"]
    heat, cool = result["heating_erg_cm3_s"], result["cooling_erg_cm3_s"]
    adv = solveT.calc_advection(radius, rho, velocity, temp, mu)
    expansion = -solveT.calc_expansion(radius, rho, velocity, temp, mu)
    adv = solveT.tools.smooth_gaus_savgol(adv, fraction=.01)
    expansion = solveT.tools.smooth_gaus_savgol(expansion, fraction=.01)
    adv_heat, adv_cool = np.maximum(adv, 0), np.maximum(-adv, 0)
    ratio = solveT.calc_HCratio(heat, cool, expansion, adv_heat, adv_cool)
    directory.mkdir(exist_ok=True)
    relaxed = solveT.relaxTstruc(radius, str(directory)+"/", iteration, temp, ratio)
    index = solveT.calc_cloc(heat, cool, expansion, adv_heat, adv_cool, ratio)
    if index < len(radius):
        relaxed = solveT.constructTstruc(radius, relaxed, int(index), velocity, rho, mu, heat, cool)
    if freeze_base:
        relaxed[0] = temp[0]
    applied = temp + relaxation*(relaxed-temp)
    history = pd.read_csv(directory / "iterations.txt", sep=" ")
    history[f"Te{iteration}"] = applied
    history.to_csv(directory / "iterations.txt", sep=" ", index=False)
    return relaxed, applied


def run(args):
    """Run fixed-profile comparison or a reported, bounded relaxation experiment."""
    original = validate_profiles(np.load(args.profiles))
    profiles = {key: value.copy() for key, value in original.items()}
    if args.dense_lower_patch:
        profiles["temperature_k"][0] = args.base_temperature_k
    working_mu = np.full_like(profiles["radius_cm"],
                              {"pure-h": 1., "h-he": 1.4/1.1, "solar": 1.415276/1.10106}[args.composition])
    mass_per_h = 1.6735575e-24*{"pure-h":1., "h-he":1.4, "solar":1.415276}[args.composition]
    args.output.mkdir(parents=True, exist_ok=False)
    nmax, resolved_levels = ((25, 10) if args.hydrogen_atom == "cloudy-default"
                            else (args.nmax, 3))
    receipt = {
        "scope": "Offline specified-composition reference, not the authors' input or a native forward model.",
        "driver_sha256": sha256(Path(__file__)),
        "source_profiles_sha256": sha256(args.profiles), "cloudy_sha256": sha256(args.cloudy),
        "composition": args.composition, "molecules": args.molecules,
        "boundary": "Top-incident beam; double optical depths" if args.double_tau else "Top-incident beam; open back",
        "nmax": nmax, "resolved_levels": resolved_levels,
        "hydrogen_atom": args.hydrogen_atom,
        "hydrogen_topoff": args.hydrogen_atom == "cloudy-default",
        "continuum_resolution": args.resolution,
        "max_zone_cm": args.max_zone_cm,
        "spectrum": {"blackbody_k": args.blackbody_k, "dilution": args.dilution},
        "thermal_equation": "sunbather Eq. 3; no population advection; gamma=5/3",
        "thermal_density": "Prescribed mass-conserving wind; Cloudy density and zoning offsets saved separately",
        "iteration_scheme": "Converge thermal state at fixed density/velocity using public sunbather, then update wind",
        "thermal_convergence_factor": args.convergence_factor,
        "allow_smoothed_thermal_convergence": args.allow_smoothed_thermal,
        "temperature_update_relaxation": args.temperature_relaxation,
        "wind_relative_convergence": args.wind_tolerance,
        "wind_log_relaxation": args.wind_relaxation,
        "wind_update_requested": args.mass_loss_g_s is not None,
        "dense_lower_patch": args.dense_lower_patch,
        "lower_patch_law": "Linear T(r); Saha-to-Cloudy ionization bridge; LTE excitation; boundary non-H electron fraction" if args.dense_lower_patch else None,
        "base_temperature_k": args.base_temperature_k if args.dense_lower_patch else None,
        "thermal_boundary": "Fixed innermost temperature; energy balance checked at remaining nodes",
        "mass_loss_g_s": args.mass_loss_g_s, "planet_mass_mj": args.planet_mass_mj,
        "outer_wind_radius_cm": args.outer_wind_radius_cm,
        "tidal_coefficient_s2": args.tidal_coefficient_s2,
        "thermal_converged": False,
        "converged": False, "iterations": [],
    }
    if args.sed is not None:
        receipt["spectrum"] = {"npz_sha256": sha256(args.sed), "flux_scale": args.flux_scale,
                               "tails": "zero outside supplied frequency coverage"}
    if args.continuum_mesh is not None:
        receipt["continuum_mesh_sha256"] = sha256(args.continuum_mesh)
    if args.iterations:
        from sunbather import solveT
        receipt["sunbather_solveT_sha256"] = sha256(Path(solveT.__file__))
    environment = os.environ.copy()
    if args.cloudy_data is not None:
        environment["CLOUDY_DATA_PATH"] = f".:{args.cloudy_data.resolve()}"
        receipt["cloudy_default_abundances_sha256"] = sha256(args.cloudy_data / "abundances/default.abn")
    receipt["cloudy_data_search_path"] = environment.get("CLOUDY_DATA_PATH", "+ (compiled default)")
    write_json(args.output / "receipt.json", receipt)
    thermal_iteration, outer_iteration = 2, 0
    previous_start = None
    forced_start = 0
    for iteration in range(args.iterations + 1):
        pressure = profiles["n_h_cm3"]*mass_per_h*KB*profiles["temperature_k"]/(working_mu*MH)*1e-6
        invalid = (profiles["n_h_cm3"] > 1e15) | (pressure > 1e-3)
        start = int(np.flatnonzero(invalid)[-1])+1 if args.dense_lower_patch and np.any(invalid) else 0
        start = max(start, forced_start)
        if len(pressure)-start < 4:
            raise ValueError("Dense patch leaves fewer than four Cloudy nodes; expand/refine the atmospheric domain.")
        if start != previous_start:
            thermal_iteration = 2
        previous_start = start
        column = {key:value[start:] for key,value in profiles.items()}
        directory = args.output / f"iteration_{iteration:03d}"
        directory.mkdir()
        if args.continuum_mesh is not None:
            shutil.copyfile(args.continuum_mesh, directory / "continuum_mesh.ini")
        np.savez(directory / "input_profiles.npz", **profiles)
        spectrum = ([f"blackbody {args.blackbody_k:.16g} K dilution {args.dilution:.16g}"]
                    if args.sed is None else write_sed(directory, args.sed, args.flux_scale))
        (directory / "model.in").write_text(column_input(
            column, spectrum, composition=args.composition, nmax=args.nmax,
            double_tau=args.double_tau, resolution=args.resolution, molecules=args.molecules,
            hydrogen_atom=args.hydrogen_atom, max_zone_cm=args.max_zone_cm))
        with (directory / "console.txt").open("w") as console:
            process = subprocess.run([str(args.cloudy.resolve()), "-r", "model"], cwd=directory,
                                     stdout=console, stderr=subprocess.STDOUT, timeout=args.timeout,
                                     env=environment)
        if process.returncode:
            raise RuntimeError(f"Cloudy failed ({process.returncode}); inspect {directory}.")
        resolved = parse_column(directory, profiles["radius_cm"][-1], nmax, resolved_levels)
        np.savez(directory / "cloudy_profiles.npz", **resolved, **state_metadata(nmax))
        # Derivatives of rounded Cloudy output on sub-centimeter surface zones
        # are meaningless. Evaluate the flow balance on the declared input grid.
        radius = profiles["radius_cm"]
        result = {}
        for key, value in resolved.items():
            result[key] = (np.interp(radius, resolved["radius_cm"], value) if value.ndim == 1
                           else np.column_stack([np.interp(radius, resolved["radius_cm"], col)
                                                 for col in value.T]))
        result["radius_cm"] = radius
        dense_lower_patch(profiles, result, start, args.base_temperature_k)
        working_mu = result["mean_molecular_weight"].copy()
        mass_per_h = float(np.median(result["density_g_cm3"][start:]/result["n_h_cm3"][start:]))
        # Like public sunbather.simtogrid, flow derivatives use the prescribed
        # wind, not Cloudy's back-evaluated DLAW density reported at zone centers.
        result["wind_density_g_cm3"] = profiles["n_h_cm3"]*mass_per_h
        active = {key:value[start:] for key,value in result.items()}
        active_profiles = {key:value[start:] for key,value in profiles.items()}
        velocity = profiles["velocity_cm_s"]
        advection_active, expansion_active, ratio_active = thermal_balance(active, velocity[start:])
        advection, expansion, ratio = (np.r_[np.full(start,np.nan),x] for x in (advection_active,expansion_active,ratio_active))
        result.update(velocity_cm_s=velocity, advection_erg_cm3_s=advection,
                      expansion_erg_cm3_s=expansion, total_heating_cooling_ratio=ratio,
                      cloudy_mask=np.arange(len(radius)) >= start)
        np.savez(directory / "profiles.npz", **result, **state_metadata(nmax))
        checked = ratio_active if start else ratio_active[1:]
        imbalance = float(np.max(np.maximum(checked, 1/checked)))
        step = {"iteration": iteration, "zones": len(resolved["radius_cm"]),
                "balance_grid_points": len(ratio), "max_heating_cooling_factor": imbalance,
                "max_n_h_cm3": float(np.max(result["n_h_cm3"][start:])),
                "max_pressure_bar": float(np.max(result["gas_pressure_dyn_cm2"][start:]))*1e-6,
                "base_pressure_bar": float(result["gas_pressure_dyn_cm2"][0])*1e-6,
                "base_pressure_below_10_bar": bool(result["gas_pressure_dyn_cm2"][0]<1e7),
                "patch_nodes": start, "join_radius_rp": float(radius[start]/radius[0]),
                "wind_iteration": outer_iteration,
                "max_cloudy_density_projection_relative_offset": float(np.max(abs(
                    result["density_g_cm3"][start:]/result["wind_density_g_cm3"][start:]-1))),
                "input_sha256": sha256(directory / "model.in"),
                "output_sha256": sha256(directory / "model.out")}
        receipt["thermal_converged"] = imbalance < args.convergence_factor
        proposed_temperature = None
        applied_temperature = None
        if args.iterations and not receipt["thermal_converged"]:
            from sunbather import solveT
            proposed_temperature, applied_temperature = sunbather_temperature(
                active_profiles, active, args.output / f"thermal_{outer_iteration:03d}_{start:03d}",
                thermal_iteration, freeze_base=not start,
                relaxation=args.temperature_relaxation)
            stabilized = thermal_iteration > 3 and bool(solveT.check_converged(
                args.convergence_factor, np.maximum(ratio_active, 1/ratio_active),
                proposed_temperature, profiles["temperature_k"][start:], linthresh=50.))
            step["sunbather_temperature_stabilized"] = stabilized
            step["max_proposed_temperature_change_k"] = float(np.max(abs(
                proposed_temperature-profiles["temperature_k"][start:])))
            step["max_applied_temperature_change_k"] = float(np.max(abs(
                applied_temperature-profiles["temperature_k"][start:])))
        else:
            stabilized = False
        thermal_ready = receipt["thermal_converged"] or (stabilized and args.allow_smoothed_thermal)
        receipt["thermal_convergence_kind"] = ("energy_balance" if receipt["thermal_converged"]
                                                else "sunbather_smoothing_limit" if stabilized
                                                else "not_converged")
        domain_valid = step["max_n_h_cm3"] <= 1e15 and step["max_pressure_bar"] <= 1e-3
        step["within_paper_cloudy_domain"] = domain_valid
        if start:
            lower, upper = result["n2_cm3"][start], result["n3_cm3"][start]
            lte_ratio = 2.25*np.exp(-1.889*1.602176634e-12/(KB*result["temperature_k"][start]))
            step["join_n3_n2_departure_from_lte"] = float(upper/lower/lte_ratio)
            n2_lte = result["n_h_cm3"][start]*(1-result["ionized_fraction"][start])*4*np.exp(
                -10.198826*1.602176634e-12/(KB*result["temperature_k"][start]))
            step["join_n2_departure_from_lte"] = float(lower/n2_lte)
        if args.mass_loss_g_s is not None and thermal_ready:
            proposal = updated_wind(args, profiles, result)
            wind_error = max(float(np.max(abs(proposal[key]/profiles[key]-1)))
                             for key in ("velocity_cm_s", "n_h_cm3"))
            step["wind_fixed_point_relative_error"] = wind_error
            receipt["converged"] = domain_valid and thermal_ready and wind_error < args.wind_tolerance and step["base_pressure_below_10_bar"]
        elif args.mass_loss_g_s is None:
            receipt["converged"] = domain_valid and thermal_ready
        else:
            receipt["converged"] = False
        receipt["iterations"].append(step)
        write_json(args.output / "receipt.json", receipt)
        print(f"Iteration {iteration}: {step['zones']} zones, max thermal imbalance {imbalance:.6g}", flush=True)
        if not domain_valid:
            if args.dense_lower_patch:
                forced_start = start+1
                print("Move the lower join outward to satisfy Cloudy's actual pressure/density domain.", flush=True)
                continue
            print("Dense lower region requires an explicit lower-atmosphere patch; stopping relaxation.", flush=True)
        if receipt["converged"] or iteration == args.iterations or not domain_valid:
            break
        if thermal_ready and args.mass_loss_g_s is not None:
            # Convergence above uses the undamped fixed-point proposal. A small
            # numerical update must never be mistaken for a small residual.
            profiles = relax_wind(profiles, proposal, args.wind_relaxation)
            thermal_iteration = 2
            outer_iteration += 1
            forced_start = 0
        else:
            profiles["temperature_k"][start:] = applied_temperature
            if start:
                coordinate = (radius[:start]-radius[0])/(radius[start]-radius[0])
                profiles["temperature_k"][:start] = args.base_temperature_k + coordinate*(applied_temperature[0]-args.base_temperature_k)
            thermal_iteration += 1
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--cloudy", type=Path, required=True)
    parser.add_argument("--cloudy-data", type=Path, help="Explicit data search directory, recorded with default.abn checksum.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composition", choices=("pure-h", "h-he", "solar"), required=True)
    parser.add_argument("--hydrogen-atom", choices=("cloudy-default", "native-matched"),
                        default="cloudy-default",
                        help="C23.01 default: resolved through10, collapsed through25, topoff on. Native matching is a separate control.")
    parser.add_argument("--nmax", type=int, default=15,
                        help="Principal-shell limit for --hydrogen-atom=native-matched only.")
    parser.add_argument("--resolution", type=float, default=.1)
    parser.add_argument("--continuum-mesh", type=Path,
                        help="Explicit Cloudy continuum_mesh.ini override; hash recorded, multiplied by --resolution.")
    parser.add_argument("--max-zone-cm", type=float,
                        help="Optional Cloudy maximum zone width for a spatial convergence control.")
    parser.add_argument("--molecules", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable full Cloudy chemistry, including H-minus; off isolates the atomic comparison.")
    parser.add_argument("--double-tau", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sed", type=Path, help="Surface F_nu NPZ; overrides the diagnostic blackbody.")
    parser.add_argument("--flux-scale", type=float, default=.1)
    parser.add_argument("--blackbody-k", type=float, default=10000.)
    parser.add_argument("--dilution", type=float, default=.025)
    parser.add_argument("--iterations", type=int, default=0, help="0 exports a fixed-profile reference only.")
    parser.add_argument("--convergence-factor", type=float, default=1.1)
    parser.add_argument("--allow-smoothed-thermal", action="store_true",
                        help="Also accept public sunbather's temperature-change criterion; strict energy balance is the default.")
    parser.add_argument("--temperature-relaxation", type=float, default=1.,
                        help="Damp the public temperature proposal; convergence uses the physical energy residual.")
    parser.add_argument("--wind-tolerance", type=float, default=.01)
    parser.add_argument("--wind-relaxation", type=float, default=1.,
                        help="Logarithmic velocity update weight; the undamped residual determines convergence.")
    parser.add_argument("--dense-lower-patch", action="store_true",
                        help="Exclude P>1e-3 bar/nH>1e15 from Cloudy and prescribe the documented lower join.")
    parser.add_argument("--base-temperature-k", type=float, default=4500.)
    parser.add_argument("--mass-loss-g-s", type=float, help="Also update the transonic wind after each thermal step.")
    parser.add_argument("--planet-mass-mj", type=float)
    parser.add_argument("--outer-wind-radius-cm", type=float)
    parser.add_argument("--tidal-coefficient-s2", type=float, default=0.)
    parser.add_argument("--timeout", type=float, default=600.)
    args = parser.parse_args()
    if not 0 < args.wind_relaxation <= 1:
        parser.error("--wind-relaxation must be in (0, 1].")
    if not 0 < args.temperature_relaxation <= 1:
        parser.error("--temperature-relaxation must be in (0, 1].")
    if args.max_zone_cm is not None and args.max_zone_cm <= 0:
        parser.error("--max-zone-cm must be positive.")
    if args.mass_loss_g_s is not None and (args.planet_mass_mj is None or args.outer_wind_radius_cm is None):
        parser.error("Wind updates require --planet-mass-mj and --outer-wind-radius-cm.")
    run(args)


if __name__ == "__main__":
    main()
