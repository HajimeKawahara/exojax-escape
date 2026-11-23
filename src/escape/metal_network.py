"""Metal populations, shared radiation and kinetic energy for a static column.

The reduced network retains q=0..2 for ten elements and twelve excitation
atoms. It includes ground PI, electron collisional ionization, total RR/DR/
three-body recombination, H/He charge exchange, selected excited N/O/Mg/Ca
photoionization and line excitation. Overlap-group rates and their continuum
corrections may replace isolated line operators. Atomic truncation, omitted
Auger/secondary electrons and other coolants need assessment for atmospheric use.

The continuum and population equations share the same densities and radiation.
No reference-solver rates, fitted heating curves, or frozen ion fractions are
needed at runtime. All temperature-dependent quantities are evaluated per call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from escape.line_escape_probability import (
    line_operators, cell_averaged_line_operators, line_cell_quadrature,
    _optical_depth_cell_segments, LineOperators, CORE, WING, PRD,
)
from escape.radiation import _cell_weights
from escape.metal_atom import (
    load_metal_atom,
    load_metal_colliders,
    tabulated_collision_coefficients,
    electron_collision_coefficients,
    populations,
    kinetic_line_heating,
    PLANCK,
    C,
    KB,
)
from escape.metal_rates import (
    load_metal_rates,
    photoionization_cross_sections,
    radiative_recombination,
    raw_dielectronic_recombination,
    dielectronic_suppression,
    three_body_recombination,
    charge_exchange,
    collisional_ionization,
    ground_recombination_continuum,
    oxygen_ground_valence_cross_section,
    excited_photoionization_cross_section,
    EV_ERG,
)

# Separate elemental abundance from excitation atoms: C I/C II and Fe II/III
# must share one conserved elemental total, not independent abundances.
ELEMENTS = (
    (6, 12.011),
    (7, 14.007),
    (8, 15.999),
    (12, 24.305),
    (13, 26.982),
    (14, 28.085),
    (16, 32.06),
    (20, 40.078),
    (25, 54.938),
    (26, 55.845),
)
ATOMS = (
    ("C I", 6, 0),
    ("C II", 6, 1),
    ("N I", 7, 0),
    ("O I", 8, 0),
    ("Mg II", 12, 1),
    ("Al II", 13, 1),
    ("Si II", 14, 1),
    ("S II", 16, 1),
    ("Ca II", 20, 1),
    ("Mn II", 25, 1),
    ("Fe II", 26, 1),
    ("Fe III", 26, 2),
)


def metal_frequency_edges():
    """All retained PI thresholds [Hz] for a shared continuum quadrature."""
    data = load_metal_rates()
    edges = []
    for z, _ in ELEMENTS:
        for q in range(3):
            fit = data["photo"][z, q]
            edges.extend(
                np.asarray(fit.threshold_ev)[np.asarray(fit.valid)] * EV_ERG / PLANCK
            )
            if q < 2:
                valence = (
                    float(fit.threshold_ev[int(fit.outer_shell) - 1]) * EV_ERG / PLANCK
                )
                edges.extend((0.25 * valence, 0.75 * valence))
    edges.extend(np.array([0.779, 0.856, 0.893, 1.0, 2.0]) * C * 109737.3156816)
    edges.extend(np.array([13.62, 16.94, 18.79, 28.48, 50.0, 110.5]) * EV_ERG / PLANCK)
    return np.unique(edges)


def _depth_averaged_metal_lines(kind, increment, frequency, incident,
                               continuum_front, continuum_depth, diffuse_field,
                               order, shielding_model, core_escape_model, *,
                               return_normalization_error=False,
                               line_mean_intensity_override=None, **arguments):
    """Integrate the unchanged line kernel and actual point continuum together.

    PRD joins at a*tau=1 and core-fit joins are followed on both faces.
    A checkpointed weighted scan avoids storing node x cell x Fe-line arrays.
    """
    grid = diffuse_field["frequency_hz"]
    right = jnp.clip(jnp.searchsorted(grid, frequency), 1, len(grid)-1)
    left = right-1
    indices = jnp.stack((left, right), axis=-1)
    fraction = jnp.clip((frequency-grid[left])/(grid[right]-grid[left]), 0., 1.)
    interpolation = jnp.stack((1-fraction, fraction), axis=-1)
    incoming, depth, emitted = (diffuse_field[name][..., indices] for name in
                                ("incoming", "optical_depth", "emitted"))
    angular = diffuse_field["angular_weight"][:, None, None, None]
    front = arguments["tau_in"]
    shape = front.shape
    total = jnp.broadcast_to(arguments["tau_total"], shape)
    damping = jnp.broadcast_to(arguments["damping"], shape)
    core_joins = ((.01, 11.) if core_escape_model == "cloudy" else (10., 12.))
    joins = jnp.stack([jnp.where(kind == PRD, 1/damping, value/jnp.sqrt(jnp.pi))
                       for value in core_joins]+[jnp.zeros_like(front)])
    continuum_scale = jnp.sqrt(1+continuum_depth**2+jnp.sum(depth**2, axis=(0, 3)))
    starts, spans, scales = _optical_depth_cell_segments(
        front.ravel(), total.ravel(), increment.ravel(), continuum_scale.ravel(),
        jnp.zeros(front.size), joins.reshape(3, -1))
    starts, spans, scales = (array.reshape((-1,)+shape) for array in
                             (starts, spans, scales))
    lengths = jnp.log1p(scales)
    x, w = line_cell_quadrature(order)

    def step(index, sums):
        segment, remainder = index//(2*order), index % (2*order)
        node, side = remainder % order, remainder//order
        local = .5*jnp.expm1(x[node]*lengths[segment])/scales[segment]
        position = starts[segment]+spans[segment]*jnp.where(side == 0, local, 1-local)
        weight = (w[node]*spans[segment]*.5*lengths[segment]
                  *jnp.exp(x[node]*lengths[segment])/scales[segment])
        location = position[None, ..., None]
        diffuse = jnp.sum(angular*interpolation[None, None, ...]*(
            incoming*jnp.exp(-location*depth)
            +location*emitted*_cell_weights(location*depth)[0]), axis=(0, 3))
        point_intensity = incident*jnp.exp(-continuum_front-position*continuum_depth)+diffuse
        if line_mean_intensity_override is not None:
            mask, mean = line_mean_intensity_override
            point_intensity = jnp.where(mask[None, :], mean, point_intensity)
        values = dict(arguments, tau_in=front+position*increment,
                      tau_pump=arguments["tau_pump"]+position*increment,
                      incident_occupation=point_intensity/(2*PLANCK*frequency**3/C**2))
        result = line_operators(kind, average_zone=False, shielding_model=shielding_model,
                                core_escape_model=core_escape_model, **values)
        return (sums[0]+weight, tuple(old+weight*value for old, value in zip(sums[1], result)))

    zero = jnp.zeros_like(front)
    mass, values = jax.lax.fori_loop(
        0, len(starts)*2*order, jax.checkpoint(step, prevent_cse=False),
        (zero, tuple(zero for _ in LineOperators._fields)))
    valid = ((jnp.minimum(front, front+increment) > -30)
             & (jnp.minimum(arguments["tau_pump"], arguments["tau_pump"]+increment) > -30)
             & ((kind != CORE) | (total > -30)) & jnp.isfinite(mass) & (mass > 0))
    result = LineOperators(*(jnp.where(valid, value/mass, jnp.nan) for value in values))
    return (result, mass-1) if return_normalization_error else result


class MetalNetwork:
    """Static geometry/data; populations and temperature remain differentiable.

    Cells are ordered from the lower boundary to the illuminated upper face.
    State columns are two charge ratios per element, then excited/ground level
    ratios for each atom. Optional ``atoms`` selects a declared subset for
    controlled tests; every selected element still has its own charge ladder.
    """

    def __init__(
        self,
        frequency_hz,
        weights_dnu,
        widths_cm,
        *,
        atoms=ATOMS,
        line_optical_depth_scale=1.0,
        iron_levels=80,
        sulfur_levels=72,
        line_sampling="front",
        line_quadrature_order=4,
        line_quadrature="gauss",
        shielding_model="cloudy",
        core_escape_model="cloudy",
    ):
        self.frequency_hz = jnp.asarray(frequency_hz)
        self.weight_hz = jnp.asarray(weights_dnu)
        self.widths_cm = jnp.asarray(widths_cm)
        if (
            self.frequency_hz.ndim != 1
            or self.weight_hz.shape != self.frequency_hz.shape
            or np.any(np.diff(frequency_hz) <= 0)
            or np.any(np.asarray(weights_dnu) <= 0)
            or self.widths_cm.ndim != 1
            or np.any(np.asarray(widths_cm) <= 0)
        ):
            raise ValueError(
                "Provide increasing frequencies, positive quadrature weights and cell widths."
            )
        if not np.isfinite(line_optical_depth_scale) or line_optical_depth_scale <= 0:
            raise ValueError("Line optical-depth scale must be positive and finite.")
        self.line_optical_depth_scale = float(line_optical_depth_scale)
        if line_sampling not in ("front", "average"):
            raise ValueError("Line sampling must be 'front' or 'average'.")
        self.line_sampling = line_sampling
        if line_quadrature not in ("gauss", "optical_depth"):
            raise ValueError("Choose gauss or optical_depth line quadrature.")
        if line_quadrature == "optical_depth" and line_sampling != "average":
            raise ValueError("Optical-depth quadrature requires cell-averaged lines.")
        self.line_quadrature = line_quadrature
        if shielding_model not in ("cloudy", "continuous_core"):
            raise ValueError("Choose 'cloudy' or 'continuous_core' shielding.")
        self.shielding_model = shielding_model
        if core_escape_model not in ("cloudy", "continuous"):
            raise ValueError("Choose cloudy or continuous Doppler core escape.")
        self.core_escape_model = core_escape_model
        self.line_quadrature_order = line_quadrature_order
        self._line_nodes, _ = line_cell_quadrature(line_quadrature_order)
        self.data = load_metal_rates()
        self.atom_specs = tuple(atoms)
        if not self.atom_specs or any(spec not in ATOMS for spec in self.atom_specs):
            raise ValueError("Select one or more distinct packaged excitation atoms.")
        if len(set(self.atom_specs)) != len(self.atom_specs):
            raise ValueError("Excitation atoms must be unique.")
        self.elements = tuple(
            spec for spec in ELEMENTS if any(atom[1] == spec[0] for atom in atoms)
        )
        self.atomic_number = tuple(z for z, _ in self.elements)
        self.iron_levels = iron_levels
        self.sulfur_levels = sulfur_levels
        selected_levels = {"Fe II": iron_levels, "S II": sulfur_levels}
        self.atoms = tuple(load_metal_atom(label, levels=selected_levels.get(label))
                           for label, _, _ in atoms)
        self.colliders = tuple(load_metal_colliders(label, levels=selected_levels.get(label))
                               for label, _, _ in atoms)
        self.photo = tuple(
            tuple(self.data["photo"][z, q] for q in range(3)) for z, _ in self.elements
        )
        self.excitation_slices, self.lines = [], []
        offset = 2 * len(self.elements)
        line_offset = 0
        self.line_slices = []
        for atom, (_, z, q) in zip(self.atoms, self.atom_specs):
            count = len(atom.weight)
            self.excitation_slices.append(slice(offset, offset + count - 1))
            offset += count - 1
            upper, lower = np.where(np.asarray(atom.spontaneous) > 0)
            upper, lower = jnp.asarray(upper), jnp.asarray(lower)
            a = atom.spontaneous[upper, lower]
            nu = atom.radiative_wavenumber[upper, lower] * C
            lifetime = jnp.sum(atom.spontaneous, axis=1)[upper]
            kind = (
                jnp.where(lower == 0, CORE, WING)
                if z == 26 and q == 1
                else jnp.full_like(lower, PRD)
            )
            self.lines.append((upper, lower, nu, a, lifetime, kind))
            self.line_slices.append(slice(line_offset, line_offset + len(nu)))
            line_offset += len(nu)
        self.line_frequency_hz = jnp.concatenate([line[2] for line in self.lines])
        self.state_size = offset
        self.threshold_erg = jnp.array(
            [
                [fit.threshold_ev[fit.outer_shell - 1] * EV_ERG for fit in item]
                for item in self.photo
            ]
        )
        # These frequency-only kernels are shared by opacity, photo rates and
        # heat. Prepare the two fixed grids once, outside the atmospheric JIT;
        # expanding all shell fits at every one of the Fe II line frequencies
        # otherwise dominates compilation. Dynamic-frequency callers still
        # evaluate the same differentiable formulas below.
        self._photo_kernel_grids = (self.frequency_hz, self.line_frequency_hz)
        prepare = jax.jit(self._compute_photo_kernels)
        self._photo_kernel_values = tuple(prepare(grid) for grid in self._photo_kernel_grids)
        self.cross = self.cross_sections(self.frequency_hz)

    def oxygen_lybeta_group_metadata(self):
        """Static O I 1025-A multiplet metadata for a shared H/O transfer group.

        Return all six fine-structure transitions, preserving their actual
        frequencies. The external group decides their overlap spectrally.
        ``line_indices`` address this atom's isolated line operators, so the
        caller can replace those operators and remove their old OTS photons.
        """
        if ("O I", 8, 0) not in self.atom_specs:
            raise ValueError("The overlap group requires the O I excitation atom.")
        index = self.atom_specs.index(("O I", 8, 0))
        atom = self.atoms[index]
        upper, lower, nu, a, lifetime, _ = self.lines[index]
        selected = np.flatnonzero(np.isin(np.asarray(upper), [20, 21, 22])
                                  & (np.asarray(lower) < 3))
        if len(selected) != 6:
            raise ValueError("The packaged O I atom lacks its six 1025-A components.")
        selected_upper, selected_lower = upper[selected], lower[selected]
        return dict(
            atom_index=index, line_indices=jnp.asarray(selected),
            upper=selected_upper, lower=selected_lower,
            line_frequency_hz=nu[selected], einstein_a_s=a[selected],
            transition_energy_erg=PLANCK * C * (
                atom.energy_wavenumber[selected_upper] - atom.energy_wavenumber[selected_lower]),
            lower_weight=atom.weight[selected_lower], upper_weight=atom.weight[selected_upper],
            mass_g=jnp.full(6, dict(self.elements)[8] * 1.66053906660e-24),
            damping_hz=lifetime[selected] / (4 * jnp.pi),
        )

    def _shell_cross_sections(self, z, q, frequency_hz):
        fit = self.data["photo"][z, q]
        energy = jnp.asarray(frequency_hz) * PLANCK / EV_ERG
        cross = photoionization_cross_sections(fit, energy)
        if z == 8 and q == 0:
            cross = cross.at[..., fit.outer_shell - 1].set(
                oxygen_ground_valence_cross_section(energy)
            )
        return cross

    def _compute_photo_kernels(self, frequency_hz):
        energy_ev = jnp.asarray(frequency_hz) * PLANCK / EV_ERG
        stages, excited = {}, {}
        for (z, _), group in zip(self.elements, self.photo):
            for q, fit in enumerate(group):
                cross = self._shell_cross_sections(z, q, frequency_hz)
                excess = jnp.maximum(energy_ev[:, None] - fit.threshold_ev, 0) * EV_ERG
                stages[z, q] = (jnp.sum(cross, axis=-1),
                    jnp.sum(cross * excess, axis=-1), cross[..., fit.outer_shell - 1])
        threshold = {"N I": .893, "O I": .856, "Mg II": .779}
        for label, _, _ in self.atom_specs:
            if label in threshold:
                cross = excited_photoionization_cross_section(label, frequency_hz)
                excess = jnp.maximum(PLANCK * frequency_hz
                    - threshold[label] * C * 109737.3156816 * PLANCK, 0.)
                excited[label] = (cross, cross * excess)
        return stages, excited

    def _photo_kernels(self, frequency_hz):
        for grid, kernels in zip(self._photo_kernel_grids, self._photo_kernel_values):
            if frequency_hz is grid:
                return kernels
        return self._compute_photo_kernels(frequency_hz)

    def cross_sections(self, frequency_hz):
        stages, _ = self._photo_kernels(frequency_hz)
        return jnp.stack(
            [
                jnp.stack(
                    [stages[z, q][0] for q in range(3)]
                )
                for z, _ in self.elements
            ]
        )

    def _excited_groups(self, level_population_cm3):
        """Yield selected excited density and ground-weighting conventions."""
        if level_population_cm3 is None:
            return
        for (label, z, q), level in zip(self.atom_specs, level_population_cm3):
            if label == "N I":
                yield (
                    label,
                    z,
                    q,
                    jnp.sum(level[:, 1:3], axis=1),
                    jnp.sum(level, axis=1) - jnp.sum(level[:, 1:3], axis=1),
                )
            elif label == "O I":
                yield label, z, q, level[:, 3], jnp.sum(level[:, :3], axis=1)
            elif label == "Mg II":
                yield label, z, q, jnp.sum(level[:, 1:3], axis=1), None

    def unpack(self, state, element_density_cm3):
        """Return ion[C,element,q] and per-atom levels, conserving each element."""
        state, abundance = jnp.asarray(state), jnp.asarray(element_density_cm3)
        cells, elements = abundance.shape
        charge = state[:, : 2 * elements].reshape(cells, elements, 2)
        charge = jnp.stack(
            (charge[..., 0], jnp.zeros_like(charge[..., 0]), charge[..., 1]), axis=-1
        )
        ions = abundance[..., None] * jax.nn.softmax(charge, axis=-1)
        levels = []
        for (_, z, q), block in zip(self.atom_specs, self.excitation_slices):
            fractions = jax.nn.softmax(
                jnp.column_stack((jnp.zeros(cells), state[:, block])), axis=1
            )
            levels.append(ions[:, self.atomic_number.index(z), q, None] * fractions)
        return ions, tuple(levels)

    def initialize(self, temperature_k, element_density_cm3):
        """LTE excitation with equal ionic fractions; only a numerical seed."""
        temperature = jnp.broadcast_to(temperature_k, (len(self.widths_cm),))
        blocks = [jnp.zeros((len(temperature), 2 * len(self.elements)))]
        for atom in self.atoms:
            excitation = jnp.log(atom.weight[1:] / atom.weight[0])[None, :]
            excitation = excitation - PLANCK * C * (
                atom.energy_wavenumber[1:] - atom.energy_wavenumber[0]
            )[None, :] / (KB * temperature[:, None])
            blocks.append(excitation)
        return jnp.concatenate(blocks, axis=1)

    def continuum_coefficients_at(
        self,
        temperature_k,
        ion_density_cm3,
        electron_cm3,
        frequency_hz,
        *,
        level_population_cm3=None,
    ):
        """Ground-shell absorption and Milne emission on any frequency grid.

        C23 uses total ground-ion shell absorption and a separate ground-capture
        continuum. Total RR/DR coefficients are not replaced by this integral.
        """
        ion = jnp.asarray(ion_density_cm3)
        stages, excited_kernels = self._photo_kernels(frequency_hz)
        opacity = jnp.einsum("csq,sqf->cf", ion, self.cross_sections(frequency_hz))
        for label, z, q, excited, ground in self._excited_groups(level_population_cm3):
            cross = excited_kernels[label][0]
            opacity += excited[:, None] * cross
            if ground is not None:
                index = self.atomic_number.index(z)
                valence = stages[z, q][2]
                opacity += (ground - ion[:, index, q])[:, None] * valence
        emission = jnp.zeros_like(opacity)
        for s, (z, _) in enumerate(self.elements):
            for q in range(2):
                weights = self.data["ground_weight"][z, q]
                emission += ground_recombination_continuum(
                    self.photo[s][q],
                    weights[0] / weights[1],
                    temperature_k,
                    electron_cm3,
                    ion[:, s, q + 1],
                    frequency_hz,
                    valence_cross_section_cm2=stages[z, q][2],
                )
        return opacity, emission

    def hydrogen_line_opacity_at(
        self,
        temperature_k,
        ion_density_cm3,
        electron_cm3,
        frequency_hz,
        level_population_cm3,
        *,
        lyalpha_index=0,
    ):
        """Total metal absorption at H lines, including Ca II excited Ly-alpha PI.

        Four Ca II cross sections are supplied by C23 ``setXtraRatesCa2`` and
        ``ion_photo`` specifically at Ly-alpha;
        no unmeasured broad continuum shape is inferred. Their absorption enters
        H line destruction once, and the same destroyed-photon field feeds Ca
        ionization. It must not also be added as an independent H downward rate.
        """
        absorption, _ = self.continuum_coefficients_at(
            temperature_k,
            ion_density_cm3,
            electron_cm3,
            frequency_hz,
            level_population_cm3=level_population_cm3,
        )
        if ("Ca II", 20, 1) in self.atom_specs:
            index = self.atom_specs.index(("Ca II", 20, 1))
            sigma = jnp.array([4.8e-18, 7.0e-18, 8.4e-19, 1.7e-18])
            absorption = absorption.at[:, lyalpha_index].add(
                jnp.sum(level_population_cm3[index][:, 1:5] * sigma, axis=1)
            )
        return absorption

    def recombination_line_emission(self, temperature_k, ion_density_cm3, electron_cm3):
        """Heavy-ion recombination pseudo-lines: frequencies and photons/cm3/s.

        C23 approximates a cascade by one photon at 3/4 IP and one at 1/4 IP
        per total RR+DR event (ion_recomb.cpp; cont_createpointers.cpp). These
        are source events, not heat. Transport their emission or use a fully
        declared OTS closure; never add both for the same photons.
        """
        frequencies, photons = [], []
        for s, (z, _) in enumerate(self.elements):
            for q in (1, 2):
                rr = radiative_recombination(temperature_k, z, q, self.data)
                dr = raw_dielectronic_recombination(
                    temperature_k, z, q, self.data
                ) * dielectronic_suppression(temperature_k, electron_cm3, z, q)
                events = electron_cm3 * ion_density_cm3[:, s, q] * (rr + dr)
                for fraction in (0.75, 0.25):
                    frequencies.append(fraction * self.threshold_erg[s, q - 1] / PLANCK)
                    photons.append(events)
        return jnp.stack(frequencies), jnp.stack(photons, axis=1)

    def continuum_coefficients(
        self, temperature_k, ion_density_cm3, electron_cm3, *, level_population_cm3=None
    ):
        """Shared-grid continuum, including transported recombination cascades.

        The two adjacent frequency nodes receive nonnegative photon weights
        that conserve both total photons and emitted energy. No additional
        OTS absorption is to be added for these transported pseudo-lines.
        """
        opacity, emission = self.continuum_coefficients_at(
            temperature_k,
            ion_density_cm3,
            electron_cm3,
            self.frequency_hz,
            level_population_cm3=level_population_cm3,
        )
        emission += self.recombination_emissivity(
            temperature_k, ion_density_cm3, electron_cm3,
            self.frequency_hz, self.weight_hz,
        )
        return opacity, emission

    def recombination_emissivity(
        self, temperature_k, ion_density_cm3, electron_cm3,
        frequency_hz, weight_hz, *, partial_coverage=False,
    ):
        """Deposit cascade photons on an explicit frequency quadrature.

        A restricted overlap grid uses ``partial_coverage=True`` to discard
        only sources outside its sampled frequency range. A full continuum
        grid must cover every cascade; missing coverage returns NaN. Within
        the range, two-node deposition conserves photon number and energy.
        """
        grid, weight = jnp.asarray(frequency_hz), jnp.asarray(weight_hz)
        if grid.ndim != 1 or grid.shape != weight.shape or len(grid) < 2:
            raise ValueError("Use matching one-dimensional frequencies and weights.")
        emission = jnp.zeros((len(ion_density_cm3), len(grid)))
        frequency, photons = self.recombination_line_emission(
            temperature_k, ion_density_cm3, electron_cm3
        )
        upper = jnp.clip(
            jnp.searchsorted(grid, frequency), 1, len(grid) - 1,
        )
        lower = upper - 1
        fraction = jnp.clip((frequency - grid[lower]) / (grid[upper] - grid[lower]), 0, 1)
        in_range = (frequency >= grid[0]) & (frequency <= grid[-1])
        photons = jnp.where(in_range[None, :], photons, 0.0 if partial_coverage else jnp.nan)
        emission = emission.at[:, lower].add(
            photons
            * (1 - fraction)
            * PLANCK
            * grid[lower]
            / (4 * jnp.pi * weight[lower])
        )
        emission = emission.at[:, upper].add(
            photons
            * fraction
            * PLANCK
            * grid[upper]
            / (4 * jnp.pi * weight[upper])
        )
        valid = jnp.all(jnp.diff(grid) > 0) & jnp.all(weight > 0)
        return jnp.where(valid, emission, jnp.nan)

    def _photo_rates_and_heat(self, photon_flux, frequency_hz):
        rates, heat = [], []
        stages, _ = self._photo_kernels(frequency_hz)
        for z, _ in self.elements:
            group_rate, group_heat = [], []
            for q in range(3):
                cross, energy, _ = stages[z, q]
                group_rate.append(jnp.einsum("cf,f->c", photon_flux, cross))
                group_heat.append(jnp.einsum("cf,f->c", photon_flux, energy))
            rates.append(jnp.stack(group_rate, axis=1))
            heat.append(jnp.stack(group_heat, axis=1))
        return jnp.stack(rates, axis=1), jnp.stack(heat, axis=1)

    def _excited_photo_correction(self, ion, levels, photon_flux, frequency_hz):
        rates, heat = jnp.zeros_like(ion), jnp.zeros_like(ion)
        loss = {}
        stages, excited_kernels = self._photo_kernels(frequency_hz)
        for label, z, q, excited, ground in self._excited_groups(levels):
            index = self.atomic_number.index(z)
            density = ion[:, index, q]
            safe = jnp.maximum(density, jnp.finfo(density.dtype).tiny)
            cross, heat_kernel = excited_kernels[label]
            gamma = jnp.einsum("cf,f->c", photon_flux, cross)
            energy = jnp.einsum("cf,f->c", photon_flux, heat_kernel)
            correction = gamma * excited / safe
            energy_correction = energy * excited / safe
            if ground is not None:
                valence = stages[z, q][2]
                ground_rate = jnp.einsum("cf,f->c", photon_flux, valence)
                ground_excess = jnp.maximum(
                    PLANCK * frequency_hz - self.threshold_erg[index, q], 0.0
                )
                ground_heat = jnp.einsum(
                    "cf,f->c", photon_flux, valence * ground_excess
                )
                correction += ground_rate * (ground - density) / safe
                energy_correction += ground_heat * (ground - density) / safe
            rates = rates.at[:, index, q].add(correction)
            heat = heat.at[:, index, q].add(energy_correction)
            loss[label] = gamma
        return rates, heat, loss

    def line_geometry(self, atom_index, temperature_k, level_population_cm3):
        """Shared Doppler opacity and columns for an atom and coupled line groups."""
        atom = self.atoms[atom_index]
        _, z, _ = self.atom_specs[atom_index]
        upper, lower, nu, a, lifetime, _ = self.lines[atom_index]
        mass = dict(self.elements)[z] * 1.66053906660e-24
        width = nu[None, :] * jnp.sqrt(2*KB*jnp.asarray(temperature_k)[:, None]/mass) / C
        damping = lifetime[None, :] / (4*jnp.pi*width)
        factor = C*C*a[None, :] / (8*jnp.pi*nu[None, :]**2*jnp.sqrt(jnp.pi)*width)
        absorption = factor*level_population_cm3[:, lower]*atom.weight[upper]/atom.weight[lower]
        kappa = absorption-factor*level_population_cm3[:, upper]
        delta = kappa*self.widths_cm[:, None]
        front = jnp.concatenate(
            (jnp.cumsum(delta[::-1], axis=0)[:-1][::-1], jnp.zeros_like(delta[:1])), axis=0)
        return dict(damping=damping, lower_absorption_cm1=absorption,
                    extinction_cm1=kappa, increment=delta, tau_in=front,
                    tau_total=self.line_optical_depth_scale*jnp.sum(delta, axis=0))

    def evaluate(
        self,
        temperature_k,
        state,
        element_density_cm3,
        electron_cm3,
        hydrogen_ground_cm3,
        proton_cm3,
        helium_ground_cm3,
        helium_ion_cm3,
        radiation,
        hydrogen_line_frequency_hz,
        *,
        additional_ionization_s=0.0,
        additional_recombination_s=0.0,
        line_incident_mean_intensity=None,
        line_continuum_opacity_cm1=None,
        line_diffuse_mean_intensity=None,
        line_node_diffuse_mean_intensity=None,
        line_diffuse_field=None,
        line_mean_intensity_override=None,
        additional_collider_density_cm3=None,
        hydrogen_neutral_cm3=None,
        line_scattering_cm1=0.0,
        lyalpha_index=0,
        line_rate_overrides=None,
        additional_line_rates=None,
        additional_photon_frequency_hz=None,
        additional_photon_flux_cm2_s=None,
    ):
        """Return the population fixed point, reaction feedback and energy ledger.

        ``radiation`` contains continuum J/pump J, positive absorption, and H-line
        destroyed photon fluxes, from the common transfer solution. Additional
        ionic rates have shape [cell,element,2] and permit chemical backreaction.
        Their energy is owned by the caller, not silently counted here.

        For structured stellar spectra, supply J at each exact metal line center.
        The direct normal beam is attenuated with the supplied exact-frequency
        total absorption and averaged over each cell. Only the diffuse field may
        be interpolated from the shared continuum grid. The default interpolation
        of the entire pump field is suitable for smooth-continuum controls only.
        With cell-averaged lines, optional node diffuse fields have shape
        [line_quadrature_order, cell, line]. Without them the supplied diffuse
        mean is held constant within a cell; the direct beam still varies at
        the exact quadrature positions.
        Optical-depth quadrature instead requires ``line_diffuse_field`` from
        ``HydrogenEscapeProbabilityModel.forward_diffuse_field`` and the separate
        exact stellar input. Both bracketing-frequency formal solutions are
        evaluated before interpolation at every per-line quadrature node.
        ``line_quadrature_normalization_error`` reports each cell's largest
        raw mapped weight-sum error (zero when no remapping is used). It is a
        diagnostic, not an added validity threshold or accuracy certificate.

        ``line_mean_intensity_override=(mask, mean)`` replaces the complete
        pumping field at selected exact line centers, with shapes [line] and
        [cell,line]. The mask must be boolean. This cell-volume mean is held
        constant at every depth node; it is not added to the beam or diffuse
        field. Line shielding, escape, destruction and photon ownership retain
        their existing definitions. Only selected means must be finite and
        nonnegative. A supplied volume mean is an explicit spatial reduction,
        not a reconstruction of the incident angular or depth distribution.

        A coupled line group may replace isolated rates through
        ``{atom_label: (local_line_indices, upward, downward, destroyed_photons)}``.
        These arrays have shape [cell, selected_line]; downward includes induced
        emission. Set its old destroyed photons to zero when the group returns
        its own radiation field. Additional photons are the quadrature-integrated
        signed difference from the continuum-only solution on the same grid and
        boundaries, not the total field (which would count continuum twice).
        additional_line_rates maps an atom label to (upward, downward) arrays
        over all its lines. These exchanges add level rates, while their
        photon/energy accounting belongs to the caller; continuum OTS is unchanged.
        """
        temperature = jnp.broadcast_to(temperature_k, (len(self.widths_cm),))
        ne = jnp.asarray(electron_cm3)
        neutral_h = (
            hydrogen_ground_cm3
            if hydrogen_neutral_cm3 is None
            else hydrogen_neutral_cm3
        )
        # C23's explicit neutral-H collision surrogate, not extra electrons.
        # It is also used for heavy-ion collisional ionization (ion_collis.cpp).
        collision_density = ne + 1.7e-4 * neutral_h
        ion, levels_by_atom = self.unpack(state, element_density_cm3)
        photon_integral = (
            4
            * jnp.pi
            * radiation["mean_intensity"]
            / (PLANCK * self.frequency_hz)
            * self.weight_hz
        )
        photo, photo_heat = self._photo_rates_and_heat(
            photon_integral, self.frequency_hz
        )
        hphoto, hheat = self._photo_rates_and_heat(
            radiation["ots_photon_flux_cm2_s"], hydrogen_line_frequency_hz
        )
        photo, photo_heat = photo + hphoto, photo_heat + hheat
        if (additional_photon_frequency_hz is None) != (
            additional_photon_flux_cm2_s is None
        ):
            raise ValueError("Supply both additional photon frequencies and fluxes.")
        if additional_photon_frequency_hz is not None:
            correction, energy = self._photo_rates_and_heat(
                additional_photon_flux_cm2_s, additional_photon_frequency_hz
            )
            photo, photo_heat = photo + correction, photo_heat + energy
        if line_rate_overrides is None:
            line_rate_overrides = {}
        if set(line_rate_overrides) - {spec[0] for spec in self.atom_specs}:
            raise ValueError("Line overrides must reference a selected atom.")
        additional_line_rates = {} if additional_line_rates is None else additional_line_rates
        if set(additional_line_rates) - {spec[0] for spec in self.atom_specs}:
            raise ValueError("Additional line rates must reference a selected atom.")
        if line_continuum_opacity_cm1 is None:
            line_continuum_opacity_cm1 = jax.vmap(
                lambda row: jnp.interp(self.line_frequency_hz, self.frequency_hz, row)
            )(radiation["continuum_opacity_cm1"])
        node_pump = None
        if self.line_quadrature == "optical_depth" and (
            line_diffuse_field is None or line_incident_mean_intensity is None
            or line_node_diffuse_mean_intensity is not None
        ):
            raise ValueError("Optical-depth lines require a prepared diffuse field and exact stellar input.")
        if line_incident_mean_intensity is None:
            if line_node_diffuse_mean_intensity is not None:
                raise ValueError("Node diffuse fields require a separate direct stellar field.")
            pump = jax.vmap(
                lambda row: jnp.interp(self.line_frequency_hz, self.frequency_hz, row)
            )(radiation["pump_mean_intensity"])
        else:
            if line_diffuse_mean_intensity is None:
                raise ValueError(
                    "Supply the diffuse line field separately from exact stellar line pumping."
                )
            depth = line_continuum_opacity_cm1 * self.widths_cm[:, None]
            depth += jnp.log1p(
                jnp.asarray(line_scattering_cm1) * self.widths_cm[:, None]
            )
            front = jnp.concatenate(
                (jnp.cumsum(depth[::-1], axis=0)[:-1][::-1], jnp.zeros_like(depth[:1])),
                axis=0,
            )
            beam_front, beam_depth = front, depth
            safe = jnp.maximum(depth, 1e-5)
            average = jnp.where(
                depth > 1e-5, -jnp.expm1(-safe) / safe, 1 - depth / 2 + depth**2 / 6
            )
            pump = (
                line_incident_mean_intensity * jnp.exp(-front) * average
                + line_diffuse_mean_intensity
            )
            if self.line_sampling == "average" and self.line_quadrature == "gauss":
                node_diffuse = line_node_diffuse_mean_intensity
                if node_diffuse is None:
                    node_diffuse = jnp.asarray(line_diffuse_mean_intensity)[None, ...]
                elif jnp.shape(node_diffuse) != (
                    self.line_quadrature_order, len(state), len(self.line_frequency_hz)
                ):
                    raise ValueError("Node diffuse fields must have shape [quadrature,cell,line].")
                node_pump = (
                    line_incident_mean_intensity[None, None, :]
                    * jnp.exp(-front[None, ...]
                              - self._line_nodes[:, None, None] * depth[None, ...])
                    + node_diffuse
                )
        override_valid = jnp.asarray(True)
        if line_mean_intensity_override is not None:
            mask, mean = map(jnp.asarray, line_mean_intensity_override)
            shape = (len(state), len(self.line_frequency_hz))
            if mask.shape != (shape[1],) or mask.dtype != jnp.bool_ or mean.shape != shape:
                raise ValueError("Line mean overrides require a boolean [line] mask and [cell,line] means.")
            override_valid = jnp.all(jnp.where(mask[None, :],
                                               jnp.isfinite(mean) & (mean >= 0), True))
            line_mean_intensity_override = mask, mean
            pump = jnp.where(mask[None, :], mean, pump)
            if node_pump is not None:
                node_pump = jnp.where(mask[None, None, :], mean[None, ...], node_pump)
        excitation, line_heat, destroyed = [], [], []
        radiative_matrices, collision_matrices = [], []
        valid = override_valid
        quadrature_error = jnp.zeros_like(ne)
        for atom_index, (atom, spec, levels, line, block, tables) in enumerate(zip(
            self.atoms,
            self.atom_specs,
            levels_by_atom,
            self.lines,
            self.line_slices,
            self.colliders,
        )):
            _, z, q = spec
            upper, lower, nu, a, lifetime, kind = line
            g = atom.weight
            geometry = self.line_geometry(atom_index, temperature, levels)
            damping, absorption, kappa, delta, front = (geometry[key] for key in (
                "damping", "lower_absorption_cm1", "extinction_cm1", "increment", "tau_in"))
            continuum = line_continuum_opacity_cm1[:, block]
            operator, sampling = line_operators, {}
            if self.line_sampling == "average":
                operator = cell_averaged_line_operators
                sampling = dict(
                    line_optical_depth_increment=delta,
                    quadrature_order=self.line_quadrature_order,
                    node_incident_occupation=(
                        None if node_pump is None
                        else node_pump[:, :, block] / (2 * PLANCK * nu**3 / C**2)
                    ),
                )
                if self.line_quadrature == "optical_depth":
                    selected_mean = (None if line_mean_intensity_override is None else
                        (line_mean_intensity_override[0][block],
                         line_mean_intensity_override[1][:, block]))
                    operator = lambda selected_kind, **kwargs: _depth_averaged_metal_lines(
                        selected_kind, delta, nu, line_incident_mean_intensity[block],
                        beam_front[:, block], beam_depth[:, block], line_diffuse_field,
                        self.line_quadrature_order, return_normalization_error=True,
                        line_mean_intensity_override=selected_mean, **kwargs)
                    sampling = {}
            # Recompute inexpensive line-transfer intermediates during AD.
            # Saving every quadrature/line array would dominate root derivatives.
            operators = jax.checkpoint(
                lambda: operator(
                    kind,
                    tau_in=front,
                    tau_total=geometry["tau_total"],
                    tau_pump=front,
                    damping=damping,
                    line_extinction_cm1=kappa,
                    lower_absorption_cm1=absorption,
                    continuum_absorption_cm1=continuum,
                    electron_cm3=ne[:, None],
                    einstein_a_s=a,
                    lower_weight=g[lower],
                    upper_weight=g[upper],
                    incident_occupation=pump[:, block] / (2 * PLANCK * nu**3 / C**2),
                    outward_continuum_occupation=0.0,
                    zone_width_cm=self.widths_cm[:, None],
                    upper_population_cm3=levels[:, upper],
                    shielding_model=self.shielding_model,
                    core_escape_model=self.core_escape_model,
                    **sampling,
                ),
                prevent_cse=False,
            )()
            if self.line_quadrature == "optical_depth":
                operators, error = operators
                quadrature_error = jnp.maximum(quadrature_error, jnp.max(jnp.abs(error), axis=1))
            if spec[0] in line_rate_overrides:
                indices, up, down, photons = line_rate_overrides[spec[0]]
                indices = jnp.asarray(indices, dtype=int)
                shape = (len(state), len(indices))
                if indices.ndim != 1 or any(
                    jnp.shape(array) != shape for array in (up, down, photons)
                ):
                    raise ValueError("Line override rates must have shape [cell,line].")
                valid &= jnp.all((indices >= 0) & (indices < len(nu)))
                valid &= jnp.all(jnp.diff(jnp.sort(indices)) > 0)
                valid &= jnp.all(jnp.stack([
                    jnp.all(jnp.asarray(value) >= 0) for value in (up, down, photons)
                ]))
                operators = operators._replace(
                    pump_up_s=operators.pump_up_s.at[:, indices].set(up),
                    radiative_down_s=operators.radiative_down_s.at[:, indices].set(down),
                    destroyed_photons_cm3_s=operators.destroyed_photons_cm3_s.at[:, indices].set(photons),
                )
            if spec[0] in additional_line_rates:
                extra_up, extra_down = additional_line_rates[spec[0]]
                shape = (len(state), len(nu))
                if any(jnp.shape(value) != shape for value in (extra_up, extra_down)):
                    raise ValueError("Additional line rates must have shape [cell, atom line].")
                valid &= jnp.all(jnp.isfinite(extra_up) & (extra_up >= 0))
                valid &= jnp.all(jnp.isfinite(extra_down) & (extra_down >= 0))
                operators = operators._replace(
                    pump_up_s=operators.pump_up_s+extra_up,
                    radiative_down_s=operators.radiative_down_s+extra_down,
                )
            radiative = (
                jnp.zeros((len(state), len(g), len(g)))
                .at[:, lower, upper]
                .add(operators.pump_up_s)
            )
            radiative = radiative.at[:, upper, lower].add(operators.radiative_down_s)
            collision = collision_density[:, None, None] * jax.vmap(
                lambda t: electron_collision_coefficients(atom, t)
            )(temperature)
            collider_density = {
                "H": neutral_h,
                "PROTON": proton_cm3,
                "HE": helium_ground_cm3,
                "HE+": helium_ion_cm3,
            }
            if additional_collider_density_cm3 is not None:
                collider_density.update(additional_collider_density_cm3)
            for name, table in tables.items():
                if name in collider_density:
                    coefficients = jax.vmap(
                        lambda t: tabulated_collision_coefficients(atom, table, t)
                    )(temperature)
                    collision += (
                        jnp.asarray(collider_density[name])[:, None, None]
                        * coefficients
                    )
            radiative_matrices.append(radiative)
            collision_matrices.append(collision)
            line_heat.append(
                jax.vmap(kinetic_line_heating, in_axes=(0, 0, None))(
                    levels, collision, atom.energy_wavenumber
                )
            )
            # Zero opacity means no destruction. Do not manufacture an absorber.
            safe_continuum = jnp.where(continuum > 0, continuum, 1.0)
            destroyed.append(
                jnp.where(
                    continuum > 0,
                    operators.destroyed_photons_cm3_s / safe_continuum,
                    0.0,
                )
            )
            valid &= jnp.all(continuum >= 0)
        event_flux = jnp.concatenate(destroyed, axis=1)
        mphoto, mheat = self._photo_rates_and_heat(event_flux, self.line_frequency_hz)
        photo, photo_heat = photo + mphoto, photo_heat + mheat
        calcium_events = jnp.zeros_like(ne)
        calcium_loss = None
        if ("Ca II", 20, 1) in self.atom_specs:
            atom_index = self.atom_specs.index(("Ca II", 20, 1))
            element_index = self.atomic_number.index(20)
            sigma = jnp.array([4.8e-18, 7.0e-18, 8.4e-19, 1.7e-18])
            calcium_loss = (
                radiation["ots_photon_flux_cm2_s"][:, lyalpha_index, None] * sigma
            )
            events = levels_by_atom[atom_index][:, 1:5] * calcium_loss
            calcium_events = jnp.sum(events, axis=1)
            denominator = jnp.maximum(
                ion[:, element_index, 1], jnp.finfo(temperature.dtype).tiny
            )
            photo = photo.at[:, element_index, 1].add(calcium_events / denominator)
            threshold = (
                self.threshold_erg[element_index, 1]
                - PLANCK * C * self.atoms[atom_index].energy_wavenumber[1:5]
            )
            excess = jnp.maximum(
                PLANCK * hydrogen_line_frequency_hz[lyalpha_index] - threshold, 0.0
            )
            photo_heat = photo_heat.at[:, element_index, 1].add(
                jnp.sum(events * excess, axis=1) / denominator
            )
        excited_loss = {}
        photo_fields = [
            (photon_integral, self.frequency_hz),
            (radiation["ots_photon_flux_cm2_s"], hydrogen_line_frequency_hz),
            (event_flux, self.line_frequency_hz),
        ]
        if additional_photon_frequency_hz is not None:
            photo_fields.append((additional_photon_flux_cm2_s, additional_photon_frequency_hz))
        for photons, frequencies in photo_fields:
            correction, energy, loss = self._excited_photo_correction(
                ion, levels_by_atom, photons, frequencies
            )
            photo, photo_heat = photo + correction, photo_heat + energy
            for label, value in loss.items():
                excited_loss[label] = (
                    excited_loss.get(label, jnp.zeros_like(value)) + value
                )
        valid &= jnp.all(photo >= 0)
        for spec, levels, radiative, collision in zip(
            self.atom_specs, levels_by_atom, radiative_matrices, collision_matrices
        ):
            # C23 recycles selected excited-ion photo-loss to the ground term
            # in its normalized excitation matrix; charge changes separately.
            if spec[0] == "N I":
                radiative = radiative.at[:, 1:3, 0].add(excited_loss["N I"][:, None])
            elif spec[0] == "Ca II":
                radiative = radiative.at[:, 1:5, 0].add(calcium_loss)
            elif spec[0] == "O I":
                radiative = radiative.at[:, 3, :3].add(excited_loss["O I"][:, None] / 3)
            solved = jax.vmap(populations)(
                radiative + collision, jnp.sum(levels, axis=1)
            )
            excitation.append(jnp.log(solved[:, 1:] / solved[:, :1]))
            valid &= jnp.all(jnp.isfinite(solved))
        extra_up = jnp.broadcast_to(additional_ionization_s, ion[:, :, :2].shape)
        extra_down = jnp.broadcast_to(additional_recombination_s, ion[:, :, :2].shape)
        charge_state, rr_cool, ci_cool, ct_heat, cota_binding = [], [], [], [], []
        h_ion = jnp.zeros_like(ne)
        h_rec = jnp.zeros_like(ne)
        he_ion = jnp.zeros_like(ne)
        he_rec = jnp.zeros_like(ne)
        for s, (z, _) in enumerate(self.elements):
            upward, downward = [], []
            rr_energy = jnp.zeros_like(ne)
            ci_energy = jnp.zeros_like(ne)
            ct_energy = jnp.zeros_like(ne)
            three_energy = jnp.zeros_like(ne)
            for q in (1, 2):
                hi, hr, hei, her = charge_exchange(temperature, z, q, self.data)
                rr = radiative_recombination(temperature, z, q, self.data)
                dr = raw_dielectronic_recombination(
                    temperature, z, q, self.data
                ) * dielectronic_suppression(temperature, ne, z, q)
                cota = three_body_recombination(temperature, ne, q, self.data["cota"])
                ci = collisional_ionization(temperature, z, q - 1, self.data)
                upward.append(
                    photo[:, s, q - 1]
                    + collision_density * ci
                    + hi * proton_cm3
                    + hei * helium_ion_cm3
                    + extra_up[:, s, q - 1]
                )
                downward.append(
                    ne * (rr + dr + cota)
                    + hr * hydrogen_ground_cm3
                    + her * helium_ground_cm3
                    + extra_down[:, s, q - 1]
                )
                h_ion += hr * ion[:, s, q]
                h_rec += hi * ion[:, s, q - 1]
                he_ion += her * ion[:, s, q]
                he_rec += hei * ion[:, s, q - 1]
                rr_energy += ne * ion[:, s, q] * rr * KB * temperature
                ci_energy += (
                    collision_density
                    * ion[:, s, q - 1]
                    * ci
                    * self.threshold_erg[s, q - 1]
                )
                ct_energy += (
                    hi
                    * proton_cm3
                    * ion[:, s, q - 1]
                    * self.data["ct_ion"][z - 1, q - 1, 7]
                    + hr
                    * hydrogen_ground_cm3
                    * ion[:, s, q]
                    * self.data["ct_rec"][z - 1, q - 1, 6]
                ) * EV_ERG
                three_energy += ne * ion[:, s, q] * cota * self.threshold_erg[s, q - 1]
            # Exact-zero rates remain zero physically; logarithmic state targets
            # use only the smallest representable positive coordinate value.
            tiny = jnp.finfo(temperature.dtype).tiny
            log_up = [jnp.log(jnp.maximum(value, tiny)) for value in upward]
            log_down = [jnp.log(jnp.maximum(value, tiny)) for value in downward]
            valid &= jnp.all((upward[0] + downward[0]) > 0) & jnp.all(
                (upward[1] + downward[1]) > 0
            )
            charge_state.append(
                jnp.column_stack((log_down[0] - log_up[0], log_up[1] - log_down[1]))
            )
            rr_cool.append(rr_energy)
            ci_cool.append(ci_energy)
            ct_heat.append(ct_energy)
            cota_binding.append(three_energy)
        energy = dict(
            photoelectron_heat=jnp.sum(ion * photo_heat, axis=(1, 2)),
            collision_line_heat=jnp.sum(jnp.stack(line_heat, axis=1), axis=1),
            radiative_recombination_cooling=jnp.sum(jnp.stack(rr_cool, axis=1), axis=1),
            collisional_ionization_cooling=jnp.sum(jnp.stack(ci_cool, axis=1), axis=1),
            hydrogen_charge_exchange_heat=jnp.sum(jnp.stack(ct_heat, axis=1), axis=1),
        )
        energy["net_kinetic_heat"] = (
            energy["photoelectron_heat"]
            + energy["collision_line_heat"]
            + energy["hydrogen_charge_exchange_heat"]
            - energy["radiative_recombination_cooling"]
            - energy["collisional_ionization_cooling"]
        )
        return dict(
            target_state=jnp.concatenate(charge_state + excitation, axis=1),
            ion_density_cm3=ion,
            level_population_cm3=levels_by_atom,
            singly_ionized_density_cm3=ion[:, :, 1],
            electron_density_cm3=jnp.einsum("csq,q->c", ion, jnp.arange(3)),
            collision_heat_by_atom_erg_cm3_s=jnp.stack(line_heat, axis=1),
            energy_erg_cm3_s=energy,
            three_body_binding_energy_erg_cm3_s=jnp.stack(cota_binding, axis=1),
            destroyed_line_frequency_hz=self.line_frequency_hz,
            ots_photon_flux_cm2_s=event_flux,
            photoionization_s=photo,
            calcium_lyalpha_photoionization_events_cm3_s=calcium_events,
            upper_charge_photoionization_events_cm3_s=ion[:, :, 2] * photo[:, :, 2],
            hydrogen_ct_ionization_s=h_ion,
            hydrogen_ct_recombination_s=h_rec,
            helium_ct_ionization_s=he_ion,
            helium_ct_recombination_s=he_rec,
            line_quadrature_normalization_error=quadrature_error,
            valid=valid,
        )
