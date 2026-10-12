# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Physical-invariant regressions for a candidate midpoint walker.

For a purely toroidal magnetic field, zero-diffusion field lines have
constant cylindrical radius. Midpoint reduces endpoint drift, but straight
collision chords still require step-size convergence near a PFC surface.
"""

import numpy as np
import pytest

from bluemira.radiation_transport.particle_transport import (
    ParticleTermination,
    walk_particle,
)
from bluemira.radiation_transport.surface_hits import axisymmetric_surface_hit


def _toroidal_field(position):
    # B_phi magnitude is irrelevant: the walker normalizes the field.
    return np.array([-position[1], position[0], 0.0])


def test_zero_diffusion_preserves_toroidal_flux_surface():
    result = walk_particle(
        [2.0, 0.0, 0.0],
        _toroidal_field,
        lambda _start, _end: None,
        parallel_step=0.1,
        diffusion_coefficient=0.0,
        max_steps=126,
        seed=0,
    )

    assert result.termination is ParticleTermination.MAX_STEPS
    radius = np.linalg.norm(result.positions[:, :2], axis=1)
    assert np.max(np.abs(radius - 2.0)) < 1e-3


def test_zero_diffusion_does_not_strike_outside_toroidal_wall():
    def wall_hit(start, end):
        return axisymmetric_surface_hit(
            start,
            end,
            [[2.2, -1.0], [2.2, 1.0]],
            component_ids=["outer_wall"],
        )

    result = walk_particle(
        [2.0, 0.0, 0.0],
        _toroidal_field,
        wall_hit,
        parallel_step=0.1,
        diffusion_coefficient=0.0,
        max_steps=126,
        seed=0,
    )

    assert result.termination is ParticleTermination.MAX_STEPS
    assert result.hit is None

@pytest.mark.parametrize("step", [0.2, 0.1, 0.05])
def test_helical_field_radius_is_nearly_constant(step):
    def helical_field(position):
        return np.array([-position[1], position[0], 0.5])

    n_steps = round(2 * np.pi * np.hypot(2.0, 0.5) / step)
    result = walk_particle(
        [2.0, 0.0, 0.0],
        helical_field,
        lambda _start, _end: None,
        parallel_step=step,
        diffusion_coefficient=0.0,
        max_steps=n_steps,
        seed=7,
    )

    radius = np.linalg.norm(result.positions[:, :2], axis=1)
    assert result.termination is ParticleTermination.MAX_STEPS
    assert abs(radius[-1] - 2.0) < step**2


def test_midpoint_invalid_field_returns_explicit_failure():
    def field(position):
        if position[0] <= 0.0:
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    result = walk_particle(
        [0.0, 0.0, 0.0],
        field,
        lambda _start, _end: None,
        parallel_step=1.0,
        diffusion_coefficient=0.0,
        max_steps=3,
        seed=1,
    )

    assert result.termination is ParticleTermination.INVALID_FIELD
    assert result.hit is None
    np.testing.assert_array_equal(result.positions, [[0.0, 0.0, 0.0]])


def _inner_wall(start, end):
    return axisymmetric_surface_hit(
        start, end, [[1.999, -1.0], [1.999, 1.0]], component_ids=["inner_wall"]
    )


def test_inner_wall_is_not_hit_when_step_is_refined():
    result = walk_particle(
        [2.0, 0.0, 0.0],
        _toroidal_field,
        _inner_wall,
        parallel_step=0.1,
        diffusion_coefficient=0.0,
        max_steps=126,
        seed=4,
    )

    assert result.termination is ParticleTermination.MAX_STEPS
    assert result.hit is None


@pytest.mark.xfail(
    strict=True,
    reason="Finite midpoint chord may intersect nearby inner wall at coarse step",
)
def test_coarse_step_inner_wall_chord_is_a_known_limitation():
    result = walk_particle(
        [2.0, 0.0, 0.0],
        _toroidal_field,
        _inner_wall,
        parallel_step=0.2,
        diffusion_coefficient=0.0,
        max_steps=63,
        seed=4,
    )

    assert result.termination is ParticleTermination.MAX_STEPS
    assert result.hit is None

