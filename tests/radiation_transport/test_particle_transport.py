# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.radiation_transport.particle_transport import (
    ParticleTermination,
    walk_particle,
)
from bluemira.radiation_transport.surface_hits import SurfaceHit


def _constant_z_field(_position):
    return np.array([0.0, 0.0, 2.0])


def _no_hit(_start, _end):
    return None


def test_zero_diffusion_is_a_straight_parallel_walk():
    result = walk_particle(
        [0.0, 0.0, 0.0],
        _constant_z_field,
        _no_hit,
        parallel_step=0.5,
        diffusion_coefficient=0.0,
        max_steps=3,
        seed=17,
    )

    assert result.termination is ParticleTermination.MAX_STEPS
    assert result.iterations == 3
    assert result.hit is None
    np.testing.assert_allclose(
        result.positions,
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.5],
        ],
    )


def test_same_seed_replays_the_exact_diffusive_trajectory():
    kwargs = {
        "parallel_step": 0.25,
        "diffusion_coefficient": 0.04,
        "max_steps": 5,
        "seed": 1234,
    }
    first = walk_particle(
        [1.0, 0.0, 0.0], _constant_z_field, _no_hit, **kwargs
    )
    second = walk_particle(
        [1.0, 0.0, 0.0], _constant_z_field, _no_hit, **kwargs
    )

    np.testing.assert_array_equal(first.positions, second.positions)
    assert first.termination is second.termination


def test_different_seeds_change_the_diffusive_trajectory():
    first = walk_particle(
        [0.0, 0.0, 0.0],
        _constant_z_field,
        _no_hit,
        parallel_step=0.5,
        diffusion_coefficient=0.2,
        max_steps=2,
        seed=1,
    )
    second = walk_particle(
        [0.0, 0.0, 0.0],
        _constant_z_field,
        _no_hit,
        parallel_step=0.5,
        diffusion_coefficient=0.2,
        max_steps=2,
        seed=2,
    )

    assert not np.array_equal(first.positions, second.positions)


def test_diffusive_step_has_requested_length_and_is_perpendicular():
    parallel_step = 0.5
    diffusion_coefficient = 0.18
    result = walk_particle(
        [0.0, 0.0, 0.0],
        _constant_z_field,
        _no_hit,
        parallel_step=parallel_step,
        diffusion_coefficient=diffusion_coefficient,
        max_steps=1,
        seed=7,
    )

    parallel_end = result.positions[1]
    diffusion_delta = result.positions[2] - parallel_end
    expected_length = np.sqrt(diffusion_coefficient * parallel_step)

    assert np.linalg.norm(diffusion_delta) == pytest.approx(expected_length)
    assert np.dot(diffusion_delta, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_surface_hit_stops_at_exact_intersection():
    def plane_hit(start, end):
        if start[2] <= 0.4 <= end[2]:
            fraction = (0.4 - start[2]) / (end[2] - start[2])
            point = start + fraction * (end - start)
            return SurfaceHit(
                point=point,
                step_fraction=float(fraction),
                distance=float(np.linalg.norm(point - start)),
                surface_index=2,
                component_id="first_wall",
            )
        return None

    result = walk_particle(
        [0.0, 0.0, 0.0],
        _constant_z_field,
        plane_hit,
        parallel_step=1.0,
        diffusion_coefficient=0.1,
        max_steps=10,
        seed=5,
    )

    assert result.termination is ParticleTermination.SURFACE_HIT
    assert result.iterations == 1
    assert result.hit is not None
    assert result.hit.component_id == "first_wall"
    np.testing.assert_allclose(result.positions, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])


def test_invalid_field_has_explicit_bounded_termination():
    result = walk_particle(
        [0.0, 0.0, 0.0],
        lambda _position: np.zeros(3),
        _no_hit,
        parallel_step=0.1,
        diffusion_coefficient=0.1,
        max_steps=10,
        seed=1,
    )

    assert result.termination is ParticleTermination.INVALID_FIELD
    assert result.iterations == 1
    np.testing.assert_array_equal(result.positions, [[0.0, 0.0, 0.0]])


def test_backward_walk_follows_negative_field_direction():
    result = walk_particle(
        [0.0, 0.0, 0.0],
        _constant_z_field,
        _no_hit,
        parallel_step=0.25,
        diffusion_coefficient=0.0,
        max_steps=2,
        seed=3,
        forward=False,
    )

    np.testing.assert_allclose(
        result.positions,
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.25], [0.0, 0.0, -0.5]],
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parallel_step": 0.0}, "parallel_step"),
        ({"diffusion_coefficient": -1.0}, "diffusion_coefficient"),
        ({"max_steps": 0}, "max_steps"),
        ({"field_tolerance": -1.0}, "field_tolerance"),
    ],
)
def test_invalid_scalar_parameters_are_rejected(kwargs, message):
    defaults = {
        "parallel_step": 0.1,
        "diffusion_coefficient": 0.0,
        "max_steps": 1,
        "seed": 1,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        walk_particle(
            [0.0, 0.0, 0.0],
            _constant_z_field,
            _no_hit,
            **defaults,
        )
