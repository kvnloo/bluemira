# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Bounded subdivision contracts for experimental curved-field transport."""

import numpy as np
import pytest

from bluemira.radiation_transport.particle_transport import (
    ParticleTermination,
    walk_particle,
)
from bluemira.radiation_transport.surface_hits import axisymmetric_surface_hit


def _toroidal_field(position):
    return [-position[1], position[0], 0.0]


def _no_hit(_start, _end):
    return None


def _cylinder(radius):
    return lambda a, b: axisymmetric_surface_hit(
        a, b, [[radius, -1.0], [radius, 1.0]], component_ids=["wall"]
    )


@pytest.mark.parametrize("substeps", [2, 4, 8])
def test_refinement_reduces_toroidal_drift(substeps):
    options = dict(
        parallel_step=0.2, diffusion_coefficient=0.0, max_steps=63, seed=7
    )
    baseline = walk_particle([2.0, 0.0, 0.0], _toroidal_field, _no_hit, **options)
    refined = walk_particle(
        [2.0, 0.0, 0.0], _toroidal_field, _no_hit,
        parallel_substeps=substeps, **options
    )
    base_error = abs(np.linalg.norm(baseline.positions[-1, :2]) - 2.0)
    refined_error = abs(np.linalg.norm(refined.positions[-1, :2]) - 2.0)
    assert refined.termination is ParticleTermination.MAX_STEPS
    assert refined_error < base_error / (substeps**2)
    assert len(refined.positions) == 1 + options["max_steps"] * substeps


@pytest.mark.parametrize("substeps", [2, 4])
def test_refinement_removes_known_one_mm_inner_wall_false_hit(substeps):
    result = walk_particle(
        [2.0, 0.0, 0.0], _toroidal_field, _cylinder(1.999),
        parallel_step=0.2, diffusion_coefficient=0.0, max_steps=63, seed=7,
        parallel_substeps=substeps,
    )
    assert result.termination is ParticleTermination.MAX_STEPS
    assert result.hit is None


def test_substeps_one_retains_original_one_mm_false_hit():
    result = walk_particle(
        [2.0, 0.0, 0.0], _toroidal_field, _cylinder(1.999),
        parallel_step=0.2, diffusion_coefficient=0.0, max_steps=63, seed=7,
        parallel_substeps=1,
    )
    assert result.termination is ParticleTermination.SURFACE_HIT
    assert result.hit.component_id == "wall"


def test_four_substeps_remove_half_mm_false_hit():
    result = walk_particle(
        [2.0, 0.0, 0.0], _toroidal_field, _cylinder(1.9995),
        parallel_step=0.2, diffusion_coefficient=0.0, max_steps=63, seed=7,
        parallel_substeps=4,
    )
    assert result.termination is ParticleTermination.MAX_STEPS


@pytest.mark.xfail(
    strict=True,
    reason="Two finite chords can still cross a near-tangent inner wall",
)
def test_two_substeps_do_not_certify_collision_safety_at_half_mm():
    result = walk_particle(
        [2.0, 0.0, 0.0], _toroidal_field, _cylinder(1.9995),
        parallel_step=0.2, diffusion_coefficient=0.0, max_steps=63, seed=7,
        parallel_substeps=2,
    )
    assert result.termination is ParticleTermination.MAX_STEPS


@pytest.mark.parametrize("substeps", [1, 2, 4])
def test_genuine_radial_wall_hit_retains_identity(substeps):
    result = walk_particle(
        [2.0, 0.0, 0.0], lambda _p: [1.0, 0.0, 0.0],
        _cylinder(2.5), parallel_step=1.0, diffusion_coefficient=0.0,
        max_steps=2, seed=1, parallel_substeps=substeps,
    )
    assert result.termination is ParticleTermination.SURFACE_HIT
    assert result.iterations == 1
    assert result.hit.component_id == "wall"
    np.testing.assert_allclose(result.hit.point, [2.5, 0.0, 0.0])


def test_diffusion_occurs_once_per_full_parallel_step():
    step = 0.2
    diffusion = 0.08
    result = walk_particle(
        [0.0, 0.0, 0.0], lambda _p: [0.0, 0.0, 1.0], _no_hit,
        parallel_step=step, diffusion_coefficient=diffusion,
        max_steps=1, seed=99, parallel_substeps=4,
    )
    assert result.termination is ParticleTermination.MAX_STEPS
    assert result.positions.shape == (6, 3)
    np.testing.assert_allclose(
        np.linalg.norm(result.positions[-1] - result.positions[-2]),
        np.sqrt(diffusion * step), rtol=1e-12,
    )
    assert result.positions[-1, 2] == pytest.approx(step)


def test_seeded_diffusion_replays_with_substeps():
    opts = dict(parallel_step=0.1, diffusion_coefficient=0.2,
                max_steps=10, seed=5, parallel_substeps=4)
    p = [1.0, 0.0, 0.0]
    first = walk_particle(p, lambda _p: [0.0, 0.0, 1.0], _no_hit, **opts)
    second = walk_particle(p, lambda _p: [0.0, 0.0, 1.0], _no_hit, **opts)
    np.testing.assert_array_equal(first.positions, second.positions)


def test_invalid_substep_field_preserves_partial_path():
    def field(p):
        return [1.0, 0.0, 0.0] if p[0] < 0.75 else [0.0, 0.0, 0.0]

    result = walk_particle(
        [0.0, 0.0, 0.0], field, _no_hit,
        parallel_step=2.0, diffusion_coefficient=0.0,
        max_steps=2, seed=1, parallel_substeps=2,
    )
    assert result.termination is ParticleTermination.INVALID_FIELD
    assert result.iterations == 1
    np.testing.assert_array_equal(result.positions, [[0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("invalid", [0, -1, 1.5, None, True])
def test_invalid_substep_count_is_rejected(invalid):
    with pytest.raises(ValueError, match="parallel_substeps"):
        walk_particle(
            [0, 0, 0], lambda _p: [0, 0, 1], _no_hit,
            parallel_step=1.0, diffusion_coefficient=0.0,
            max_steps=1, seed=1, parallel_substeps=invalid,
        )


def test_substeps_find_wall_before_magnetic_field_becomes_undefined():
    def field(position):
        if position[2] < 0.4:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]

    def wall(a, b):
        return axisymmetric_surface_hit(
            a, b, [[0.0, 0.4], [2.0, 0.4]], component_ids=["wall"]
        )

    coarse = walk_particle(
        [1.0, 0.0, 0.0], field, wall,
        parallel_step=1.0, diffusion_coefficient=0.0,
        max_steps=2, seed=6,
    )
    refined = walk_particle(
        [1.0, 0.0, 0.0], field, wall,
        parallel_step=1.0, diffusion_coefficient=0.0,
        max_steps=2, seed=6, parallel_substeps=2,
    )

    # The full-step midpoint lies beyond the wall, where this field is invalid.
    assert coarse.termination is ParticleTermination.INVALID_FIELD
    assert refined.termination is ParticleTermination.SURFACE_HIT
    assert refined.hit.component_id == "wall"
    np.testing.assert_allclose(refined.hit.point, [1.0, 0.0, 0.4])
