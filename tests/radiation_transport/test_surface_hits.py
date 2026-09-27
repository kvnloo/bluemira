# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.radiation_transport.surface_hits import (
    SurfaceTerminationReason,
    first_surface_hit,
)


@pytest.fixture
def unit_triangle():
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


@pytest.mark.parametrize(
    "xy",
    [
        (0.25, 0.25),
        (0.5, 0.0),
        (0.0, 0.0),
    ],
)
def test_segment_hits_triangle_interior_edge_and_vertex(unit_triangle, xy):
    vertices = unit_triangle
    triangles = np.array([[0, 1, 2]])

    result = first_surface_hit(
        [xy[0], xy[1], 1.0],
        [xy[0], xy[1], -1.0],
        vertices,
        triangles,
        surface_ids=[7],
    )

    assert result.termination is SurfaceTerminationReason.SURFACE_HIT
    assert result.point == pytest.approx((xy[0], xy[1], 0.0))
    assert result.distance == pytest.approx(1.0)
    assert result.fraction == pytest.approx(0.5)
    assert result.surface_id == 7
    assert result.triangle_index == 0


def test_segment_miss_is_explicit(unit_triangle):
    result = first_surface_hit(
        [1.1, 1.1, 1.0],
        [1.1, 1.1, -1.0],
        unit_triangle,
        np.array([[0, 1, 2]]),
    )

    assert result.termination is SurfaceTerminationReason.NO_SURFACE_HIT
    assert result.point is None
    assert result.distance is None
    assert result.fraction is None
    assert result.surface_id is None
    assert result.triangle_index is None


def test_nearest_intersection_wins():
    vertices = np.array([
        [0.0, 0.0, 0.25],
        [1.0, 0.0, 0.25],
        [0.0, 1.0, 0.25],
        [0.0, 0.0, -0.25],
        [1.0, 0.0, -0.25],
        [0.0, 1.0, -0.25],
    ])
    triangles = np.array([
        [3, 4, 5],
        [0, 1, 2],
    ])

    result = first_surface_hit(
        [0.25, 0.25, 1.0],
        [0.25, 0.25, -1.0],
        vertices,
        triangles,
        surface_ids=[20, 10],
    )

    assert result.termination is SurfaceTerminationReason.SURFACE_HIT
    assert result.point == pytest.approx((0.25, 0.25, 0.25))
    assert result.surface_id == 10
    assert result.triangle_index == 1
    assert result.fraction == pytest.approx(0.375)


def test_equal_distance_tie_uses_lowest_triangle_index(unit_triangle):
    vertices = np.vstack([unit_triangle, unit_triangle])
    triangles = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ])

    result = first_surface_hit(
        [0.25, 0.25, 1.0],
        [0.25, 0.25, -1.0],
        vertices,
        triangles,
        surface_ids=[100, 200],
    )

    assert result.termination is SurfaceTerminationReason.SURFACE_HIT
    assert result.triangle_index == 0
    assert result.surface_id == 100


def test_coplanar_motion_is_an_explicit_miss(unit_triangle):
    result = first_surface_hit(
        [0.1, 0.1, 0.0],
        [0.7, 0.1, 0.0],
        unit_triangle,
        np.array([[0, 1, 2]]),
    )

    assert result.termination is SurfaceTerminationReason.NO_SURFACE_HIT


def test_short_segment_is_an_explicit_miss(unit_triangle):
    result = first_surface_hit(
        [0.25, 0.25, 1.0],
        [0.25, 0.25, 1.0 + 1e-7],
        unit_triangle,
        np.array([[0, 1, 2]]),
    )

    assert result.termination is SurfaceTerminationReason.NO_SURFACE_HIT
