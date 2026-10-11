# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.radiation_transport.surface_hits import (
    axisymmetric_surface_hit,
    triangle_surface_hit,
)


def test_axisymmetric_cylinder_hit_preserves_component_identity():
    hit = axisymmetric_surface_hit(
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [[2.0, -1.0], [2.0, 1.0]],
        component_ids=["first_wall"],
    )

    assert hit is not None
    assert hit.step_fraction == pytest.approx(0.5)
    assert hit.distance == pytest.approx(1.0)
    assert hit.surface_index == 0
    assert hit.component_id == "first_wall"
    np.testing.assert_allclose(hit.point, [2.0, 0.0, 0.0])


def test_axisymmetric_conical_hit_is_exact_for_a_3d_segment():
    hit = axisymmetric_surface_hit(
        [1.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [[2.0, -1.0], [3.0, 1.0]],
    )

    assert hit is not None
    assert hit.step_fraction == pytest.approx(0.4)
    np.testing.assert_allclose(hit.point, [2.5, 0.0, 0.0])


def test_axisymmetric_horizontal_segment_is_an_annulus():
    hit = axisymmetric_surface_hit(
        [2.0, 0.0, -1.0],
        [2.0, 0.0, 1.0],
        [[1.0, 0.0], [3.0, 0.0]],
    )

    assert hit is not None
    assert hit.step_fraction == pytest.approx(0.5)
    np.testing.assert_allclose(hit.point, [2.0, 0.0, 0.0])


def test_triangle_hit_uses_same_contract_as_axisymmetric_hit():
    vertices = np.array([
        [2.0, -1.0, -1.0],
        [2.0, 1.0, -1.0],
        [2.0, 0.0, 1.0],
    ])
    hit = triangle_surface_hit(
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        vertices,
        [[0, 1, 2]],
        component_ids=["first_wall"],
    )

    assert hit is not None
    assert hit.step_fraction == pytest.approx(0.5)
    assert hit.distance == pytest.approx(1.0)
    assert hit.surface_index == 0
    assert hit.component_id == "first_wall"
    np.testing.assert_allclose(hit.point, [2.0, 0.0, 0.0])


def test_triangle_hit_selects_nearest_surface():
    vertices = np.array([
        [2.5, -1.0, -1.0],
        [2.5, 1.0, -1.0],
        [2.5, 0.0, 1.0],
        [2.0, -1.0, -1.0],
        [2.0, 1.0, -1.0],
        [2.0, 0.0, 1.0],
    ])
    hit = triangle_surface_hit(
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        vertices,
        [[0, 1, 2], [3, 4, 5]],
        component_ids=["far", "near"],
    )

    assert hit is not None
    assert hit.surface_index == 1
    assert hit.component_id == "near"
    assert hit.step_fraction == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (
            axisymmetric_surface_hit,
            (
                [1.0, 0.0, 2.0],
                [3.0, 0.0, 2.0],
                [[2.0, -1.0], [2.0, 1.0]],
            ),
        ),
        (
            triangle_surface_hit,
            (
                [1.0, 0.0, 2.0],
                [3.0, 0.0, 2.0],
                [[2.0, -1.0, -1.0], [2.0, 1.0, -1.0], [2.0, 0.0, 1.0]],
                [[0, 1, 2]],
            ),
        ),
    ],
)
def test_surface_hit_returns_none_for_miss(function, args):
    assert function(*args) is None


def test_component_ids_must_cover_candidate_surface():
    with pytest.raises(
        ValueError, match="component_ids must match the number of candidate surfaces"
    ):
        triangle_surface_hit(
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [
                [2.0, -1.0, -1.0],
                [2.0, 1.0, -1.0],
                [2.0, 0.0, 1.0],
            ],
            [[0, 1, 2]],
            component_ids=[],
        )


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.01, 0.001, 0.0001])
def test_triangle_hit_is_invariant_under_uniform_scale(scale):
    vertices = scale * np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    )
    start = scale * np.array([0.02, 0.02, -0.05])
    end = scale * np.array([0.02, 0.02, 0.05])

    hit = triangle_surface_hit(start, end, vertices, [[0, 1, 2]])

    assert hit is not None
    assert hit.step_fraction == pytest.approx(0.5)
    np.testing.assert_allclose(
        hit.point, scale * np.array([0.02, 0.02, 0.0]), rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.01, 0.001, 0.0001])
def test_triangle_outside_ray_remains_a_miss_under_uniform_scale(scale):
    vertices = scale * np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    )
    start = scale * np.array([0.08, 0.08, -0.05])
    end = scale * np.array([0.08, 0.08, 0.05])

    assert triangle_surface_hit(start, end, vertices, [[0, 1, 2]]) is None


@pytest.mark.parametrize("panels", [16, 32, 64, 128])
@pytest.mark.parametrize("scale", [1.0, 0.0001])
def test_triangle_cylinder_facets_converge_to_axisymmetric_hit(panels, scale):
    # An inscribed polygon converges to a circular cylinder as panels increase.
    angles = 2.0 * np.pi * np.arange(panels) / panels
    vertices = []
    triangles = []
    for angle in angles:
        x, y = 2.0 * scale * np.cos(angle), 2.0 * scale * np.sin(angle)
        vertices.extend([[x, y, -scale], [x, y, scale]])
    for panel in range(panels):
        left = 2 * panel
        right = 2 * ((panel + 1) % panels)
        triangles.extend([[left, right, left + 1], [right, right + 1, left + 1]])

    midpoint = np.pi / panels
    direction = np.array([np.cos(midpoint), np.sin(midpoint), 0.0])
    offset = np.array([0.0, 0.0, -0.2 * scale])
    start, end = scale * direction + offset, 3.0 * scale * direction + offset
    axis_hit = axisymmetric_surface_hit(
        start, end, [[2.0 * scale, -scale], [2.0 * scale, scale]]
    )
    mesh_hit = triangle_surface_hit(start, end, vertices, triangles)

    assert axis_hit is not None
    assert mesh_hit is not None
    assert axis_hit.step_fraction == pytest.approx(0.5)
    assert mesh_hit.step_fraction == pytest.approx(
        np.cos(np.pi / panels) - 0.5, abs=1e-12
    )
    assert axis_hit.step_fraction - mesh_hit.step_fraction == pytest.approx(
        1.0 - np.cos(np.pi / panels), abs=1e-12
    )
