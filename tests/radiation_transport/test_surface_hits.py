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
