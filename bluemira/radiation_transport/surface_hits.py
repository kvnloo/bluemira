# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Surface-hit primitives for particle transport."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


ComponentId = str | int | None


@dataclass(frozen=True)
class SurfaceHit:
    """A particle-step intersection with a transport surface."""

    point: npt.NDArray[np.float64]
    step_fraction: float
    distance: float
    surface_index: int
    component_id: ComponentId = None


def _component_id(
    component_ids: Sequence[ComponentId] | None, surface_index: int
) -> ComponentId:
    if component_ids is None:
        return None
    if len(component_ids) <= surface_index:
        raise ValueError("component_ids must cover every candidate surface")
    return component_ids[surface_index]


def _surface_hit(
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    step_fraction: float,
    surface_index: int,
    component_ids: Sequence[ComponentId] | None,
) -> SurfaceHit:
    delta = end - start
    point = start + step_fraction * delta
    return SurfaceHit(
        point=point,
        step_fraction=step_fraction,
        distance=float(np.linalg.norm(delta) * step_fraction),
        surface_index=surface_index,
        component_id=_component_id(component_ids, surface_index),
    )


def _quadratic_roots(a: float, b: float, c: float, tolerance: float) -> list[float]:
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            return []
        return [-c / b]

    discriminant = b * b - 4.0 * a * c
    if discriminant < -tolerance:
        return []

    root = np.sqrt(max(discriminant, 0.0))
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


def axisymmetric_surface_hit(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    wall_rz: npt.ArrayLike,
    *,
    component_ids: Sequence[ComponentId] | None = None,
    tolerance: float = 1e-12,
) -> SurfaceHit | None:
    """
    Find the nearest hit of a 3-D segment on an axisymmetric R-Z wall.

    Each adjacent pair in wall_rz defines a line segment in the R-Z plane.
    Revolving that segment about the z-axis gives the corresponding 3-D surface.
    Horizontal wall segments are treated as annuli; all other segments are
    treated as conical frusta, including cylindrical segments as a special case.

    Parameters
    ----------
    start:
        Start of the particle step, (x, y, z).
    end:
        End of the particle step, (x, y, z).
    wall_rz:
        Ordered wall points with shape (n, 2) storing (R, Z).
    component_ids:
        Optional component identifier for each wall segment.
    tolerance:
        Floating-point tolerance for degenerate geometry and interval checks.

    Returns
    -------
    :
        The nearest hit along the particle step, or None when there is no hit.

    Notes
    -----
    A particle segment lying entirely in the plane of a horizontal annulus is
    intentionally treated as ambiguous and skipped.
    """
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    wall = np.asarray(wall_rz, dtype=float)

    if start_array.shape != (3,) or end_array.shape != (3,):
        raise ValueError("start and end must be 3-vectors")
    if wall.ndim != 2 or wall.shape[1] != 2 or len(wall) < 2:
        raise ValueError("wall_rz must have shape (n, 2) with n >= 2")

    delta = end_array - start_array
    candidates: list[tuple[float, int]] = []

    for index, (wall_start, wall_end) in enumerate(zip(wall[:-1], wall[1:])):
        radius_0, z_0 = wall_start
        radius_1, z_1 = wall_end
        delta_radius = radius_1 - radius_0
        delta_z_wall = z_1 - z_0

        if abs(delta_z_wall) <= tolerance:
            if abs(delta[2]) <= tolerance:
                continue

            fraction = (z_0 - start_array[2]) / delta[2]
            if not -tolerance <= fraction <= 1.0 + tolerance:
                continue

            point = start_array + fraction * delta
            radius = np.hypot(point[0], point[1])
            radius_min, radius_max = sorted((radius_0, radius_1))
            if radius_min - tolerance <= radius <= radius_max + tolerance:
                candidates.append((float(np.clip(fraction, 0.0, 1.0)), index))
            continue

        slope = delta_radius / delta_z_wall
        target_radius_0 = radius_0 + slope * (start_array[2] - z_0)
        target_radius_delta = slope * delta[2]

        qa = delta[0] ** 2 + delta[1] ** 2 - target_radius_delta**2
        qb = 2.0 * (
            start_array[0] * delta[0]
            + start_array[1] * delta[1]
            - target_radius_0 * target_radius_delta
        )
        qc = start_array[0] ** 2 + start_array[1] ** 2 - target_radius_0**2

        for fraction in _quadratic_roots(qa, qb, qc, tolerance):
            if not -tolerance <= fraction <= 1.0 + tolerance:
                continue

            fraction = float(np.clip(fraction, 0.0, 1.0))
            z_value = start_array[2] + fraction * delta[2]
            wall_fraction = (z_value - z_0) / delta_z_wall
            if not -tolerance <= wall_fraction <= 1.0 + tolerance:
                continue

            target_radius = radius_0 + wall_fraction * delta_radius
            if target_radius < -tolerance:
                continue
            candidates.append((fraction, index))

    if not candidates:
        return None

    fraction, surface_index = min(candidates, key=lambda candidate: candidate[0])
    return _surface_hit(
        start_array,
        end_array,
        fraction,
        surface_index,
        component_ids,
    )


def triangle_surface_hit(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    vertices: npt.ArrayLike,
    triangles: npt.ArrayLike,
    *,
    component_ids: Sequence[ComponentId] | None = None,
    tolerance: float = 1e-12,
) -> SurfaceHit | None:
    """
    Find the nearest hit of a 3-D segment on a triangle mesh.

    The intersection uses the Moller-Trumbore algorithm and returns the nearest
    triangle hit along the finite particle step.

    Parameters
    ----------
    start:
        Start of the particle step, (x, y, z).
    end:
        End of the particle step, (x, y, z).
    vertices:
        Mesh vertices with shape (n, 3).
    triangles:
        Triangle vertex indices with shape (m, 3).
    component_ids:
        Optional component identifier for each triangle.
    tolerance:
        Floating-point tolerance for parallel and interval checks.

    Returns
    -------
    :
        The nearest hit along the particle step, or None when there is no hit.
    """
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    vertex_array = np.asarray(vertices, dtype=float)
    triangle_array = np.asarray(triangles, dtype=int)

    if start_array.shape != (3,) or end_array.shape != (3,):
        raise ValueError("start and end must be 3-vectors")
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")

    direction = end_array - start_array
    candidates: list[tuple[float, int]] = []

    for index, triangle in enumerate(triangle_array):
        vertex_0, vertex_1, vertex_2 = vertex_array[triangle]
        edge_1 = vertex_1 - vertex_0
        edge_2 = vertex_2 - vertex_0

        p_vector = np.cross(direction, edge_2)
        determinant = float(np.dot(edge_1, p_vector))
        if abs(determinant) <= tolerance:
            continue

        inverse_determinant = 1.0 / determinant
        t_vector = start_array - vertex_0
        barycentric_u = float(np.dot(t_vector, p_vector) * inverse_determinant)
        if barycentric_u < -tolerance or barycentric_u > 1.0 + tolerance:
            continue

        q_vector = np.cross(t_vector, edge_1)
        barycentric_v = float(np.dot(direction, q_vector) * inverse_determinant)
        if (
            barycentric_v < -tolerance
            or barycentric_u + barycentric_v > 1.0 + tolerance
        ):
            continue

        fraction = float(np.dot(edge_2, q_vector) * inverse_determinant)
        if -tolerance <= fraction <= 1.0 + tolerance:
            candidates.append((float(np.clip(fraction, 0.0, 1.0)), index))

    if not candidates:
        return None

    fraction, surface_index = min(candidates, key=lambda candidate: candidate[0])
    return _surface_hit(
        start_array,
        end_array,
        fraction,
        surface_index,
        component_ids,
    )
