# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Small deterministic surface-intersection primitives for particle transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bluemira.geometry.constants import D_TOLERANCE

if TYPE_CHECKING:
    from collections.abc import Sequence


class SurfaceTerminationReason(Enum):
    """Reason a segment-to-surface query terminated."""

    SURFACE_HIT = "surface_hit"
    NO_SURFACE_HIT = "no_surface_hit"


@dataclass(frozen=True)
class SurfaceHitResult:
    """Result of tracing one finite segment against a triangle surface."""

    termination: SurfaceTerminationReason
    point: tuple[float, float, float] | None = None
    distance: float | None = None
    fraction: float | None = None
    surface_id: int | None = None
    triangle_index: int | None = None

    @classmethod
    def miss(cls) -> SurfaceHitResult:
        """
        Construct an explicit no-hit result.

        Returns
        -------
        :
            A typed result representing no surface intersection.
        """
        return cls(termination=SurfaceTerminationReason.NO_SURFACE_HIT)


def _segment_triangle_distance(
    start: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    segment_length: float,
    triangle: npt.NDArray[np.float64],
) -> float | None:
    """
    Return ray distance to a triangle when the hit lies on the finite segment.

    Returns
    -------
    :
        Distance from segment start, or None when the segment does not hit.
    """
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    p_vec = np.cross(direction, edge_2)
    determinant = float(np.dot(edge_1, p_vec))

    determinant_scale = max(float(np.linalg.norm(edge_1) * np.linalg.norm(edge_2)), 1.0)
    determinant_tolerance = 64 * np.finfo(float).eps * determinant_scale
    if abs(determinant) <= determinant_tolerance:
        return None

    inverse_determinant = 1.0 / determinant
    t_vec = start - triangle[0]
    barycentric_u = float(np.dot(t_vec, p_vec) * inverse_determinant)

    barycentric_tolerance = 64 * np.finfo(float).eps
    if barycentric_u < -barycentric_tolerance or barycentric_u > 1 + barycentric_tolerance:
        return None

    q_vec = np.cross(t_vec, edge_1)
    barycentric_v = float(np.dot(direction, q_vec) * inverse_determinant)
    if (
        barycentric_v < -barycentric_tolerance
        or barycentric_u + barycentric_v > 1 + barycentric_tolerance
    ):
        return None

    distance = float(np.dot(edge_2, q_vec) * inverse_determinant)
    if distance < -D_TOLERANCE or distance > segment_length + D_TOLERANCE:
        return None
    return float(np.clip(distance, 0.0, segment_length))


def first_surface_hit(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    vertices: npt.ArrayLike,
    triangles: npt.ArrayLike,
    surface_ids: Sequence[int] | None = None,
) -> SurfaceHitResult:
    """
    Return the nearest triangle hit along a finite segment.

    Parameters
    ----------
    start:
        Segment start point with shape (3,).
    end:
        Segment end point with shape (3,).
    vertices:
        Surface vertices with shape (N, 3).
    triangles:
        Triangle vertex indices with shape (M, 3).
    surface_ids:
        Optional surface/component identity for each triangle.

    Returns
    -------
    :
        Nearest hit, or an explicit no-hit result.

    Notes
    -----
    Equal-distance hits within the geometry distance tolerance are resolved by the
    lowest triangle index. Coplanar segment motion is intentionally treated as no hit;
    a later transport layer can choose how to handle that ambiguous contact mode.
    """
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    vertex_array = np.asarray(vertices, dtype=float)
    triangle_array = np.asarray(triangles, dtype=int)

    if start_array.shape != (3,) or end_array.shape != (3,):
        raise ValueError("start and end must each have shape (3,)")
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    if surface_ids is not None and len(surface_ids) != len(triangle_array):
        raise ValueError("surface_ids must contain one identity per triangle")

    segment = end_array - start_array
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= D_TOLERANCE:
        return SurfaceHitResult.miss()
    direction = segment / segment_length

    best_hit: SurfaceHitResult | None = None
    for triangle_index, indices in enumerate(triangle_array):
        triangle = vertex_array[indices]
        distance = _segment_triangle_distance(
            start_array,
            direction,
            segment_length,
            triangle,
        )
        if distance is None:
            continue

        if best_hit is not None and best_hit.distance is not None:
            if distance >= best_hit.distance - D_TOLERANCE:
                continue

        point = start_array + distance * direction
        best_hit = SurfaceHitResult(
            termination=SurfaceTerminationReason.SURFACE_HIT,
            point=tuple(float(value) for value in point),
            distance=distance,
            fraction=distance / segment_length,
            surface_id=None if surface_ids is None else surface_ids[triangle_index],
            triangle_index=triangle_index,
        )

    return SurfaceHitResult.miss() if best_hit is None else best_hit
