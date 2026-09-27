# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Deterministic particle-walk primitives for Monte Carlo heat transport."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from bluemira.radiation_transport.surface_hits import SurfaceHit


_XYZ_DIMENSIONS = 3
_XYZ_VECTOR_SHAPE = (_XYZ_DIMENSIONS,)
_AXIS_ALIGNMENT_LIMIT = 0.9
_TWO_PI = 2.0 * np.pi


MagneticField = Callable[[npt.NDArray[np.float64]], npt.ArrayLike]
SurfaceIntersector = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], SurfaceHit | None
]


class ParticleTermination(Enum):
    """Reason a bounded particle walk stopped."""

    SURFACE_HIT = auto()
    MAX_STEPS = auto()
    INVALID_FIELD = auto()


@dataclass(frozen=True)
class ParticleWalkResult:
    """Result of a bounded, seeded particle walk."""

    positions: npt.NDArray[np.float64]
    termination: ParticleTermination
    iterations: int
    hit: SurfaceHit | None = None


def _perpendicular_basis(
    direction: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(direction[2]) < _AXIS_ALIGNMENT_LIMIT
        else np.array([1.0, 0.0, 0.0])
    )
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def _normalised_field(
    magnetic_field: MagneticField,
    position: npt.NDArray[np.float64],
    tolerance: float,
) -> npt.NDArray[np.float64] | None:
    field = np.asarray(magnetic_field(position), dtype=float)
    if field.shape != _XYZ_VECTOR_SHAPE or not np.all(np.isfinite(field)):
        return None

    magnitude = float(np.linalg.norm(field))
    if magnitude <= tolerance:
        return None
    return field / magnitude


def _hit_result(
    positions: list[npt.NDArray[np.float64]],
    hit: SurfaceHit,
    iterations: int,
) -> ParticleWalkResult:
    positions.append(hit.point.copy())
    return ParticleWalkResult(
        positions=np.asarray(positions),
        termination=ParticleTermination.SURFACE_HIT,
        iterations=iterations,
        hit=hit,
    )


def walk_particle(
    start: npt.ArrayLike,
    magnetic_field: MagneticField,
    surface_intersector: SurfaceIntersector,
    *,
    parallel_step: float,
    diffusion_coefficient: float,
    max_steps: int,
    seed: int,
    forward: bool = True,
    field_tolerance: float = 1e-14,
) -> ParticleWalkResult:
    """
    Walk one particle along a magnetic field with perpendicular diffusion.

    Each iteration advances by ``parallel_step`` along the local magnetic
    field, then by ``sqrt(diffusion_coefficient * parallel_step)`` in a
    uniformly sampled direction in the plane perpendicular to the field.
    Both finite segments are checked independently for the nearest surface hit.

    Parameters
    ----------
    start:
        Initial Cartesian position ``(x, y, z)``.
    magnetic_field:
        Callable returning the Cartesian magnetic-field vector at a position.
    surface_intersector:
        Callable returning the nearest surface hit along a finite segment.
    parallel_step:
        Parallel step length [m].
    diffusion_coefficient:
        Field-line diffusion coefficient ``Dm`` [m^2/m].
    max_steps:
        Maximum number of parallel-plus-diffusive iterations.
    seed:
        Seed for the local NumPy random generator.
    forward:
        Whether to walk along ``+B`` rather than ``-B``.
    field_tolerance:
        Field magnitude at or below which the walk terminates as invalid.

    Returns
    -------
    :
        Seeded trajectory, termination reason, iteration count, and optional hit.

    Raises
    ------
    ValueError
        If the initial position or scalar walk parameters are invalid.

    Notes
    -----
    The random generator is local to this call. Identical inputs and seed
    therefore replay the same diffusive trajectory without consuming global
    random state.
    """
    position = np.asarray(start, dtype=float)
    if position.shape != _XYZ_VECTOR_SHAPE or not np.all(np.isfinite(position)):
        raise ValueError("start must be a finite 3-vector")
    if parallel_step <= 0.0:
        raise ValueError("parallel_step must be positive")
    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if field_tolerance < 0.0:
        raise ValueError("field_tolerance must be non-negative")

    rng = np.random.default_rng(seed)
    positions = [position.copy()]
    sign = 1.0 if forward else -1.0
    diffusion_step = np.sqrt(diffusion_coefficient * parallel_step)

    for step_index in range(max_steps):
        direction = _normalised_field(magnetic_field, position, field_tolerance)
        iterations = step_index + 1
        if direction is None:
            return ParticleWalkResult(
                positions=np.asarray(positions),
                termination=ParticleTermination.INVALID_FIELD,
                iterations=iterations,
            )

        parallel_end = position + sign * parallel_step * direction
        if hit := surface_intersector(position, parallel_end):
            return _hit_result(positions, hit, iterations)

        positions.append(parallel_end.copy())
        position = parallel_end

        if diffusion_step == 0.0:
            continue

        diffusion_normal = _normalised_field(
            magnetic_field, position, field_tolerance
        )
        if diffusion_normal is None:
            return ParticleWalkResult(
                positions=np.asarray(positions),
                termination=ParticleTermination.INVALID_FIELD,
                iterations=iterations,
            )

        first, second = _perpendicular_basis(diffusion_normal)
        angle = rng.uniform(0.0, _TWO_PI)
        diffusion_direction = np.cos(angle) * first + np.sin(angle) * second
        diffusion_end = position + diffusion_step * diffusion_direction

        if hit := surface_intersector(position, diffusion_end):
            return _hit_result(positions, hit, iterations)

        positions.append(diffusion_end.copy())
        position = diffusion_end

    return ParticleWalkResult(
        positions=np.asarray(positions),
        termination=ParticleTermination.MAX_STEPS,
        iterations=max_steps,
    )
