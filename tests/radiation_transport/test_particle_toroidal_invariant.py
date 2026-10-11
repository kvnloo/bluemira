# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Pending physical-invariant regressions for the experimental walker.

For a purely toroidal magnetic field, zero-diffusion field lines have
constant cylindrical radius: finite-step integration must not deposit
energy onto an enclosing cylindrical wall solely from integration drift.
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


@pytest.mark.xfail(
    strict=True,
    reason="Experimental Euler parallel step drifts off circular field lines",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="Experimental Euler drift causes a false cylinder collision",
)
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
