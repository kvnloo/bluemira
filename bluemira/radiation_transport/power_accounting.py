# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Power-accounting primitives for Monte Carlo heat transport."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bluemira.radiation_transport.particle_transport import (
    ParticleTermination,
    ParticleWalkResult,
)


ComponentId = str | int


@dataclass(frozen=True)
class PowerBalance:
    """Power accounting derived from equally weighted particle outcomes."""

    launched_power: float
    particle_count: int
    deposited_counts: dict[ComponentId, int]
    unresolved_particles: int

    @property
    def particle_power(self) -> float:
        """Power represented by one macroparticle."""
        return self.launched_power / self.particle_count

    @property
    def deposited_by_component(self) -> dict[ComponentId, float]:
        """Deposited power grouped by component identifier."""
        return {
            component_id: count * self.particle_power
            for component_id, count in self.deposited_counts.items()
        }

    @property
    def deposited_power(self) -> float:
        """Total power deposited on attributed components."""
        return sum(self.deposited_by_component.values())

    @property
    def unresolved_power(self) -> float:
        """Power carried by particles without an attributed surface hit."""
        return self.unresolved_particles * self.particle_power

    @property
    def accounted_power(self) -> float:
        """Total deposited plus unresolved power."""
        return self.deposited_power + self.unresolved_power

    @property
    def balance_error(self) -> float:
        """Numerical residual between launched and accounted power."""
        return self.launched_power - self.accounted_power


def account_particle_power(
    results: Sequence[ParticleWalkResult], total_power: float
) -> PowerBalance:
    """
    Account equally weighted particle power by hit component.

    A particle that reaches a component-attributed surface deposits its full
    macroparticle power on that component. All other outcomes remain unresolved.
    Classification uses integer particle counts before converting to power.

    Parameters
    ----------
    results:
        Completed particle-walk results.
    total_power:
        Total power represented by the particle population [W].

    Returns
    -------
    :
        Component deposition counts and unresolved population with power views.

    Raises
    ------
    ValueError
        If no particles are supplied, power is invalid, or a walk result has
        inconsistent termination and hit state.
    """
    if not results:
        raise ValueError("results must contain at least one particle")
    if not np.isfinite(total_power) or total_power < 0.0:
        raise ValueError("total_power must be finite and non-negative")

    deposited_counts: dict[ComponentId, int] = {}
    unresolved_particles = 0

    for result in results:
        is_hit = result.termination is ParticleTermination.SURFACE_HIT
        if is_hit and result.hit is None:
            raise ValueError("SURFACE_HIT termination requires a hit")
        if not is_hit and result.hit is not None:
            raise ValueError("non-hit termination cannot carry a hit")

        if is_hit and result.hit is not None and result.hit.component_id is not None:
            component_id = result.hit.component_id
            deposited_counts[component_id] = deposited_counts.get(component_id, 0) + 1
        else:
            unresolved_particles += 1

    particle_count = len(results)
    return PowerBalance(
        launched_power=float(total_power),
        particle_count=particle_count,
        deposited_counts=deposited_counts,
        unresolved_particles=unresolved_particles,
    )
