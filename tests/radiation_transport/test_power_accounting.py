# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.radiation_transport.particle_transport import (
    ParticleTermination,
    ParticleWalkResult,
)
from bluemira.radiation_transport.power_accounting import account_particle_power
from bluemira.radiation_transport.surface_hits import SurfaceHit


def _result(termination, component_id=None):
    hit = None
    if termination is ParticleTermination.SURFACE_HIT:
        hit = SurfaceHit(
            point=np.zeros(3),
            step_fraction=0.5,
            distance=1.0,
            surface_index=0,
            component_id=component_id,
        )
    return ParticleWalkResult(
        positions=np.zeros((1, 3)),
        termination=termination,
        iterations=1,
        hit=hit,
    )


def test_equal_particle_weights_conserve_total_power():
    results = [
        _result(ParticleTermination.SURFACE_HIT, "blanket"),
        _result(ParticleTermination.SURFACE_HIT, "blanket"),
        _result(ParticleTermination.SURFACE_HIT, "divertor"),
        _result(ParticleTermination.MAX_STEPS),
        _result(ParticleTermination.INVALID_FIELD),
    ]

    balance = account_particle_power(results, total_power=100.0)

    assert balance.particle_count == 5
    assert balance.particle_power == pytest.approx(20.0)
    assert balance.deposited_counts == {"blanket": 2, "divertor": 1}
    assert balance.deposited_by_component == pytest.approx(
        {"blanket": 40.0, "divertor": 20.0}
    )
    assert balance.deposited_power == pytest.approx(60.0)
    assert balance.unresolved_particles == 2
    assert balance.unresolved_power == pytest.approx(40.0)
    assert balance.accounted_power == pytest.approx(100.0)
    assert balance.balance_error == pytest.approx(0.0)


def test_surface_hit_without_component_remains_unresolved():
    balance = account_particle_power(
        [_result(ParticleTermination.SURFACE_HIT)], total_power=7.5
    )

    assert balance.deposited_counts == {}
    assert balance.deposited_power == 0.0
    assert balance.unresolved_particles == 1
    assert balance.unresolved_power == pytest.approx(7.5)


def test_non_binary_fraction_still_closes_power_balance():
    results = [
        _result(ParticleTermination.SURFACE_HIT, "wall"),
        _result(ParticleTermination.SURFACE_HIT, "wall"),
        _result(ParticleTermination.MAX_STEPS),
        _result(ParticleTermination.MAX_STEPS),
        _result(ParticleTermination.MAX_STEPS),
        _result(ParticleTermination.MAX_STEPS),
        _result(ParticleTermination.INVALID_FIELD),
    ]

    balance = account_particle_power(results, total_power=100.0)

    assert balance.deposited_counts == {"wall": 2}
    assert balance.unresolved_particles == 5
    assert balance.accounted_power == pytest.approx(balance.launched_power)
    assert abs(balance.balance_error) < 1e-12


def test_zero_total_power_is_valid():
    balance = account_particle_power(
        [_result(ParticleTermination.MAX_STEPS)], total_power=0.0
    )

    assert balance.particle_power == 0.0
    assert balance.accounted_power == 0.0


@pytest.mark.parametrize("total_power", [-1.0, np.inf, -np.inf, np.nan])
def test_invalid_total_power_is_rejected(total_power):
    with pytest.raises(ValueError, match="total_power"):
        account_particle_power(
            [_result(ParticleTermination.MAX_STEPS)], total_power=total_power
        )


def test_empty_population_is_rejected():
    with pytest.raises(ValueError, match="at least one particle"):
        account_particle_power([], total_power=1.0)


def test_surface_hit_termination_requires_hit_record():
    inconsistent = ParticleWalkResult(
        positions=np.zeros((1, 3)),
        termination=ParticleTermination.SURFACE_HIT,
        iterations=1,
    )

    with pytest.raises(ValueError, match="requires a hit"):
        account_particle_power([inconsistent], total_power=1.0)


def test_non_hit_termination_rejects_hit_record():
    hit = SurfaceHit(
        point=np.zeros(3),
        step_fraction=0.5,
        distance=1.0,
        surface_index=0,
        component_id="wall",
    )
    inconsistent = ParticleWalkResult(
        positions=np.zeros((1, 3)),
        termination=ParticleTermination.MAX_STEPS,
        iterations=1,
        hit=hit,
    )

    with pytest.raises(ValueError, match="cannot carry a hit"):
        account_particle_power([inconsistent], total_power=1.0)
