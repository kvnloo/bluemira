# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from tests.radiation_transport.axisymmetric_ab import (
    axisymmetric_cartesian_field,
    legacy_integrated_power,
    legacy_midplane_residual_power,
    legacy_weighted_trace_pairs,
    load_single_null_ab_fixture,
    representative_trace_pairs,
)


class _ConstantCylindricalField:
    @staticmethod
    def Bx(_radius, _z):
        return 2.0

    @staticmethod
    def Bz(_radius, _z):
        return 3.0

    @staticmethod
    def Bt(_radius):
        return 4.0


def test_axisymmetric_field_conversion_at_zero_toroidal_angle():
    field = axisymmetric_cartesian_field(
        _ConstantCylindricalField(), [2.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(field, [2.0, 4.0, 3.0])


def test_axisymmetric_field_conversion_rotates_with_toroidal_angle():
    field = axisymmetric_cartesian_field(
        _ConstantCylindricalField(), [0.0, 2.0, 1.0]
    )
    np.testing.assert_allclose(field, [-4.0, 2.0, 3.0])


@pytest.fixture(scope="module")
def sn_fixture():
    return load_single_null_ab_fixture()


def test_fixture_reproduces_existing_legacy_power_regression(sn_fixture):
    assert legacy_integrated_power(sn_fixture) == pytest.approx(100.0, rel=2e-2)


def test_legacy_weighted_traces_plus_midplane_residual_close_power(sn_fixture):
    pairs = legacy_weighted_trace_pairs(sn_fixture.legacy_solver)
    resolved = sum(pair.power for pair in pairs)
    residual = legacy_midplane_residual_power(sn_fixture.legacy_solver)

    assert resolved + residual == pytest.approx(
        sn_fixture.legacy_solver.params.P_sep_particle
    )


def test_legacy_resolved_target_split_matches_imposed_fractions(sn_fixture):
    pairs = legacy_weighted_trace_pairs(sn_fixture.legacy_solver)
    lfs = sum(pair.power for pair in pairs if pair.branch == "lfs_lower")
    hfs = sum(pair.power for pair in pairs if pair.branch == "hfs_lower")
    resolved = lfs + hfs

    assert lfs / resolved == pytest.approx(0.75)
    assert hfs / resolved == pytest.approx(0.25)


def test_ci_representative_trace_subset_covers_each_branch(sn_fixture):
    pairs = legacy_weighted_trace_pairs(sn_fixture.legacy_solver)
    selected = representative_trace_pairs(pairs)

    assert len(selected) == 6
    assert sum(pair.branch == "lfs_lower" for pair in selected) == 3
    assert sum(pair.branch == "hfs_lower" for pair in selected) == 3
