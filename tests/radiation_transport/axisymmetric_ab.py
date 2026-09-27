# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Shared fixture/helpers for the #4474 axisymmetric A/B benchmark."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.surface_hits import (
    SurfaceHit,
    axisymmetric_surface_hit,
)


TEST_PATH = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
EQ_PATH = get_bluemira_path("equilibria", subfolder="data")

SN_PARAMS = {
    "P_sep_particle": 100.0,
    "f_p_sol_near": 0.50,
    "fw_lambda_q_near_omp": 0.05,
    "fw_lambda_q_far_omp": 0.05,
    "f_lfs_lower_target": 0.75,
    "f_hfs_lower_target": 0.25,
    "f_lfs_upper_target": 0.0,
    "f_hfs_upper_target": 0.0,
}


@dataclass(frozen=True)
class AxisymmetricABFixture:
    """Single-null physical fixture shared by the legacy and Monte Carlo paths."""

    equilibrium: Equilibrium
    first_wall: Coordinates
    legacy_solver: ChargedParticleSolver
    legacy_x: npt.NDArray[np.float64]
    legacy_z: npt.NDArray[np.float64]
    legacy_heat_flux: npt.NDArray[np.float64]


@dataclass(frozen=True)
class LegacyTracePair:
    """One legacy open-flux-surface start/end pair in the R-Z plane."""

    branch: str
    start_rz: npt.NDArray[np.float64]
    end_rz: npt.NDArray[np.float64]
    power: float | None = None


def load_single_null_ab_fixture() -> AxisymmetricABFixture:
    """Load the existing EU-DEMO single-null charged-particle regression case."""
    equilibrium = Equilibrium.from_eqdsk(
        Path(EQ_PATH, "EU-DEMO_EOF.json"),
        from_cocos=3,
        qpsi_positive=False,
    )
    first_wall = Coordinates.from_json(Path(TEST_PATH, "first_wall.json"))
    solver = ChargedParticleSolver(SN_PARAMS, equilibrium, dx_mp=0.001)
    x, z, heat_flux = solver.analyse(first_wall)
    return AxisymmetricABFixture(
        equilibrium=equilibrium,
        first_wall=solver.first_wall,
        legacy_solver=solver,
        legacy_x=np.asarray(x),
        legacy_z=np.asarray(z),
        legacy_heat_flux=np.asarray(heat_flux),
    )


def axisymmetric_cartesian_field(
    equilibrium: Equilibrium, position: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Convert Bluemira's cylindrical equilibrium field to Cartesian B."""
    point = np.asarray(position, dtype=float)
    x_coord, y_coord, z_coord = point
    radius = float(np.hypot(x_coord, y_coord))
    if radius == 0.0:
        return np.full(3, np.nan)

    radial = float(equilibrium.Bx(radius, z_coord))
    vertical = float(equilibrium.Bz(radius, z_coord))
    toroidal = float(equilibrium.Bt(radius))
    cos_phi = x_coord / radius
    sin_phi = y_coord / radius

    return np.array([
        radial * cos_phi - toroidal * sin_phi,
        radial * sin_phi + toroidal * cos_phi,
        vertical,
    ])


def make_axisymmetric_intersector(first_wall: Coordinates):
    """Return a finite-segment intersector for the revolved first wall."""
    wall_rz = np.column_stack((first_wall.x, first_wall.z))

    def intersect(start, end) -> SurfaceHit | None:
        return axisymmetric_surface_hit(start, end, wall_rz)

    return intersect


def legacy_trace_pairs(solver: ChargedParticleSolver) -> list[LegacyTracePair]:
    """Return OMP starts and wall endpoints used by the legacy SN solver."""
    pairs = []
    for branch, flux_surfaces in (
        ("lfs_lower", solver.flux_surfaces_ob_down),
        ("hfs_lower", solver.flux_surfaces_ob_up),
    ):
        for surface in flux_surfaces:
            pairs.append(
                LegacyTracePair(
                    branch=branch,
                    start_rz=np.array([surface.x_start, surface.z_start]),
                    end_rz=np.array([surface.x_end, surface.z_end]),
                )
            )
    return pairs


def legacy_weighted_trace_pairs(
    solver: ChargedParticleSolver,
) -> list[LegacyTracePair]:
    """Return legacy lower-target traces with their resolved tube power [MW]."""
    x_omp, z_omp, *_ = solver._get_arrays(solver.flux_surfaces_ob_down)
    dx_omp = x_omp - solver.x_sep_omp
    bp_omp = solver.eq.Bp(x_omp, z_omp)
    bt_omp = solver.eq.Bt(x_omp)
    field_omp = np.hypot(bp_omp, bt_omp)
    q_parallel = solver._q_par(x_omp, dx_omp, field_omp, bp_omp)
    resolved_power = (
        2.0
        * np.pi
        * q_parallel
        * bp_omp
        / field_omp
        * solver.dx_mp
        * x_omp
    )

    pairs = []
    branch_groups = (
        (
            "lfs_lower",
            solver.flux_surfaces_ob_down,
            solver.params.f_lfs_lower_target,
        ),
        (
            "hfs_lower",
            solver.flux_surfaces_ob_up,
            solver.params.f_hfs_lower_target,
        ),
    )
    for branch, surfaces, fraction in branch_groups:
        for surface, tube_power in zip(surfaces, resolved_power, strict=True):
            pairs.append(
                LegacyTracePair(
                    branch=branch,
                    start_rz=np.array([surface.x_start, surface.z_start]),
                    end_rz=np.array([surface.x_end, surface.z_end]),
                    power=float(fraction * tube_power),
                )
            )
    return pairs


def legacy_midplane_residual_power(solver: ChargedParticleSolver) -> float:
    """Return the single-null power assigned to the legacy mid-plane workaround."""
    x_omp, z_omp, *_ = solver._get_arrays(solver.flux_surfaces_ob_down)
    dx_omp = x_omp - solver.x_sep_omp
    bp_omp = solver.eq.Bp(x_omp, z_omp)
    bt_omp = solver.eq.Bt(x_omp)
    field_omp = np.hypot(bp_omp, bt_omp)
    q_parallel = solver._q_par(x_omp, dx_omp, field_omp, bp_omp)
    resolved = 2.0 * np.pi * np.sum(
        q_parallel * bp_omp / field_omp * solver.dx_mp * x_omp
    )
    return float(solver.params.P_sep_particle - resolved)


def representative_trace_pairs(
    pairs: list[LegacyTracePair],
) -> list[LegacyTracePair]:
    """Select first/middle/last trace from each branch for the CI-sized A0 gate."""
    selected = []
    branches = sorted({pair.branch for pair in pairs})
    for branch in branches:
        group = [pair for pair in pairs if pair.branch == branch]
        indices = sorted({0, len(group) // 2, len(group) - 1})
        selected.extend(group[index] for index in indices)
    return selected

def wall_arclength(first_wall: Coordinates, hit: SurfaceHit) -> float:
    """Map an axisymmetric SurfaceHit to poloidal wall arc length."""
    wall_rz = np.column_stack((first_wall.x, first_wall.z))
    segment_vectors = np.diff(wall_rz, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    index = hit.surface_index
    start = wall_rz[index]
    vector = segment_vectors[index]
    length = segment_lengths[index]
    hit_rz = np.array([np.hypot(hit.point[0], hit.point[1]), hit.point[2]])
    if length == 0.0:
        return float(cumulative[index])

    fraction = float(np.dot(hit_rz - start, vector) / length**2)
    return float(cumulative[index] + np.clip(fraction, 0.0, 1.0) * length)


def legacy_integrated_power(fixture: AxisymmetricABFixture) -> float:
    """Reproduce the existing single-null integrated-power regression [MW]."""
    n_flux_surfaces = len(fixture.legacy_solver.flux_surfaces)
    x_lfs = fixture.legacy_x[:n_flux_surfaces]
    x_hfs = fixture.legacy_x[n_flux_surfaces:]
    z_lfs = fixture.legacy_z[:n_flux_surfaces]
    z_hfs = fixture.legacy_z[n_flux_surfaces:]
    heat_lfs = fixture.legacy_heat_flux[:n_flux_surfaces]
    heat_hfs = fixture.legacy_heat_flux[n_flux_surfaces:]

    dx_lfs = x_lfs[:-1] - x_lfs[1:]
    dz_lfs = z_lfs[:-1] - z_lfs[1:]
    d_lfs = np.hypot(dx_lfs, dz_lfs)
    q_lfs = np.sum(
        heat_lfs[:-1] * d_lfs * (x_lfs[:-1] + 0.5 * np.abs(dx_lfs))
    )

    dx_hfs = x_hfs[:-1] - x_hfs[1:]
    dz_hfs = z_hfs[:-1] - z_hfs[1:]
    d_hfs = np.hypot(dx_hfs, dz_hfs)
    q_hfs = np.sum(
        heat_hfs[:-1] * d_hfs * (x_hfs[:-1] + 0.5 * np.abs(dx_hfs))
    )
    return float(q_lfs + q_hfs)
