"""Stdlib-only geometric counterexample for bluemira fork PR #9.

Source/provenance: DarioV86's Bluemira #4474 and kvnloo/bluemira #8/#9.
This is *not* a Bluemira, OpenMC, DAGMC, or reactor validation run.

Run: python -m unittest -v test_toroidal_chord_contact.py
"""
from __future__ import annotations

import math
import unittest


def toroidal_exact_endpoint(radius: float, ds: float) -> tuple[float, float]:
    """Exact purely toroidal B=(-y,x,0) field-line endpoint after arc ds."""
    theta = ds / radius
    return radius * math.cos(theta), radius * math.sin(theta)


def midpoint_endpoint(radius: float, ds: float) -> tuple[float, float]:
    """Explicit midpoint for dp/ds=B/|B| with initial p=(radius, 0)."""
    mx, my = radius, ds / 2.0
    mag = math.hypot(mx, my)
    return radius - ds * my / mag, ds * mx / mag


def segment_min_radius(p0: tuple[float,float], p1: tuple[float,float]) -> float:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    denom = dx*dx + dy*dy
    if denom == 0:
        return math.hypot(*p0)
    t = min(1.0, max(0.0, -(p0[0]*dx + p0[1]*dy) / denom))
    return math.hypot(p0[0] + t*dx, p0[1] + t*dy)


def finite_segment_crosses_cylinder(p0: tuple[float,float], p1: tuple[float,float], wall_radius: float) -> bool:
    """Exact line/cylinder intersection test, not a curved-path intersection."""
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    a = dx*dx + dy*dy
    if a == 0.0:
        return math.isclose(math.hypot(*p0), wall_radius, abs_tol=1e-12)
    b = 2.0*(p0[0]*dx+p0[1]*dy)
    c = p0[0]**2+p0[1]**2-wall_radius**2
    disc = b*b-4*a*c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    return any(0.0 <= (-b + s*root)/(2*a) <= 1.0 for s in (-1,1))


def exact_sagitta(radius: float, ds: float) -> float:
    return radius * (1.0 - math.cos(ds / (2.0*radius)))


def max_arc_step_for_clearance(radius: float, clearance: float) -> float:
    """Circle-specific threshold only: not a general CAD/contact step policy."""
    if not 0 < clearance < radius:
        raise ValueError("require 0 < clearance < radius")
    return 2.0*radius*math.acos(1.0-clearance/radius)


class ToroidalChordContactTests(unittest.TestCase):
    def test_exact_field_line_never_crosses_inner_wall(self):
        radius = 2.0
        for ds in (0.05,0.1,0.2,0.4):
            with self.subTest(ds=ds):
                x,y = toroidal_exact_endpoint(radius, ds)
                self.assertAlmostEqual(math.hypot(x,y), radius, places=13)
                self.assertGreater(radius, 1.999)

    def test_even_exact_endpoint_chord_can_create_false_wall_hit(self):
        p0=(2.0,0.0)
        short=toroidal_exact_endpoint(2.0,0.1)
        long=toroidal_exact_endpoint(2.0,0.2)
        self.assertFalse(finite_segment_crosses_cylinder(p0, short, 1.999))
        self.assertTrue(finite_segment_crosses_cylinder(p0, long, 1.999))
        self.assertLess(segment_min_radius(p0,long),1.999)
        self.assertAlmostEqual(segment_min_radius(p0,long),2.0-exact_sagitta(2.0,0.2),places=12)

    def test_midpoint_endpoint_chord_failure_is_separate_from_endpoint_accuracy(self):
        p0=(2.0,0.0)
        p1=midpoint_endpoint(2.0,0.2)
        exact=toroidal_exact_endpoint(2.0,0.2)
        error=math.dist(p1,exact)
        self.assertLess(error, 1e-4)
        self.assertTrue(finite_segment_crosses_cylinder(p0,p1,1.999))
        self.assertLess(segment_min_radius(p0,p1),1.999)
        self.assertFalse(finite_segment_crosses_cylinder(p0,midpoint_endpoint(2.0,0.1),1.999))

    def test_geometric_clearance_threshold(self):
        threshold=max_arc_step_for_clearance(2.0,0.001)
        self.assertGreater(threshold,0.1)
        self.assertLess(threshold,0.2)
        self.assertLess(exact_sagitta(2.0,0.1),0.001)
        self.assertGreater(exact_sagitta(2.0,0.2),0.001)

    def test_analytic_segment_minimum_matches_sagitta_sweep(self):
        p0=(2.0,0.0)
        for ds in (0.025,0.05,0.1,0.15,0.2,0.3,0.4):
            with self.subTest(ds=ds):
                p1=toroidal_exact_endpoint(2.0,ds)
                self.assertAlmostEqual(2.0-segment_min_radius(p0,p1),exact_sagitta(2.0,ds),places=12)


if __name__ == '__main__':
    unittest.main()
