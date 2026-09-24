# SPDX-FileCopyrightText: 2026-present Kevin Rajan
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Regression tests for component-owned plotting resolution inheritance."""

from copy import deepcopy

import numpy as np
import pytest

from bluemira.base.components import Component, PhysicalComponent
from bluemira.display import plotter
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle


@pytest.fixture(params=["wire", "face"])
def component_tree(request):
    root = Component("parent")
    middle = Component("group", parent=root)
    wire = make_circle()
    shape = wire if request.param == "wire" else BluemiraFace(wire)
    first = PhysicalComponent("first", shape=shape, parent=middle)
    second = PhysicalComponent("second", shape=shape, parent=root)
    for child in (first, second):
        child.plot_options.byedges = False
    second.plot_options.face_options["color"] = "green"
    second.plot_options.wire_options["linewidth"] = 2.5
    return root, middle, first, second, wire


class TestComponentNdiscr:
    @pytest.mark.parametrize("mode", ["plot_2d", "plot_3d"])
    def test_parent_resolution_reaches_geometry(self, component_tree, mode):
        root, _, first, second, wire = component_tree
        root.plot_options.ndiscr = 240
        renderer = root._plotter
        getattr(renderer, mode)(root, show=False)

        assert [p.options.ndiscr for p in renderer._cplotters] == [240, 240]
        expected = wire.discretise(ndiscr=240, byedges=False).T
        for child_plotter in renderer._cplotters:
            np.testing.assert_allclose(child_plotter._data, expected)
        assert renderer._cplotters[1].options.face_options["color"] == "green"
        assert renderer._cplotters[1].options.wire_options["linewidth"] == pytest.approx(2.5)
        assert first.plot_options.ndiscr == second.plot_options.ndiscr == 100

    @pytest.mark.parametrize("ndiscr", [100, 64])
    @pytest.mark.parametrize("mode", ["assignment", "constructor", "modify", "copy"])
    def test_explicit_child_resolution(self, component_tree, ndiscr, mode):
        root, _, first, _, _ = component_tree
        root.plot_options.ndiscr = 240
        if mode == "assignment":
            first.plot_options.ndiscr = ndiscr
        elif mode == "constructor":
            first.plot_options = plotter.PlotOptions(ndiscr=ndiscr)
        elif mode == "modify":
            first.plot_options.modify(ndiscr=ndiscr)
        else:
            first.plot_options = deepcopy(plotter.PlotOptions(ndiscr=ndiscr))

        renderer = root._plotter
        renderer._populate_data(root)
        assert [p.options.ndiscr for p in renderer._cplotters] == [ndiscr, 240]

    def test_nearest_parent_and_sibling_isolation(self, component_tree):
        root, middle, first, second, _ = component_tree
        root.plot_options.ndiscr = 240
        middle.plot_options.ndiscr = 75
        renderer = root._plotter
        renderer._populate_data(root)

        assert [p.options.ndiscr for p in renderer._cplotters] == [75, 240]
        assert first.plot_options.ndiscr == second.plot_options.ndiscr == 100
        assert middle.plot_options.ndiscr == 75
        assert root.plot_options.ndiscr == 240

    def test_repeated_plot_does_not_store_inherited_resolution(self, component_tree):
        root, _, first, second, _ = component_tree
        first.plot_options = deepcopy(first.plot_options)
        for ndiscr in (240, 360):
            root.plot_options.ndiscr = ndiscr
            renderer = root._plotter
            renderer._populate_data(root)
            assert [p.options.ndiscr for p in renderer._cplotters] == [ndiscr, ndiscr]
        assert first.plot_options.ndiscr == second.plot_options.ndiscr == 100

    def test_explicit_call_site_resolution_keeps_existing_precedence(
        self, component_tree
    ):
        root, _, first, _, _ = component_tree
        root.plot_options.ndiscr = 240
        first.plot_options.ndiscr = 100
        options = plotter.PlotOptions(ndiscr=300)
        options._user_options = True
        renderer = plotter.ComponentPlotter(options)
        renderer._populate_data(root)

        # The existing explicit-options path applies to the default-colour child.
        assert renderer._cplotters[0].options.ndiscr == 300

    def test_unrelated_call_site_options_do_not_reset_resolution(self, component_tree):
        root, _, _, _, _ = component_tree
        root.plot_options.ndiscr = 240
        options = plotter.PlotOptions(show_wires=False)
        options._user_options = True
        renderer = plotter.ComponentPlotter(options)
        renderer._populate_data(root)

        assert [p.options.ndiscr for p in renderer._cplotters] == [240, 240]
        assert renderer._cplotters[0].options.show_wires is False

    def test_unset_resolution_remains_default(self, component_tree):
        root, _, _, _, _ = component_tree
        renderer = root._plotter
        renderer._populate_data(root)
        assert [p.options.ndiscr for p in renderer._cplotters] == [100, 100]
