"""Prepare the scoped Bluemira #4499 candidate on the verified source blob.

AI-assisted patch. This fork-only helper is excluded from the implementation
branch. Native CAD validation is required before that branch is published.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

BASE_COMMIT = "96582c900613fd1aaf3077b52d006ade8c010ce8"
SOURCE_BLOB = "25135f8958ff37ef8598957e3a3cda8facbbafda"

OLD_OPTIONS = '''class PlotOptions(Options):
    """
    The options that are available for plotting objects
    """

    __slots__ = ()

    def __init__(self, **kwargs):
        self._options = DefaultPlotOptions()
        super().__init__(**kwargs)
'''
NEW_OPTIONS = '''class PlotOptions(Options):
    """
    The options that are available for plotting objects
    """

    __slots__ = ("_ndiscr_set",)

    def __init__(self, **kwargs):
        self._ndiscr_set = False
        self._options = DefaultPlotOptions()
        super().__init__(**kwargs)

    def __setattr__(self, attr: str, val: Any):
        """Remember explicit resolution choices, including the default value."""
        super().__setattr__(attr, val)
        if attr == "ndiscr":
            self._ndiscr_set = True
'''
OLD_WALK = '''    def _create_plotters(self, comp: Component) -> Iterator[BasePlotter]:
        if comp.is_leaf and getattr(comp, "shape", None) is not None:
            if comp.plot_options.face_options["color"] in flatten_iterable(
                BLUE_PALETTE.as_hex()
            ):
                if self.options._user_options:
                    options = self.options
                else:
                    options = comp.plot_options
            else:
                options = comp.plot_options
            yield _get_plotter_class(comp.shape)(options, data=comp.shape)
        else:
            for child in comp.children:
                yield from self._create_plotters(child)
'''
NEW_WALK = '''    def _create_plotters(
        self, comp: Component, ndiscr: int | None = None
    ) -> Iterator[BasePlotter]:
        if comp.plot_options._ndiscr_set:
            ndiscr = comp.plot_options.ndiscr

        if comp.is_leaf and getattr(comp, "shape", None) is not None:
            if comp.plot_options.face_options["color"] in flatten_iterable(
                BLUE_PALETTE.as_hex()
            ):
                if self.options._user_options:
                    options = self.options
                else:
                    options = comp.plot_options
            else:
                options = comp.plot_options
            # Keep explicit call-site options' existing precedence.
            if options is self.options and options._ndiscr_set:
                ndiscr = options.ndiscr
            # Apply only the inherited resolution to the plotter's private copy.
            kwargs = {} if ndiscr is None else {"ndiscr": ndiscr}
            yield _get_plotter_class(comp.shape)(options, data=comp.shape, **kwargs)
        else:
            for child in comp.children:
                yield from self._create_plotters(child, ndiscr)
'''


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def apply(root: Path) -> None:
    target = root / "bluemira/display/plotter.py"
    original = target.read_bytes()
    actual = git_blob(original)
    if actual != SOURCE_BLOB:
        raise SystemExit(f"Refusing source drift: expected {SOURCE_BLOB}, got {actual}")
    content = original.decode("utf-8")
    for before, after in ((OLD_OPTIONS, NEW_OPTIONS), (OLD_WALK, NEW_WALK)):
        if content.count(before) != 1:
            raise SystemExit("Expected exactly one matching source block; nothing written")
        content = content.replace(before, after, 1)
    compile(content, str(target), "exec")
    target.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    apply(parser.parse_args().root)
