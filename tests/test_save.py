import os
from pathlib import Path

import pandas as pd
import pytest

from tabra import load_data
from tabra.plot import TabraFigure


@pytest.fixture
def tab(tmp_path):
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


@pytest.fixture
def fig(tab):
    return tab.plot.scatter("price", "weight")


class TestSaveFormats:
    def test_no_ext_no_formats_defaults_png(self, fig, tmp_path):
        """No extension, formats=None → defaults to png."""
        path = str(tmp_path / "test")
        fig.save(path)
        assert os.path.exists(path + ".png")
        fig.close()

    def test_has_ext_no_formats_uses_ext(self, fig, tmp_path):
        """Has extension, formats=None → use that extension."""
        path = str(tmp_path / "test.pdf")
        fig.save(path)
        assert os.path.exists(path)
        fig.close()

    def test_formats_generates_multiple(self, fig, tmp_path):
        """formats=["png", "pdf"] → generates both files."""
        path = str(tmp_path / "test")
        fig.save(path, formats=["png", "pdf"])
        assert os.path.exists(str(tmp_path / "test.png"))
        assert os.path.exists(str(tmp_path / "test.pdf"))
        fig.close()

    def test_has_ext_with_formats_appends(self, fig, tmp_path):
        """Has extension + formats → ori_name.ext.fmt."""
        path = str(tmp_path / "test.png")
        fig.save(path, formats=["pdf"])
        assert os.path.exists(str(tmp_path / "test.png.pdf"))
        fig.close()

    def test_save_returns_self(self, fig, tmp_path):
        path = str(tmp_path / "test.png")
        result = fig.save(path)
        assert result is fig
        fig.close()


class TestSaveBase:
    def test_relative_path_uses_save_base(self, tab, tmp_path):
        """Relative path + save_base → prepend save_base."""
        tab.config.set_figure_save_base(tmp_path)
        fig = tab.plot.scatter("price", "weight")
        fig.save("test.png")
        assert os.path.exists(str(tmp_path / "test.png"))
        fig.close()

    def test_absolute_path_ignores_save_base(self, tab, tmp_path):
        """Absolute path ignores save_base."""
        tab.config.set_figure_save_base("/tmp/should_not_use")
        abs_path = str(tmp_path / "abs_test.png")
        fig = tab.plot.scatter("price", "weight")
        fig.save(abs_path)
        assert os.path.exists(abs_path)
        fig.close()

    def test_no_save_base_uses_cwd(self, tab):
        """No save_base → relative to cwd."""
        fig = tab.plot.scatter("price", "weight")
        # just check it doesn't crash with relative path
        result = fig.save("tmp/test_no_base.png")
        assert os.path.exists("tmp/test_no_base.png")
        fig.close()
