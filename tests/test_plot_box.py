import pytest
import pandas as pd

from tabra import load_data
from tabra.plot.fig_setting import PlotKind
from tabra.plot import TabraFigure


@pytest.fixture
def tab():
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


class TestBoxPlotKind:
    def test_box_in_enum(self):
        assert PlotKind.box.value == "box"


class TestBox:
    def test_single_var(self, tab):
        """Single variable box plot."""
        fig = tab.plot.box("price", title="Price")
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_single_var_by(self, tab):
        """Single variable grouped by."""
        fig = tab.plot.box("price", by="foreign", title="Price by Origin")
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_multi_var(self, tab):
        """Multiple variables side by side."""
        fig = tab.plot.box(["price", "mpg"], title="Price & MPG")
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_multi_var_by(self, tab):
        """Multiple variables grouped."""
        fig = tab.plot.box(["price", "mpg"], by="foreign")
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_returns_tabra_figure(self, tab):
        fig = tab.plot.box("price")
        assert isinstance(fig, TabraFigure)
        assert fig.figure is not None
        fig.close()

    def test_var_must_exist(self, tab):
        with pytest.raises(KeyError):
            tab.plot.box("not_exist")
