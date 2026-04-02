import pytest
import pandas as pd
import numpy as np

from tabra import load_data
from tabra.plot.fig_setting import PlotKind
from tabra.plot import TabraFigure


@pytest.fixture
def tab():
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


class TestLfitEnum:
    def test_lfit_in_enum(self):
        assert PlotKind.lfit.value == "lfit"

    def test_lfitci_in_enum(self):
        assert PlotKind.lfitci.value == "lfitci"


class TestLfit:
    def test_lfit_only(self, tab):
        """lfit draws a regression line."""
        fig = tab.plot.mix(
            [{PlotKind.lfit: {"y": "price", "x": "weight"}}],
            title="Linear Fit",
        )
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        assert len(ax.lines) == 1
        assert ax.get_title() == "Linear Fit"
        fig.close()

    def test_lfitci_has_fill(self, tab):
        """lfitci draws a regression line with CI band."""
        fig = tab.plot.mix(
            [{PlotKind.lfitci: {"y": "price", "x": "weight"}}],
        )
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        # line + fill_between (polyCollection)
        assert len(ax.lines) >= 1
        assert len(ax.collections) >= 1
        fig.close()

    def test_scatter_plus_lfit(self, tab):
        """Classic twoway: scatter + lfit overlay."""
        fig = tab.plot.mix(
            [
                {PlotKind.scatter: {"y": "price", "x": "weight"}},
                {PlotKind.lfit: {"y": "price", "x": "weight"}},
            ],
            title="Price vs Weight",
            xtitle="Weight",
            ytitle="Price",
        )
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        assert ax.get_title() == "Price vs Weight"
        assert ax.get_xlabel() == "Weight"
        assert ax.get_ylabel() == "Price"
        fig.close()

    def test_scatter_plus_lfitci(self, tab):
        """Scatter with CI regression band."""
        fig = tab.plot.mix(
            [
                {PlotKind.lfitci: {"y": "price", "x": "weight"}},
                {PlotKind.scatter: {"y": "price", "x": "weight"}},
            ],
            title="Price vs Weight (CI)",
        )
        assert isinstance(fig, TabraFigure)
        fig.close()
