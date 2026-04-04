import pytest
import pandas as pd
import matplotlib.pyplot as plt

from tabra import load_data
from tabra.plot.fig_setting import PlotKind
from tabra.plot.templates import AER
from tabra.plot import TabraFigure


@pytest.fixture
def tab():
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


class TestPlotKind:
    def test_enum_values(self):
        assert PlotKind.scatter.value == "scatter"
        assert PlotKind.line.value == "line"
        assert PlotKind.hist.value == "hist"
        assert PlotKind.bar.value == "bar"
        assert PlotKind.violin.value == "violin"
        assert PlotKind.pie.value == "pie"

    def test_enum_members(self):
        members = [m.name for m in PlotKind]
        assert "scatter" in members
        assert "line" in members
        assert len(members) == 13


class TestMix:
    def test_scatter_plus_line(self, tab):
        """Scatter and line overlaid on same axes."""
        fig = tab.plot.mix(
            [
                {PlotKind.scatter: {"y": "price", "x": "weight"}},
                {PlotKind.line: {"y": "price", "x": "weight"}},
            ],
            title="Scatter + Line",
            xtitle="Weight",
            ytitle="Price",
        )
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        assert ax.get_title() == "Scatter + Line"
        assert ax.get_xlabel() == "Weight"
        assert ax.get_ylabel() == "Price"
        # scatter + line = 2 collections/lines on ax
        assert len(ax.lines) + len(ax.collections) >= 2
        fig.close()

    def test_single_layer(self, tab):
        """Mix with a single layer still works."""
        fig = tab.plot.mix(
            [{PlotKind.scatter: {"y": "price", "x": "weight"}}],
        )
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_with_template(self, tab):
        """Mix respects template settings."""
        tab.config.set_plot_template(AER)
        fig = tab.plot.mix(
            [
                {PlotKind.scatter: {"y": "price", "x": "weight"}},
                {PlotKind.line: {"y": "price", "x": "weight"}},
            ],
            title="With AER",
        )
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_pie_raises(self, tab):
        """Pie cannot be mixed with other types."""
        with pytest.raises(ValueError, match="pie"):
            tab.plot.mix(
                [
                    {PlotKind.pie: {}},
                    {PlotKind.scatter: {"y": "price", "x": "weight"}},
                ],
            )

    def test_empty_layers_raises(self, tab):
        """Empty layers list raises error."""
        with pytest.raises(ValueError, match="layer"):
            tab.plot.mix([])
