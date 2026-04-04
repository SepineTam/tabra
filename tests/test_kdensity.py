import pytest
import numpy as np
import matplotlib.pyplot as plt
from tabra import load_data
from tabra.plot import TabraFigure
import pandas as pd


def _make_tab():
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


class TestKdensity:
    def test_single_var(self):
        tab = _make_tab()
        fig = tab.plot.kdensity("price")
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        assert len(ax.lines) >= 1
        fig.close()

    def test_with_by(self):
        tab = _make_tab()
        fig = tab.plot.kdensity("price", by="foreign")
        axes = fig.figure.axes
        assert len(axes) == 2
        fig.close()

    def test_custom_bw(self):
        tab = _make_tab()
        fig = tab.plot.kdensity("price", bw=0.3)
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_returns_tabra_figure(self):
        tab = _make_tab()
        fig = tab.plot.kdensity("mpg")
        assert isinstance(fig, TabraFigure)
        assert fig.figure is not None
        fig.close()

    def test_title_and_labels(self):
        tab = _make_tab()
        fig = tab.plot.kdensity("price", title="Price Density",
                                xtitle="Price", ytitle="Density")
        ax = fig.figure.axes[0]
        assert ax.get_title() == "Price Density"
        fig.close()

    def test_var_list(self):
        tab = _make_tab()
        fig = tab.plot.kdensity(["price", "mpg"])
        axes = fig.figure.axes
        assert len(axes) == 2
        fig.close()

    def test_var_list_with_by(self):
        tab = _make_tab()
        fig = tab.plot.kdensity(["price", "mpg"], by="foreign")
        # 2 vars x 2 groups = 4 visible subplots
        visible_axes = [ax for ax in fig.figure.axes if ax.get_visible()]
        assert len(visible_axes) == 4
        fig.close()
