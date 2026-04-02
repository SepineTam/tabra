#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_plot.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from tabra import load_data
from tabra.plot import TabraFigure


@pytest.fixture
def plot_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "x": np.random.randn(n),
        "y": np.random.randn(n),
        "cat": ["a", "b"] * 25,
    })


class TestPlotOps:
    def test_plot_property_exists(self, plot_df):
        tab = load_data(plot_df, is_display_result=False)
        from tabra.plot import PlotOps
        assert isinstance(tab.plot, PlotOps)

    def test_scatter_returns_tabra_figure(self, plot_df):
        tab = load_data(plot_df, is_display_result=False)
        fig = tab.plot.scatter("y", "x")
        assert isinstance(fig, TabraFigure)
        plt.close(fig._fig)

    def test_scatter_with_title(self, plot_df):
        tab = load_data(plot_df, is_display_result=False)
        fig = tab.plot.scatter("y", "x", title="Test Plot")
        ax = fig._fig.axes[0]
        assert ax.get_title() == "Test Plot"
        plt.close(fig._fig)

    def test_scatter_with_axis_labels(self, plot_df):
        tab = load_data(plot_df, is_display_result=False)
        fig = tab.plot.scatter("y", "x", xtitle="X Label", ytitle="Y Label")
        ax = fig._fig.axes[0]
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        plt.close(fig._fig)

    def test_scatter_default_labels_are_column_names(self, plot_df):
        tab = load_data(plot_df, is_display_result=False)
        fig = tab.plot.scatter("y", "x")
        ax = fig._fig.axes[0]
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        plt.close(fig._fig)

    def test_tabra_figure_save(self, plot_df, tmp_path):
        tab = load_data(plot_df, is_display_result=False)
        fig = tab.plot.scatter("y", "x", title="Save Test")
        save_path = str(tmp_path / "test_scatter.png")
        fig.save(save_path)
        import os
        assert os.path.exists(save_path)
        plt.close(fig._fig)

    def test_tabra_figure_show(self, plot_df):
        tab = load_data(plot_df, is_display_result=False)
        fig = tab.plot.scatter("y", "x")
        result = fig.show()
        assert result is not None
        plt.close(fig._fig)
