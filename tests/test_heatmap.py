import pytest
import numpy as np
import pandas as pd
from tabra import load_data
from tabra.plot import TabraFigure


def _make_tab():
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


class TestHeatmap:
    def test_from_corr_result(self):
        tab = _make_tab()
        r = tab.data.corr(["price", "mpg", "weight"])
        fig = tab.plot.heatmap(r)
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        assert len(ax.images) >= 1
        fig.close()

    def test_auto_result(self):
        tab = _make_tab()
        tab.data.corr(["price", "mpg", "weight"])
        fig = tab.plot.heatmap()
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_from_dataframe(self):
        tab = _make_tab()
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        corr_df = df.corr()
        fig = tab.plot.heatmap(corr_df)
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_from_numpy(self):
        tab = _make_tab()
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig = tab.plot.heatmap(matrix, var_names=["x1", "x2"])
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_annot(self):
        tab = _make_tab()
        r = tab.data.corr(["price", "mpg", "weight"])
        fig = tab.plot.heatmap(r, annot=True)
        ax = fig.figure.axes[0]
        texts = ax.texts
        assert len(texts) > 0
        fig.close()

    def test_no_annot(self):
        tab = _make_tab()
        r = tab.data.corr(["price", "mpg", "weight"])
        fig = tab.plot.heatmap(r, annot=False)
        ax = fig.figure.axes[0]
        assert len(ax.texts) == 0
        fig.close()

    def test_title(self):
        tab = _make_tab()
        r = tab.data.corr(["price", "mpg", "weight"])
        fig = tab.plot.heatmap(r, title="Correlation Matrix")
        ax = fig.figure.axes[0]
        assert ax.get_title() == "Correlation Matrix"
        fig.close()

    def test_no_result_raises(self):
        from tabra.core.errors import NoResultError
        tab = _make_tab()
        with pytest.raises(NoResultError):
            tab.plot.heatmap()
