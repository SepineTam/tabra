import pytest
import numpy as np
import pandas as pd
from tabra import load_data
from tabra.plot import TabraFigure


@pytest.fixture
def tab():
    df = pd.read_stata("/Applications/StataNow/auto.dta")
    return load_data(df, is_display_result=False)


class TestRvfplot:
    def test_after_ols(self, tab):
        """rvfplot works after OLS regression."""
        tab.reg(y="price", x=["mpg", "weight"])
        fig = tab.plot.rvfplot()
        assert isinstance(fig, TabraFigure)
        ax = fig.figure.axes[0]
        # Should have scatter points
        assert len(ax.collections) >= 1
        # Should have yline at 0
        lines = ax.get_lines()
        y_data = [l.get_ydata()[0] for l in lines if len(l.get_ydata()) > 0]
        assert any(abs(v) < 1e-10 for v in y_data)
        fig.close()

    def test_labels(self, tab):
        """rvfplot has correct default axis labels."""
        tab.reg(y="price", x=["mpg", "weight"])
        fig = tab.plot.rvfplot()
        ax = fig.figure.axes[0]
        assert "itted" in ax.get_xlabel().lower()
        assert "esidual" in ax.get_ylabel().lower()
        fig.close()

    def test_custom_title(self, tab):
        """rvfplot respects custom title."""
        tab.reg(y="price", x=["mpg", "weight"])
        fig = tab.plot.rvfplot(title="Residual Diagnostics")
        ax = fig.figure.axes[0]
        assert ax.get_title() == "Residual Diagnostics"
        fig.close()

    def test_no_result_raises(self, tab):
        """rvfplot raises when no regression result stored."""
        from tabra.core.errors import NoResultError
        with pytest.raises(NoResultError):
            tab.plot.rvfplot()

    def test_result_without_resid_raises(self, tab):
        """rvfplot raises when result has no resid/fitted."""
        tab.data.corr(["price", "mpg", "weight"])
        from tabra.core.errors import ResultTypeError
        with pytest.raises(ResultTypeError):
            tab.plot.rvfplot()

    def test_with_template(self, tab):
        """rvfplot respects template settings."""
        from tabra.plot.templates import AER
        tab.reg(y="price", x=["mpg", "weight"])
        fig = tab.plot.rvfplot(template=AER)
        assert isinstance(fig, TabraFigure)
        fig.close()
