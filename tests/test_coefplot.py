import pytest
import numpy as np
import matplotlib.pyplot as plt

from tabra.plot import TabraFigure


class TestErrors:
    def test_import_errors(self):
        from tabra.core.errors import (
            PlotError, NoResultError, ResultTypeError,
            NoCommonVarsError, InvalidLevelError,
        )
        assert issubclass(NoResultError, PlotError)
        assert issubclass(ResultTypeError, PlotError)
        assert issubclass(NoCommonVarsError, PlotError)
        assert issubclass(InvalidLevelError, PlotError)

    def test_error_messages(self):
        from tabra.core.errors import NoResultError
        err = NoResultError("test msg")
        assert str(err) == "test msg"


def _make_ols_result(coef, se, var_names):
    """Helper to create a minimal OLSResult for testing."""
    from tabra.results.ols_result import OLSResult
    n = len(coef)
    return OLSResult(
        coef=np.array(coef),
        std_err=np.array(se),
        t_stat=np.array(coef) / np.array(se),
        p_value=np.array([0.01] * n),
        r_squared=0.5,
        r_squared_adj=0.45,
        f_stat=10.0,
        f_pval=0.001,
        resid=np.zeros(100),
        fitted=np.zeros(100),
        n_obs=100,
        k_vars=n,
        var_names=var_names,
        SSR=10.0,
        SSE=20.0,
        SST=30.0,
        df_model=n - 1,
        df_resid=100 - n,
        mse=0.1,
        root_mse=0.316,
    )


class TestExtractCoefs:
    def test_single_equation(self):
        from tabra.plot.coefplot import _extract_coefs
        r = _make_ols_result([0.5, -1.2, 3.0], [0.1, 0.2, 0.5],
                             ["x1", "x2", "_cons"])
        series_list = _extract_coefs(r)
        assert len(series_list) == 1
        series = series_list[0]
        assert series["label"] == "coefficients"
        assert len(series["items"]) == 3
        assert series["items"][0]["name"] == "x1"
        assert series["items"][0]["coef"] == 0.5
        assert series["items"][1]["name"] == "x2"
        for item in series["items"]:
            assert "ci_lo" in item
            assert "ci_hi" in item
            assert item["ci_lo"] < item["coef"] < item["ci_hi"]


class TestApplyFilter:
    def _make_items(self):
        return [
            {"name": "x1", "coef": 0.5, "ci_lo": 0.3, "ci_hi": 0.7},
            {"name": "x2", "coef": -1.2, "ci_lo": -1.6, "ci_hi": -0.8},
            {"name": "_cons", "coef": 3.0, "ci_lo": 2.0, "ci_hi": 4.0},
        ]

    def test_keep(self):
        from tabra.plot.coefplot import _apply_filter
        items = self._make_items()
        result = _apply_filter(items, keep=["x1", "x2"])
        assert len(result) == 2
        assert result[0]["name"] == "x1"
        assert result[1]["name"] == "x2"

    def test_drop(self):
        from tabra.plot.coefplot import _apply_filter
        items = self._make_items()
        result = _apply_filter(items, drop=["_cons"])
        assert len(result) == 2
        names = [i["name"] for i in result]
        assert "_cons" not in names

    def test_sort_ascending(self):
        from tabra.plot.coefplot import _apply_filter
        items = self._make_items()
        result = _apply_filter(items, sort="ascending")
        assert result[0]["coef"] <= result[1]["coef"] <= result[2]["coef"]

    def test_sort_descending(self):
        from tabra.plot.coefplot import _apply_filter
        items = self._make_items()
        result = _apply_filter(items, sort="descending")
        assert result[0]["coef"] >= result[1]["coef"] >= result[2]["coef"]

    def test_rename(self):
        from tabra.plot.coefplot import _apply_filter
        items = self._make_items()
        result = _apply_filter(items, rename={"x1": "Education"})
        names = [i["name"] for i in result]
        assert "Education" in names
        assert "x1" not in names

    def test_keep_nonexistent_ignored(self):
        from tabra.plot.coefplot import _apply_filter
        items = self._make_items()
        result = _apply_filter(items, keep=["x1", "nonexistent"])
        assert len(result) == 1
        assert result[0]["name"] == "x1"


class TestCoefplotSingle:
    def test_single_model_horizontal(self):
        from tabra.plot.coefplot import plot_coefplot
        r = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        fig_obj = plot_coefplot(r)
        assert isinstance(fig_obj, TabraFigure)
        ax = fig_obj.figure.axes[0]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert "x1" in ylabels
        assert "x2" in ylabels
        fig_obj.close()

    def test_single_model_vertical(self):
        from tabra.plot.coefplot import plot_coefplot
        r = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        fig_obj = plot_coefplot(r, vertical=True)
        ax = fig_obj.figure.axes[0]
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        assert "x1" in xlabels
        assert "x2" in xlabels
        fig_obj.close()

    def test_xline_default(self):
        from tabra.plot.coefplot import plot_coefplot
        r = _make_ols_result([0.5], [0.1], ["x1"])
        fig_obj = plot_coefplot(r)
        ax = fig_obj.figure.axes[0]
        lines = ax.get_lines()
        assert any(abs(l.get_xdata()[0]) < 1e-10 for l in lines)
        fig_obj.close()

    def test_ci_style_area(self):
        from tabra.plot.coefplot import plot_coefplot
        r = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        fig_obj = plot_coefplot(r, ci_style='area')
        ax = fig_obj.figure.axes[0]
        patches = ax.patches
        assert len(patches) >= 2
        fig_obj.close()

    def test_no_result_raises(self):
        from tabra.plot.coefplot import plot_coefplot
        from tabra.core.errors import NoResultError
        with pytest.raises(NoResultError):
            plot_coefplot(None)

    def test_invalid_level_raises(self):
        from tabra.plot.coefplot import plot_coefplot
        from tabra.core.errors import InvalidLevelError
        r = _make_ols_result([0.5], [0.1], ["x1"])
        with pytest.raises(InvalidLevelError):
            plot_coefplot(r, level=1.5)


class TestCoefplotMultiModel:
    def test_multi_model_overlay(self):
        from tabra.plot.coefplot import plot_coefplot
        r1 = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        r2 = _make_ols_result([0.8, -0.9], [0.15, 0.25], ["x1", "x2"])
        fig_obj = plot_coefplot([r1, r2], labels=["OLS", "IV"])
        ax = fig_obj.figure.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        assert "OLS" in legend_labels
        assert "IV" in legend_labels
        fig_obj.close()

    def test_multi_model_intersection(self):
        from tabra.plot.coefplot import plot_coefplot
        r1 = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        r2 = _make_ols_result([0.8, -0.9], [0.15, 0.25], ["x1", "x3"])
        fig_obj = plot_coefplot([r1, r2], labels=["M1", "M2"])
        ax = fig_obj.figure.axes[0]
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        assert "x1" in ylabels
        assert len(ylabels) == 1
        fig_obj.close()

    def test_no_common_vars_raises(self):
        from tabra.plot.coefplot import plot_coefplot
        from tabra.core.errors import NoCommonVarsError
        r1 = _make_ols_result([0.5], [0.1], ["a"])
        r2 = _make_ols_result([0.8], [0.15], ["b"])
        with pytest.raises(NoCommonVarsError):
            plot_coefplot([r1, r2])


def _make_mlogit_result():
    from tabra.results.mlogit_result import MLogitResult
    return MLogitResult(
        coef={"cat_B": np.array([0.5, -1.2]),
              "cat_C": np.array([0.8, 0.3])},
        std_err={"cat_B": np.array([0.1, 0.2]),
                 "cat_C": np.array([0.15, 0.25])},
        z_stat={"cat_B": np.array([5.0, -6.0]),
                "cat_C": np.array([5.33, 1.2])},
        p_value={"cat_B": np.array([0.0, 0.0]),
                 "cat_C": np.array([0.0, 0.23])},
        ll=-100.0, ll_0=-200.0, pseudo_r2=0.5,
        chi2=50.0, chi2_pval=0.001,
        n_obs=200, k_vars=2, k_cat=3, df_m=4,
        var_names=["x1", "x2"],
        y_name="y", categories=["cat_B", "cat_C"],
        base_outcome="cat_A",
        converged=True, model_name="mlogit",
        V=None,
    )


def _make_heckman_result():
    from tabra.results.heckman_result import HeckmanResult
    return HeckmanResult(
        outcome_coef=np.array([0.5, -1.2]),
        outcome_se=np.array([0.1, 0.2]),
        outcome_z=np.array([5.0, -6.0]),
        outcome_p=np.array([0.0, 0.0]),
        select_coef=np.array([0.8, 0.3]),
        select_se=np.array([0.15, 0.25]),
        select_z=np.array([5.33, 1.2]),
        select_p=np.array([0.0, 0.23]),
        athrho=0.5, athrho_se=0.1,
        lnsigma=0.3, lnsigma_se=0.05,
        rho=0.46, rho_se=0.09,
        sigma=1.35, sigma_se=0.07,
        lambda_=0.62, lambda_se=0.15,
        ll=-100.0, n_obs=200, n_selected=150, n_nonselected=50,
        chi2=30.0, chi2_pval=0.001, df_m=3,
        outcome_var_names=["x1", "x2"],
        select_var_names=["z1", "z2"],
        y_name="", converged=True, method="mle",
        lr_chi2=0.0, lr_pval=1.0, V=None,
    )


class TestExtractMulti:
    def test_mlogit_extraction(self):
        from tabra.plot.coefplot import _extract_coefs
        r = _make_mlogit_result()
        series_list = _extract_coefs(r)
        assert len(series_list) == 2
        assert series_list[0]["label"] == "cat_B"
        assert series_list[1]["label"] == "cat_C"

    def test_heckman_extraction(self):
        from tabra.plot.coefplot import _extract_coefs
        r = _make_heckman_result()
        series_list = _extract_coefs(r)
        assert len(series_list) == 2
        labels = [s["label"] for s in series_list]
        assert "outcome" in labels
        assert "selection" in labels


class TestCoefplotMultiEq:
    def test_mlogit_subplots(self):
        from tabra.plot.coefplot import plot_coefplot
        r = _make_mlogit_result()
        fig_obj = plot_coefplot(r)
        assert len(fig_obj.figure.axes) == 2
        fig_obj.close()

    def test_heckman_subplots(self):
        from tabra.plot.coefplot import plot_coefplot
        r = _make_heckman_result()
        fig_obj = plot_coefplot(r)
        assert len(fig_obj.figure.axes) == 2
        fig_obj.close()

    def test_mlogit_multi_model(self):
        from tabra.plot.coefplot import plot_coefplot
        r1 = _make_mlogit_result()
        r2 = _make_mlogit_result()
        fig_obj = plot_coefplot([r1, r2], labels=["M1", "M2"])
        assert len(fig_obj.figure.axes) == 2
        fig_obj.close()


class TestCoefplotIntegration:
    @pytest.fixture
    def dta(self):
        import pandas as pd
        from tabra import load_data
        df = pd.read_stata("/Applications/StataNow/auto.dta")
        return load_data(df, is_display_result=False)

    def test_via_plotops(self, dta):
        r = dta.est.reg("price", ["mpg", "weight"])
        fig = dta.plot.coefplot(r)
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_via_plotops_auto_result(self, dta):
        dta.est.reg("price", ["mpg", "weight"])
        fig = dta.plot.coefplot()
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_via_plotops_no_result_raises(self, dta):
        from tabra.core.errors import NoResultError
        with pytest.raises(NoResultError):
            dta.plot.coefplot()

    def test_multi_model_via_plotops(self, dta):
        r1 = dta.est.reg("price", ["mpg", "weight"])
        r2 = dta.est.reg("price", ["mpg", "weight", "length"])
        fig = dta.plot.coefplot([r1, r2], labels=["Base", "Full"])
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_via_result_object(self):
        r = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        fig = r.coefplot()
        assert isinstance(fig, TabraFigure)
        fig.close()

    def test_via_result_with_args(self):
        r = _make_ols_result([0.5, -1.2], [0.1, 0.2], ["x1", "x2"])
        fig = r.coefplot(keep=["x1"], sort="ascending")
        assert isinstance(fig, TabraFigure)
        fig.close()
