import numpy as np
import pytest

from tabra.models.estimate.base import BaseModel
from tabra.models.estimate.stats import EstimationStats
from tabra.ops.stats import f_pval


class _DummyModel(BaseModel):
    def fit(self, df, y, x, **kwargs):
        raise NotImplementedError

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError


def test_model_and_residual_df_helpers():
    assert _DummyModel._model_df(5, is_con=True) == 4
    assert _DummyModel._model_df(5, is_con=False) == 5
    assert _DummyModel._resid_df(100, 5) == 95


def test_adjusted_r_squared_matches_constant_formula():
    r2 = 0.8
    n_obs = 100
    k_vars = 3
    expected = 1 - (1 - r2) * (n_obs - 1) / (n_obs - k_vars)
    assert _DummyModel._adjusted_r_squared(r2, n_obs, k_vars, is_con=True) == pytest.approx(expected)


def test_adjusted_r_squared_matches_no_constant_formula():
    r2 = 0.8
    n_obs = 100
    k_vars = 3
    expected = 1 - (1 - r2) * n_obs / (n_obs - k_vars)
    assert _DummyModel._adjusted_r_squared(r2, n_obs, k_vars, is_con=False) == pytest.approx(expected)


def test_pseudo_r_squared_handles_zero_null_ll():
    assert _DummyModel._pseudo_r_squared(-10.0, 0.0) == 0.0
    assert _DummyModel._pseudo_r_squared(-10.0, -20.0) == pytest.approx(0.5)


def test_f_statistics_return_zero_when_df_invalid():
    f_stat, f_p = _DummyModel._f_statistics(10.0, 5.0, 0, 90)
    assert f_stat == 0.0
    assert f_p == 0.0


def test_f_statistics_match_manual_formula_when_valid():
    sse = 120.0
    ssr = 60.0
    df_model = 3
    df_resid = 96
    expected_f = (sse / df_model) / (ssr / df_resid)
    expected_p = f_pval(expected_f, df_model, df_resid)
    f_stat, f_p = _DummyModel._f_statistics(sse, ssr, df_model, df_resid)
    assert f_stat == pytest.approx(expected_f)
    assert f_p == pytest.approx(expected_p)


def test_estimation_stats_chi2_p_value_and_r_squared():
    chi2 = 5.991464547107979
    assert EstimationStats.chi2_p_value(chi2, 2) == pytest.approx(0.05, abs=1e-6)
    assert EstimationStats.chi2_p_value(chi2, 0) == 1.0
    assert EstimationStats.r_squared(20.0, 100.0) == pytest.approx(0.8)
    assert EstimationStats.r_squared(20.0, 0.0) == 0.0


def test_estimation_stats_mse_and_root_mse():
    assert EstimationStats.mse(50.0, 25) == pytest.approx(2.0)
    assert EstimationStats.root_mse(50.0, 25) == pytest.approx(2.0 ** 0.5)
    assert EstimationStats.mse(50.0, 0) == 0.0
    assert EstimationStats.root_mse(50.0, 0) == 0.0


def test_estimation_stats_lr_test():
    ll_full = -120.0
    ll_null = -130.0
    chi2, p_value = EstimationStats.lr_test(ll_full, ll_null, 3)
    assert chi2 == pytest.approx(20.0)
    assert p_value == pytest.approx(EstimationStats.chi2_p_value(20.0, 3))

    chi2_zero, p_value_zero = EstimationStats.lr_test(-130.0, -120.0, 3)
    assert chi2_zero == 0.0
    assert p_value_zero == pytest.approx(1.0)


def test_estimation_stats_aic_and_bic():
    ll = -100.0
    n_params = 5
    n_obs = 200
    assert EstimationStats.aic(ll, n_params) == pytest.approx(210.0)
    expected_bic = np.log(n_obs) * n_params - 2 * ll
    assert EstimationStats.bic(ll, n_params, n_obs) == pytest.approx(expected_bic)
    assert np.isnan(EstimationStats.bic(ll, n_params, 0))
