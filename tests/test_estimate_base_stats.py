import pytest

from tabra.models.estimate.base import BaseModel
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
