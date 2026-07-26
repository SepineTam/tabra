#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_ols.py

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ols_data():
    """确定性数据 + numpy oracle"""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    noise = np.random.randn(n) * 0.5
    y = 2 + 3 * x1 + 1.5 * x2 + noise

    # numpy oracle (with constant, _cons last to match OLS)
    X = np.column_stack([x1, x2, np.ones(n)])
    beta_np = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_np = y - X @ beta_np
    sigma2 = resid_np @ resid_np / (n - 3)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_np = np.sqrt(sigma2 * np.diag(XtX_inv))
    SST = np.sum((y - y.mean()) ** 2)
    SSR = resid_np @ resid_np
    SSE = SST - SSR
    r2 = 1 - SSR / SST
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - 3)
    f_stat = (SSE / 2) / (SSR / (n - 3))

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df, beta_np, se_np, r2, r2_adj, f_stat


@pytest.fixture
def ols_data_no_constant():
    """无常数项数据 + numpy oracle"""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    noise = np.random.randn(n) * 0.5
    y = 3 * x1 + 1.5 * x2 + noise  # 无常数项

    # numpy oracle (without constant, uncentered R² as Stata)
    X = np.column_stack([x1, x2])
    beta_np = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_np = y - X @ beta_np
    sigma2 = resid_np @ resid_np / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_np = np.sqrt(sigma2 * np.diag(XtX_inv))
    SST = np.sum(y ** 2)  # uncentered
    SSR = resid_np @ resid_np
    SSE = SST - SSR
    r2 = 1 - SSR / SST
    r2_adj = 1 - (1 - r2) * n / (n - 2)  # Stata noconstant formula
    f_stat = (SSE / 2) / (SSR / (n - 2))

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df, beta_np, se_np, r2, r2_adj, f_stat


def test_coefficients_match_numpy(ols_data):
    """测试系数与 numpy 一致"""
    from tabra.models.estimate.ols import OLS

    df, beta_np, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert np.allclose(result.coef, beta_np, atol=1e-8)


def test_standard_errors_match_numpy(ols_data):
    """测试标准误与 numpy 一致"""
    from tabra.models.estimate.ols import OLS

    df, _, se_np, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert np.allclose(result.std_err, se_np, atol=1e-8)


def test_r_squared(ols_data):
    """测试 R-squared"""
    from tabra.models.estimate.ols import OLS

    df, _, _, r2, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert np.allclose(result.r_squared, r2, atol=1e-10)


def test_adjusted_r_squared(ols_data):
    """测试调整 R-squared"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, r2_adj, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert np.allclose(result.r_squared_adj, r2_adj, atol=1e-10)


def test_f_stat(ols_data):
    """测试 F 统计量"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, f_stat = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert np.allclose(result.f_stat, f_stat, atol=1e-8)


def test_residuals_sum_near_zero(ols_data):
    """测试残差和接近零（有常数项）"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert abs(result.resid.sum()) < 1e-8


def test_ssr_sse_equals_sst(ols_data):
    """测试 SSR + SSE = SST"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert abs(result.SSR + result.SSE - result.SST) < 1e-10


def test_residuals_length(ols_data):
    """测试残差长度"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert len(result.resid) == 100


def test_var_names_with_constant(ols_data):
    """测试有常数项的变量名"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert result.var_names == ["x1", "x2", "_cons"]


def test_var_names_no_constant(ols_data):
    """测试无常数项的变量名"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=False)

    assert result.var_names == ["x1", "x2"]


def test_p_values_in_range(ols_data):
    """测试 p 值在 [0, 1] 范围内"""
    from tabra.models.estimate.ols import OLS

    df, _, _, _, _, _ = ols_data
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=True)

    assert all(0 <= p <= 1 for p in result.p_value)


def test_no_constant_regression(ols_data_no_constant):
    """测试无常数项回归"""
    from tabra.models.estimate.ols import OLS

    df, beta_np, se_np, r2, r2_adj, f_stat = ols_data_no_constant
    model = OLS()
    result = model.fit(df, y="y", x=["x1", "x2"], is_con=False)

    assert np.allclose(result.coef, beta_np, atol=1e-8)
    assert np.allclose(result.std_err, se_np, atol=1e-8)
    assert np.allclose(result.r_squared, r2, atol=1e-10)
    assert np.allclose(result.r_squared_adj, r2_adj, atol=1e-10)
    assert np.allclose(result.f_stat, f_stat, atol=1e-8)


def test_full_rank_fit_is_invariant_to_uniform_scaling():
    from tabra.models.estimate.ols import OLS

    index = np.arange(30, dtype=float)
    x1 = np.linspace(-1.0, 1.0, 30)
    x2 = np.sin(index)
    y = 1.7 * x1 - 0.4 * x2 + np.cos(index) * 0.01
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    scaled_data = data.assign(x1=x1 * 1e-8, x2=x2 * 1e-8)

    result = OLS().fit(data, y="y", x=["x1", "x2"], is_con=False)
    scaled_result = OLS().fit(
        scaled_data, y="y", x=["x1", "x2"], is_con=False
    )

    assert np.allclose(scaled_result.coef * 1e-8, result.coef, rtol=1e-10)
    assert np.allclose(scaled_result.fitted, result.fitted, atol=1e-12)


def test_near_collinear_fit_matches_lstsq():
    from tabra.models.estimate.ols import OLS

    x1 = np.linspace(-1.0, 1.0, 80)
    x2 = x1 + 1e-7 * np.sin(np.arange(80, dtype=float))
    design = np.column_stack([x1, x2, np.ones(80)])
    y = design @ np.array([2.0, -0.5, 1.0]) + 1e-8 * np.cos(np.arange(80))
    data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    expected_coef = np.linalg.lstsq(design, y, rcond=None)[0]
    expected_resid = y - design @ expected_coef
    expected_sigma2 = expected_resid @ expected_resid / (len(y) - design.shape[1])
    _, singular_values, right_singular_vectors = np.linalg.svd(
        design, full_matrices=False
    )
    covariance_factor = right_singular_vectors.T / singular_values
    expected_std_err = np.sqrt(
        expected_sigma2 * np.sum(covariance_factor ** 2, axis=1)
    )

    assert 1e6 < np.linalg.cond(design) < 1e9
    result = OLS().fit(data, y="y", x=["x1", "x2"], is_con=True)

    assert np.allclose(result.coef, expected_coef, rtol=1e-8, atol=1e-8)
    assert np.allclose(result.fitted, design @ expected_coef, atol=1e-10)
    assert np.allclose(result.std_err, expected_std_err, rtol=1e-7, atol=1e-9)
    assert np.isfinite(result.std_err).all()
    assert (result.std_err > 0).all()


def test_correlated_standard_errors_match_xtx_inverse():
    from tabra.models.estimate.ols import OLS

    design = np.array(
        [[1.0, 1.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0]]
    )
    y = np.array([1.0, 2.0, 3.0, 5.0])
    data = pd.DataFrame({"y": y, "x1": design[:, 0], "x2": design[:, 1]})
    expected_coef = np.linalg.solve(design.T @ design, design.T @ y)
    expected_resid = y - design @ expected_coef
    expected_sigma2 = expected_resid @ expected_resid / (len(y) - design.shape[1])
    expected_std_err = np.sqrt(
        expected_sigma2 * np.diag(np.linalg.inv(design.T @ design))
    )

    result = OLS().fit(data, y="y", x=["x1", "x2"], is_con=False)

    assert np.allclose(result.std_err, expected_std_err, rtol=1e-12, atol=1e-12)


def test_rank_deficient_design_raises_clear_error():
    from tabra.models.estimate.ols import OLS

    x1 = np.arange(1.0, 8.0)
    data = pd.DataFrame({"y": x1 + 1.0, "x1": x1, "x2": 2.0 * x1})

    with pytest.raises(ValueError, match=r"rank deficient: rank=\d+, k=3"):
        OLS().fit(data, y="y", x=["x1", "x2"], is_con=True)


def test_infinite_regressor_is_rejected():
    from tabra.models.estimate.ols import OLS

    data = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, np.inf, 3.0]})

    with pytest.raises(ValueError, match="finite values"):
        OLS().fit(data, y="y", x=["x"], is_con=True)


def test_extreme_scale_never_returns_nonfinite_result():
    from tabra.models.estimate.ols import OLS

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.1], "x": [1e200, 2e200, 3e200, 4e200]}
    )

    result = OLS().fit(data, y="y", x=["x"], is_con=True)

    arrays = (
        result.coef,
        result.std_err,
        result.t_stat,
        result.p_value,
        result.resid,
        result.fitted,
    )
    scalars = (result.r_squared, result.f_stat, result.f_pval)
    assert all(np.isfinite(values).all() for values in arrays)
    assert np.isfinite(scalars).all()


def test_all_missing_sample_is_rejected():
    from tabra.models.estimate.ols import OLS

    data = pd.DataFrame({"y": [np.nan, np.nan], "x": [np.nan, np.nan]})

    with pytest.raises(ValueError, match="complete observation"):
        OLS().fit(data, y="y", x=["x"], is_con=True)


def test_n_equals_k_is_rejected():
    from tabra.models.estimate.ols import OLS

    data = pd.DataFrame({"y": [1.0, 2.0], "x1": [1.0, 0.0], "x2": [0.0, 1.0]})

    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        OLS().fit(data, y="y", x=["x1", "x2"], is_con=False)


def test_zero_response_marks_inference_as_undefined():
    from tabra.models.estimate.ols import OLS

    data = pd.DataFrame({"y": np.zeros(10), "x": np.arange(10, dtype=float)})

    result = OLS().fit(data, y="y", x=["x"], is_con=True)

    assert np.isnan(result.t_stat).all()
    assert np.isnan(result.p_value).all()
    assert np.isnan(result.f_stat)
    assert np.isnan(result.f_pval)


def test_intercept_only_model_marks_f_test_as_undefined():
    from tabra.models.estimate.ols import OLS

    data = pd.DataFrame({"y": [1.0, 2.0, 4.0, 8.0]})

    result = OLS().fit(data, y="y", x=[], is_con=True)

    assert result.coef[0] == pytest.approx(data["y"].mean())
    assert result._df_model == 0
    assert np.isnan(result.f_stat)
    assert np.isnan(result.f_pval)
