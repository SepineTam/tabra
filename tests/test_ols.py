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

    # numpy oracle (with constant)
    X = np.column_stack([np.ones(n), x1, x2])
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

    # numpy oracle (without constant)
    X = np.column_stack([x1, x2])
    beta_np = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_np = y - X @ beta_np
    sigma2 = resid_np @ resid_np / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_np = np.sqrt(sigma2 * np.diag(XtX_inv))
    SST = np.sum((y - y.mean()) ** 2)
    SSR = resid_np @ resid_np
    SSE = SST - SSR
    r2 = 1 - SSR / SST
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2)
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
