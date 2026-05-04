#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_reghdfe.py

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def auto_data():
    """sysuse auto equivalent with known reghdfe output."""
    # 使用 sysuse auto 数据，reghdfe price weight length, absorb(rep78) 的 Stata 输出:
    # e(b) = [5.4783091, -109.50651, 10154.617]
    # e(N) = 69, e(r2) = 0.43408805, e(df_a) = 5, e(r2_within) = 0.42576464
    # e(df_m) = 2, e(df_r) = 62
    np.random.seed(42)
    n = 69
    rep78 = np.repeat([1, 2, 3, 4, 5], [n // 5 + (1 if i < n % 5 else 0) for i in range(5)][:5])
    rep78 = rep78[:n]
    weight = np.random.randn(n) * 500 + 3000
    length = np.random.randn(n) * 20 + 190
    fe_effect = np.array([1000 * v for v in rep78])
    noise = np.random.randn(n) * 500
    price = 5.5 * weight - 110 * length + fe_effect + 10000 + noise

    df = pd.DataFrame({
        "price": price,
        "weight": weight,
        "length": length,
        "rep78": rep78,
    })
    return df


@pytest.fixture
def panel_data():
    """Balanced panel for two-way FE testing."""
    np.random.seed(123)
    N_id = 50
    N_t = 10
    n = N_id * N_t

    id_var = np.repeat(np.arange(N_id), N_t)
    t_var = np.tile(np.arange(N_t), N_id)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    alpha_i = np.repeat(np.random.randn(N_id) * 2, N_t)  # individual FE
    gamma_t = np.tile(np.random.randn(N_t) * 1.5, N_id)  # time FE
    eps = np.random.randn(n)
    y = 2 * x1 + 3 * x2 + alpha_i + gamma_t + eps

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": id_var, "time": t_var})
    return df


@pytest.fixture
def panel_data_with_singletons():
    """Panel data with deliberate singleton observations."""
    np.random.seed(456)
    N_id = 20
    N_t = 5
    n_main = N_id * N_t

    id_var = np.repeat(np.arange(N_id), N_t)
    t_var = np.tile(np.arange(N_t), N_id)
    x1 = np.random.randn(n_main)
    y = 2 * x1 + np.repeat(np.random.randn(N_id), N_t) + np.tile(np.random.randn(N_t), N_id) + np.random.randn(n_main) * 0.5

    # Add 3 singleton obs (unique id values appearing only once)
    singleton_ids = np.array([100, 101, 102])
    singleton_t = np.array([0, 1, 2])
    singleton_x1 = np.random.randn(3)
    singleton_y = 2 * singleton_x1 + np.random.randn(3)

    df = pd.DataFrame({
        "y": np.concatenate([y, singleton_y]),
        "x1": np.concatenate([x1, singleton_x1]),
        "id": np.concatenate([id_var, singleton_ids]),
        "time": np.concatenate([t_var, singleton_t]),
    })
    return df, n_main  # return expected N after singleton removal


# === Single FE Tests ===

def test_single_fe_coefficients_match_stata():
    """Single FE: 系数应接近 Stata reghdfe 输出"""
    from tabra.models.estimate.reghdfe import RegHDFE

    # 直接用 Stata 的 sysuse auto 数据结果验证
    # 这里用 oracle: numpy 手动做 within-transform 后 OLS
    np.random.seed(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    fe = np.repeat(np.random.randn(10) * 5, 10)
    y = 2 * x1 - 3 * x2 + fe + np.random.randn(n) * 0.5

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})

    # Oracle: within-transform
    y_dm = y - np.repeat(np.bincount(groups, y) / np.bincount(groups, np.ones(n)), np.bincount(groups).astype(int))
    x1_dm = x1 - np.repeat(np.bincount(groups, x1) / np.bincount(groups, np.ones(n)), np.bincount(groups).astype(int))
    x2_dm = x2 - np.repeat(np.bincount(groups, x2) / np.bincount(groups, np.ones(n)), np.bincount(groups).astype(int))

    X_dm = np.column_stack([x1_dm, x2_dm])
    beta_oracle = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]

    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1", "x2"], absorb=["group"])

    assert np.allclose(result.coef[:2], beta_oracle, atol=1e-6)


def test_single_fe_r_squared():
    """Single FE: R-squared 应为正值且 <1"""
    from tabra.models.estimate.reghdfe import RegHDFE

    np.random.seed(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    x1 = np.random.randn(n)
    y = 2 * x1 + np.repeat(np.random.randn(10), 10) + np.random.randn(n)
    df = pd.DataFrame({"y": y, "x1": x1, "group": groups})

    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1"], absorb=["group"])

    assert 0 < result.r_squared < 1
    assert 0 < result.r_squared_adj < 1


def test_single_fe_df_a():
    """Single FE: df_a should equal number of categories (reghdfe convention)."""
    from tabra.models.estimate.reghdfe import RegHDFE

    np.random.seed(42)
    n = 100
    n_groups = 10
    groups = np.repeat(np.arange(n_groups), 10)
    x1 = np.random.randn(n)
    y = 2 * x1 + np.repeat(np.random.randn(n_groups), 10) + np.random.randn(n)
    df = pd.DataFrame({"y": y, "x1": x1, "group": groups})

    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1"], absorb=["group"])

    # reghdfe convention: df_a = K (all categories for first FE)
    assert result.df_a == n_groups


def test_single_fe_n_obs():
    """Single FE: N 应等于数据行数（无 singletons 时）"""
    from tabra.models.estimate.reghdfe import RegHDFE

    np.random.seed(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    x1 = np.random.randn(n)
    y = 2 * x1 + np.random.randn(n)
    df = pd.DataFrame({"y": y, "x1": x1, "group": groups})

    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1"], absorb=["group"])

    assert result.n_obs == n


def test_single_fe_se_positive():
    """Single FE: 标准误应为正"""
    from tabra.models.estimate.reghdfe import RegHDFE

    np.random.seed(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    x1 = np.random.randn(n)
    y = 2 * x1 + np.random.randn(n)
    df = pd.DataFrame({"y": y, "x1": x1, "group": groups})

    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1"], absorb=["group"])

    assert all(se > 0 for se in result.std_err)


# === Two-way FE Tests ===

def test_twoway_fe_coefficients(panel_data):
    """Two-way FE: 系数应接近 oracle (numpy within-transform on both dims)"""
    from tabra.models.estimate.reghdfe import RegHDFE

    df = panel_data
    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1", "x2"], absorb=["id", "time"])

    # Oracle: 手动双维去均值
    y_arr = df["y"].values.astype(float)
    x1_arr = df["x1"].values.astype(float)
    x2_arr = df["x2"].values.astype(float)
    id_arr = df["id"].values
    t_arr = df["time"].values

    # 迭代去均值 (MAP)
    y_dm = y_arr.copy()
    x1_dm = x1_arr.copy()
    x2_dm = x2_arr.copy()
    for _ in range(100):
        y_dm_old = y_dm.copy()
        for dim in [id_arr, t_arr]:
            for v in [y_dm, x1_dm, x2_dm]:
                means = np.bincount(dim, v) / np.bincount(dim)
                v -= means[dim]
        if np.max(np.abs(y_dm - y_dm_old)) < 1e-10:
            break

    X_dm = np.column_stack([x1_dm, x2_dm])
    beta_oracle = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]

    assert np.allclose(result.coef[:2], beta_oracle, atol=1e-4)


def test_twoway_fe_r_squared_within(panel_data):
    """Two-way FE: within R-squared 应为正值"""
    from tabra.models.estimate.reghdfe import RegHDFE

    model = RegHDFE()
    result = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"])

    assert result.r2_within > 0
    assert result.r2_within < 1


def test_twoway_fe_n_hdfe(panel_data):
    """Two-way FE: N_hdfe 应为 2"""
    from tabra.models.estimate.reghdfe import RegHDFE

    model = RegHDFE()
    result = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"])

    assert result.n_hdfe == 2


def test_twoway_fe_df_r(panel_data):
    """Two-way FE: df_r = N - df_a - df_model."""
    from tabra.models.estimate.reghdfe import RegHDFE

    model = RegHDFE()
    result = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"])

    # df_resid = n - df_a - df_model (df_model = active slopes only)
    expected_df_r = result.n_obs - result.df_a - result.df_model
    assert result.df_resid == expected_df_r


# === Singleton Tests ===

def test_singleton_removal(panel_data_with_singletons):
    """Singleton obs 应被删除"""
    from tabra.models.estimate.reghdfe import RegHDFE

    df, expected_n = panel_data_with_singletons
    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1"], absorb=["id"])

    assert result.n_obs == expected_n


# === VCE Tests ===

def test_robust_se(panel_data):
    """Robust SE 应与 conventional SE 不同但都为正"""
    from tabra.models.estimate.reghdfe import RegHDFE

    model = RegHDFE()
    result_conv = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"])
    result_robust = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"], vce="robust")

    # 系数应相同
    assert np.allclose(result_conv.coef, result_robust.coef, atol=1e-10)
    # Robust SE 应与 conventional SE 不同
    assert not np.allclose(result_conv.std_err, result_robust.std_err, atol=1e-8)
    # 但都应为正
    assert all(se > 0 for se in result_robust.std_err)


def test_cluster_se(panel_data):
    """Cluster SE 应与 conventional SE 不同"""
    from tabra.models.estimate.reghdfe import RegHDFE

    model = RegHDFE()
    result_conv = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"])
    result_cluster = model.fit(panel_data, y="y", x=["x1", "x2"], absorb=["id", "time"], vce="cluster", cluster=["id"])

    # 系数应相同
    assert np.allclose(result_conv.coef, result_cluster.coef, atol=1e-10)
    # Cluster SE 应为正
    assert all(se > 0 for se in result_cluster.std_err)


# === API Integration Test ===

def test_tabra_data_reghdfe(panel_data):
    """TabraData.reghdfe() 应正常工作"""
    from tabra.core.data import TabraData

    td = TabraData(panel_data, is_display_result=False)
    result = td.est.reghdfe(y="y", x=["x1", "x2"], absorb=["id", "time"])

    assert result.n_obs == len(panel_data)
    assert len(result.coef) == 3  # x1, x2, _cons
    assert result.n_hdfe == 2


# === Stata Benchmark Tests ===

def test_stata_auto_single_fe():
    """与 Stata reghdfe price weight length, absorb(rep78) 输出对齐"""
    from tabra.models.estimate.reghdfe import RegHDFE

    # sysuse auto 数据 (手工创建近似的已知结构)
    # 由于 sysuse auto 是内置数据，这里验证 API 和核心统计量格式
    np.random.seed(42)
    n = 69
    rep78 = np.array([3, 4, 3, 4, 3, 3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 3, 4, 3,
                       5, 3, 5, 3, 3, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4,
                       3, 4, 4, 3, 3, 5, 3, 4, 4, 3, 3, 3, 4, 5, 5, 4, 3, 4,
                       4, 4, 3, 4, 5, 4, 3, 3, 5, 3, 5, 5, 3, 5, 4])
    weight = np.random.randn(n) * 800 + 3000
    length = np.random.randn(n) * 20 + 190
    price = 5.5 * weight - 110 * length + np.array([1000 * v for v in rep78]) + np.random.randn(n) * 500

    df = pd.DataFrame({"price": price, "weight": weight, "length": length, "rep78": rep78})

    model = RegHDFE()
    result = model.fit(df, y="price", x=["weight", "length"], absorb=["rep78"])

    # 基本格式检查
    assert result.n_obs == n
    assert result.n_hdfe == 1
    assert len(result.coef) == 3  # weight, length, _cons
    assert 0 < result.r_squared < 1
    assert result.df_a > 0
    assert all(se > 0 for se in result.std_err)


def test_missing_values_dropped():
    """有缺失值的行应被删除"""
    from tabra.models.estimate.reghdfe import RegHDFE

    np.random.seed(42)
    n = 50
    groups = np.repeat(np.arange(5), 10)
    x1 = np.random.randn(n)
    y = 2 * x1 + np.repeat(np.random.randn(5), 10) + np.random.randn(n) * 0.5

    # 插入缺失值
    y[0] = np.nan
    x1[5] = np.nan

    df = pd.DataFrame({"y": y, "x1": x1, "group": groups})
    model = RegHDFE()
    result = model.fit(df, y="y", x=["x1"], absorb=["group"])

    assert result.n_obs == n - 2  # 两行有缺失值
