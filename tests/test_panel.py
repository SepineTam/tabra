#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_panel.py

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fe_data():
    """Balanced panel: 5 groups x 10 periods, known DGP."""
    np.random.seed(42)
    n_groups = 5
    T = 10
    N = n_groups * T

    group_id = np.repeat(np.arange(n_groups), T)
    alpha = np.array([1.0, 3.0, 5.0, 2.0, 4.0])  # fixed effects
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    noise = np.random.randn(N) * 0.5
    y = alpha[group_id] + 2 * x1 + 1.5 * x2 + noise

    df = pd.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
        "group": group_id,
    })

    # --- numpy oracle: within estimator ---
    # Demean within groups
    y_demean = np.zeros(N)
    x1_demean = np.zeros(N)
    x2_demean = np.zeros(N)
    for i in range(n_groups):
        mask = group_id == i
        y_demean[mask] = y[mask] - y[mask].mean()
        x1_demean[mask] = x1[mask] - x1[mask].mean()
        x2_demean[mask] = x2[mask] - x2[mask].mean()

    # OLS on demeaned data (no constant)
    X_dm = np.column_stack([x1_demean, x2_demean])
    beta_oracle = np.linalg.lstsq(X_dm, y_demean, rcond=None)[0]

    # Residuals and stats
    resid_within = y_demean - X_dm @ beta_oracle
    SSR_w = float(resid_within @ resid_within)
    k = 2  # number of regressors (no constant in within)
    df_resid = N - n_groups - k
    sigma2_e = SSR_w / df_resid
    XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
    se_oracle = np.sqrt(sigma2_e * np.diag(XtX_inv))
    t_oracle = beta_oracle / se_oracle

    # sigma_u: standard deviation of estimated fixed effects
    group_means_y = np.array([y[group_id == i].mean() for i in range(n_groups)])
    group_means_x1 = np.array([x1[group_id == i].mean() for i in range(n_groups)])
    group_means_x2 = np.array([x2[group_id == i].mean() for i in range(n_groups)])
    group_means_X = np.column_stack([group_means_x1, group_means_x2])
    alpha_hat = group_means_y - group_means_X @ beta_oracle
    sigma2_u = float(np.var(alpha_hat, ddof=1))
    sigma_u = np.sqrt(sigma2_u)
    sigma_e = np.sqrt(sigma2_e)
    rho = sigma2_u / (sigma2_u + sigma2_e)

    return {
        "df": df,
        "beta_oracle": beta_oracle,
        "se_oracle": se_oracle,
        "t_oracle": t_oracle,
        "SSR_w": SSR_w,
        "sigma_u": sigma_u,
        "sigma_e": sigma_e,
        "rho": rho,
        "n_groups": n_groups,
        "N": N,
        "T": T,
        "k": k,
        "df_resid": df_resid,
    }


# ============ FE (Within Estimator) ============

def test_fe_coefficients_match_oracle(fe_data):
    """FE 斜率系数应与 numpy within oracle 一致（含 _cons）"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    k = fe_data["k"]
    assert np.allclose(result.coef[:k], fe_data["beta_oracle"], atol=1e-8)


def test_fe_std_errors_match_oracle(fe_data):
    """FE 斜率标准误应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    k = fe_data["k"]
    assert np.allclose(result.std_err[:k], fe_data["se_oracle"], atol=1e-8)


def test_fe_t_stats_match_oracle(fe_data):
    """FE 斜率 t 统计量应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    k = fe_data["k"]
    assert np.allclose(result.t_stat[:k], fe_data["t_oracle"], atol=1e-6)


def test_fe_sigma_u(fe_data):
    """FE sigma_u 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    assert np.allclose(result.sigma_u, fe_data["sigma_u"], atol=1e-8)


def test_fe_sigma_e(fe_data):
    """FE sigma_e 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    assert np.allclose(result.sigma_e, fe_data["sigma_e"], atol=1e-8)


def test_fe_rho(fe_data):
    """FE rho 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    assert np.allclose(result.rho, fe_data["rho"], atol=1e-8)


def test_fe_df_resid(fe_data):
    """FE df_resid = N - n_groups - k"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    assert result.df_resid == fe_data["df_resid"]


def test_fe_var_names_has_cons(fe_data):
    """FE 结果应含 _cons（平均固定效应）"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        fe_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="fe", is_con=True,
    )
    assert "_cons" in result.var_names


def test_xtset_then_xtreg(fe_data):
    """通过 TabraData 的 xtset + xtreg 调用"""
    from tabra.core.data import TabraData

    tab = TabraData(fe_data["df"], is_display_result=False)
    tab.xtset("group")
    result = tab.xtreg("y", ["x1", "x2"], model="fe")
    k = fe_data["k"]
    assert np.allclose(result.coef[:k], fe_data["beta_oracle"], atol=1e-8)


def test_xtreg_without_xtset_raises(fe_data):
    """未调用 xtset 就调用 xtreg 应报错"""
    from tabra.core.data import TabraData

    tab = TabraData(fe_data["df"])
    with pytest.raises(ValueError, match="xtset"):
        tab.xtreg("y", ["x1", "x2"], model="fe")


# ============ BE (Between Estimator) ============

@pytest.fixture
def be_data():
    """Balanced panel for BE: oracle = OLS on group means."""
    np.random.seed(42)
    n_groups = 20
    T = 5
    N = n_groups * T

    group_id = np.repeat(np.arange(n_groups), T)
    alpha = np.random.randn(n_groups) * 2
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    noise = np.random.randn(N) * 0.5
    y = alpha[group_id] + 2 * x1 + 1.5 * x2 + noise

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group_id})

    # Oracle: OLS on group means (with constant)
    gm_y = np.array([y[group_id == i].mean() for i in range(n_groups)])
    gm_x1 = np.array([x1[group_id == i].mean() for i in range(n_groups)])
    gm_x2 = np.array([x2[group_id == i].mean() for i in range(n_groups)])
    X_b = np.column_stack([np.ones(n_groups), gm_x1, gm_x2])
    beta_oracle = np.linalg.lstsq(X_b, gm_y, rcond=None)[0]
    resid_b = gm_y - X_b @ beta_oracle
    SSR_b = float(resid_b @ resid_b)
    k_b = X_b.shape[1]
    df_resid_b = n_groups - k_b
    sigma2_b = SSR_b / df_resid_b
    XtX_inv = np.linalg.inv(X_b.T @ X_b)
    se_oracle = np.sqrt(sigma2_b * np.diag(XtX_inv))
    t_oracle = beta_oracle / se_oracle

    SST_b = float((gm_y - gm_y.mean()) @ (gm_y - gm_y.mean()))
    SSE_b = SST_b - SSR_b
    r2 = 1 - SSR_b / SST_b if SST_b > 0 else 0.0
    r2_adj = 1 - (1 - r2) * (n_groups - 1) / df_resid_b if df_resid_b > 0 else 0.0
    df_model_b = k_b - 1
    f_stat = (SSE_b / df_model_b) / (SSR_b / df_resid_b) if df_model_b > 0 else 0.0

    return {
        "df": df,
        "beta_oracle": beta_oracle,
        "se_oracle": se_oracle,
        "t_oracle": t_oracle,
        "r2": r2,
        "r2_adj": r2_adj,
        "f_stat": f_stat,
        "n_groups": n_groups,
        "df_resid": df_resid_b,
    }


def test_be_coefficients_match_oracle(be_data):
    """BE 系数应与 OLS on group means oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        be_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="be", is_con=True,
    )
    assert np.allclose(result.coef, be_data["beta_oracle"], atol=1e-8)


def test_be_std_errors_match_oracle(be_data):
    """BE 标准误应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        be_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="be", is_con=True,
    )
    assert np.allclose(result.std_err, be_data["se_oracle"], atol=1e-8)


def test_be_t_stats_match_oracle(be_data):
    """BE t 统计量应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        be_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="be", is_con=True,
    )
    assert np.allclose(result.t_stat, be_data["t_oracle"], atol=1e-6)


def test_be_r_squared(be_data):
    """BE R-squared 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        be_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="be", is_con=True,
    )
    assert np.allclose(result.r_squared, be_data["r2"], atol=1e-10)


def test_be_var_names_has_cons(be_data):
    """BE 结果应含 _cons"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        be_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="be", is_con=True,
    )
    assert "_cons" in result.var_names


def test_be_df_resid(be_data):
    """BE df_resid = n_groups - k"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        be_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="be", is_con=True,
    )
    assert result.df_resid == be_data["df_resid"]


# ============ RE (Random Effects GLS) ============

@pytest.fixture
def re_data():
    """Panel data with RE oracle via manual theta transformation."""
    np.random.seed(42)
    n_groups = 20
    T = 5
    N = n_groups * T

    group_id = np.repeat(np.arange(n_groups), T)
    alpha = np.random.randn(n_groups) * 2  # random effects
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    noise = np.random.randn(N) * 0.5
    y = alpha[group_id] + 2 * x1 + 1.5 * x2 + noise

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group_id})

    # --- Within (FE) variance components ---
    y_arr = y
    X_arr = np.column_stack([x1, x2])
    k = 2

    y_dm = np.zeros(N)
    X_dm = np.zeros((N, k))
    for i in range(n_groups):
        mask = group_id == i
        y_dm[mask] = y_arr[mask] - y_arr[mask].mean()
        for j in range(k):
            X_dm[mask, j] = X_arr[mask, j] - X_arr[mask, j].mean()

    beta_fe = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    resid_w = y_dm - X_dm @ beta_fe
    SSR_w = float(resid_w @ resid_w)
    df_w = N - n_groups - k
    sigma2_e = SSR_w / df_w

    # Between variance
    gm_y = np.array([y_arr[group_id == i].mean() for i in range(n_groups)])
    gm_X = np.array([X_arr[group_id == i].mean(axis=0) for i in range(n_groups)])
    X_b = np.column_stack([np.ones(n_groups), gm_X])
    beta_b = np.linalg.lstsq(X_b, gm_y, rcond=None)[0]
    resid_b = gm_y - X_b @ beta_b
    SSR_b = float(resid_b @ resid_b)
    sigma2_u = max(0.0, SSR_b / (n_groups - k - 1) - sigma2_e / T)

    # Theta (Swamy-Arora)
    theta = 1.0 - np.sqrt(sigma2_e / (sigma2_e + T * sigma2_u))

    # Quasi-demeaned data
    y_qd = np.zeros(N)
    X_qd = np.zeros((N, k))
    for i in range(n_groups):
        mask = group_id == i
        y_qd[mask] = y_arr[mask] - theta * gm_y[i]
        for j in range(k):
            X_qd[mask, j] = X_arr[mask, j] - theta * gm_X[i, j]

    X_qd_full = np.column_stack([(1 - theta) * np.ones(N), X_qd])
    beta_re = np.linalg.lstsq(X_qd_full, y_qd, rcond=None)[0]
    resid_re = y_qd - X_qd_full @ beta_re
    SSR_re = float(resid_re @ resid_re)
    k_full = k + 1
    df_resid_re = N - k_full
    sigma2_re = SSR_re / df_resid_re
    XtX_inv = np.linalg.inv(X_qd_full.T @ X_qd_full)
    se_re = np.sqrt(sigma2_re * np.diag(XtX_inv))
    t_re = beta_re / se_re

    SST_re = float((y_qd - y_qd.mean()) @ (y_qd - y_qd.mean()))
    SSE_re = SST_re - SSR_re
    r2_re = 1 - SSR_re / SST_re if SST_re > 0 else 0.0

    return {
        "df": df,
        "beta_oracle": beta_re,
        "se_oracle": se_re,
        "t_oracle": t_re,
        "r2": r2_re,
        "theta": theta,
        "sigma_u": np.sqrt(sigma2_u),
        "sigma_e": np.sqrt(sigma2_e),
    }


def test_re_coefficients_match_oracle(re_data):
    """RE 系数应与 theta 变换 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert np.allclose(result.coef, re_data["beta_oracle"], atol=1e-8)


def test_re_std_errors_match_oracle(re_data):
    """RE 标准误应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert np.allclose(result.std_err, re_data["se_oracle"], atol=1e-8)


def test_re_t_stats_match_oracle(re_data):
    """RE t 统计量应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert np.allclose(result.t_stat, re_data["t_oracle"], atol=1e-6)


def test_re_r_squared(re_data):
    """RE R-squared 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert np.allclose(result.r_squared, re_data["r2"], atol=1e-10)


def test_re_theta(re_data):
    """RE theta 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert np.allclose(result.theta, re_data["theta"], atol=1e-8)


def test_re_sigma_u(re_data):
    """RE sigma_u 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert np.allclose(result.sigma_u, re_data["sigma_u"], atol=1e-8)


def test_re_var_names_has_cons(re_data):
    """RE 结果应含 _cons"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        re_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="re", is_con=True,
    )
    assert "_cons" in result.var_names


# ============ MLE (Maximum Likelihood RE) ============

@pytest.fixture
def mle_data():
    """Panel data with MLE oracle via Baltagi (2021) profile log-likelihood."""
    np.random.seed(42)
    n_groups = 20
    T = 5
    N = n_groups * T

    group_id = np.repeat(np.arange(n_groups), T)
    alpha = np.random.randn(n_groups) * 2
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    noise = np.random.randn(N) * 0.5
    y = alpha[group_id] + 2 * x1 + 1.5 * x2 + noise

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group_id})
    X_arr = np.column_stack([x1, x2])
    k = 2

    # Pre-compute per-group data
    T_i = np.full(n_groups, T)  # balanced panel
    gm_y = np.array([y[group_id == i].mean() for i in range(n_groups)])
    gm_X = np.array([X_arr[group_id == i].mean(axis=0) for i in range(n_groups)])
    group_indices = [np.where(group_id == i)[0] for i in range(n_groups)]

    from scipy.optimize import minimize

    def _gls_solution(sigma2_u_val, sigma2_e_val):
        theta_i = 1.0 - np.sqrt(sigma2_e_val / (sigma2_e_val + T_i * sigma2_u_val))
        y_qd = np.zeros(N)
        X_qd = np.zeros((N, k))
        for idx in range(n_groups):
            gi = group_indices[idx]
            y_qd[gi] = y[gi] - theta_i[idx] * gm_y[idx]
            for j in range(k):
                X_qd[gi, j] = X_arr[gi, j] - theta_i[idx] * gm_X[idx, j]
        ones_qd = (1 - theta_i[0]) * np.ones(N)  # balanced, same theta
        X_qd_full = np.column_stack([ones_qd, X_qd])
        beta = np.linalg.lstsq(X_qd_full, y_qd, rcond=None)[0]
        return beta, X_qd_full, y_qd

    def _neg_profile_loglik(params):
        log_su, log_se = params
        sigma2_u_val = np.exp(log_su)
        sigma2_e_val = np.exp(log_se)
        if sigma2_u_val < 1e-20 or sigma2_e_val < 1e-20:
            return 1e20
        beta, _, _ = _gls_solution(sigma2_u_val, sigma2_e_val)
        total_loglik = 0.0
        for idx in range(n_groups):
            gi = group_indices[idx]
            Ti = T_i[idx]
            yi = y[gi]
            Xi = np.column_stack([np.ones(Ti), X_arr[gi]])
            resid_i = yi - Xi @ beta
            log_det = (Ti - 1) * np.log(sigma2_e_val) + np.log(sigma2_e_val + Ti * sigma2_u_val)
            phi_i = sigma2_u_val / (sigma2_e_val + Ti * sigma2_u_val)
            resid_sum = resid_i.sum()
            quad_form = (resid_i @ resid_i - phi_i * resid_sum ** 2) / sigma2_e_val
            total_loglik += -Ti / 2 * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad_form
        return -total_loglik

    # Initial values from Swamy-Arora
    y_dm = np.zeros(N)
    X_dm = np.zeros((N, k))
    for i in range(n_groups):
        mask = group_id == i
        y_dm[mask] = y[mask] - y[mask].mean()
        for j in range(k):
            X_dm[mask, j] = X_arr[mask, j] - X_arr[mask, j].mean()
    beta_fe = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    resid_w = y_dm - X_dm @ beta_fe
    SSR_w = float(resid_w @ resid_w)
    df_w = N - n_groups - k
    sigma2_e_init = SSR_w / df_w
    resid_b = gm_y - gm_X @ beta_fe
    SSR_b = float(resid_b @ resid_b)
    T_bar_harm = n_groups / np.sum(1.0 / T_i)
    sigma2_u_init = max(0.01, SSR_b / (n_groups - k - 1) - sigma2_e_init / T_bar_harm)

    x0 = [np.log(sigma2_u_init), np.log(sigma2_e_init)]
    opt = minimize(_neg_profile_loglik, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})

    sigma2_u_final = np.exp(opt.x[0])
    sigma2_e_final = np.exp(opt.x[1])

    # Final GLS with optimal variance components
    beta_mle, X_qd_full, y_qd = _gls_solution(sigma2_u_final, sigma2_e_final)
    resid_mle = y_qd - X_qd_full @ beta_mle
    SSR = float(resid_mle @ resid_mle)
    k_full = k + 1
    df_resid = N - k_full

    # MLE SE from numerical Hessian of full log-likelihood (matches Stata)
    X_full_arr = np.column_stack([np.ones(N), X_arr])

    def _full_loglik(params_all):
        beta_p = params_all[:k_full]
        log_su_p = params_all[k_full]
        log_se_p = params_all[k_full + 1]
        s2u = np.exp(log_su_p)
        s2e = np.exp(log_se_p)
        if s2u < 1e-20 or s2e < 1e-20:
            return -1e20
        total_ll = 0.0
        for idx in range(n_groups):
            gi = group_indices[idx]
            Ti = T_i[idx]
            yi = y[gi]
            Xi = X_full_arr[gi]
            resid_i = yi - Xi @ beta_p
            log_det = (Ti - 1) * np.log(s2e) + np.log(s2e + Ti * s2u)
            phi_i = s2u / (s2e + Ti * s2u)
            resid_sum = resid_i.sum()
            quad = (resid_i @ resid_i - phi_i * resid_sum ** 2) / s2e
            total_ll += -Ti / 2 * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad
        return total_ll

    params_mle = np.concatenate([beta_mle, [np.log(sigma2_u_final), np.log(sigma2_e_final)]])
    n_p = len(params_mle)
    h_step = 1e-5
    H_mat = np.zeros((n_p, n_p))
    for ii in range(n_p):
        for jj in range(ii, n_p):
            pp = params_mle.copy(); pp[ii] += h_step; pp[jj] += h_step
            pm = params_mle.copy(); pm[ii] += h_step; pm[jj] -= h_step
            mp = params_mle.copy(); mp[ii] -= h_step; mp[jj] += h_step
            mm = params_mle.copy(); mm[ii] -= h_step; mm[jj] -= h_step
            H_mat[ii, jj] = (_full_loglik(pp) - _full_loglik(pm)
                             - _full_loglik(mp) + _full_loglik(mm)) / (4 * h_step ** 2)
            H_mat[jj, ii] = H_mat[ii, jj]
    I_obs = -H_mat
    V_all = np.linalg.inv(I_obs)
    se_mle = np.sqrt(np.diag(V_all[:k_full, :k_full]))
    t_mle = beta_mle / se_mle
    theta_opt = 1.0 - np.sqrt(sigma2_e_final / (sigma2_e_final + np.mean(T_i) * sigma2_u_final))

    return {
        "df": df,
        "beta_oracle": beta_mle,
        "se_oracle": se_mle,
        "t_oracle": t_mle,
        "theta": theta_opt,
        "sigma_u": np.sqrt(sigma2_u_final),
        "sigma_e": np.sqrt(sigma2_e_final),
    }


def test_mle_coefficients_match_oracle(mle_data):
    """MLE 系数应与 concentrated log-lik oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        mle_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="mle", is_con=True,
    )
    assert np.allclose(result.coef, mle_data["beta_oracle"], atol=1e-6)


def test_mle_std_errors_match_oracle(mle_data):
    """MLE 标准误应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        mle_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="mle", is_con=True,
    )
    assert np.allclose(result.std_err, mle_data["se_oracle"], atol=1e-4)


def test_mle_t_stats_match_oracle(mle_data):
    """MLE t 统计量应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        mle_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="mle", is_con=True,
    )
    assert np.allclose(result.t_stat, mle_data["t_oracle"], atol=1e-4)


def test_mle_theta(mle_data):
    """MLE theta 应与 oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        mle_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="mle", is_con=True,
    )
    assert np.allclose(result.theta, mle_data["theta"], atol=1e-6)


def test_mle_var_names_has_cons(mle_data):
    """MLE 结果应含 _cons"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        mle_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="mle", is_con=True,
    )
    assert "_cons" in result.var_names


# ============ PA (Population-Averaged GEE) ============

@pytest.fixture
def pa_data():
    """Panel data with PA oracle via manual GEE iteration."""
    np.random.seed(42)
    n_groups = 20
    T = 5
    N = n_groups * T

    group_id = np.repeat(np.arange(n_groups), T)
    alpha = np.random.randn(n_groups) * 2
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    noise = np.random.randn(N) * 0.5
    y = alpha[group_id] + 2 * x1 + 1.5 * x2 + noise

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group_id})
    X_full = np.column_stack([np.ones(N), x1, x2])
    k_full = 3

    # Oracle: GEE with exchangeable correlation
    group_indices = [np.where(group_id == i)[0] for i in range(n_groups)]
    T_i = np.full(n_groups, T)
    beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
    for _ in range(50):
        resid = y - X_full @ beta
        phi = float(resid @ resid) / N
        # Estimate exchangeable alpha
        num = 0.0
        den = 0.0
        for i in range(n_groups):
            ri = resid[group_indices[i]]
            Ti = len(ri)
            ri_sum = ri.sum()
            ri_sq_sum = (ri ** 2).sum()
            num += (ri_sum ** 2 - ri_sq_sum) / 2.0
            den += Ti * (Ti - 1) / 2.0
        alpha_corr = num / den / phi if den > 0 and phi > 0 else 0.0
        T_max = T_i.max()
        alpha_corr = max(-1.0 / (T_max - 1) if T_max > 1 else 0.0,
                         min(alpha_corr, 0.9999))

        # GEE normal equation: sum_i X_i' [I - rho_inv * J] X_i
        H = np.zeros((k_full, k_full))
        g = np.zeros(k_full)
        for i in range(n_groups):
            Xi = X_full[group_indices[i]]
            yi = y[group_indices[i]]
            Ti = len(Xi)
            rho_inv_i = alpha_corr / (1 - alpha_corr + Ti * alpha_corr)
            si = Xi.sum(axis=0)
            yi_sum = yi.sum()
            H += Xi.T @ Xi - rho_inv_i * np.outer(si, si)
            g += Xi.T @ yi - rho_inv_i * si * yi_sum

        beta_new = np.linalg.solve(H, g)
        tol = np.max(np.abs(beta_new - beta) / np.maximum(1.0, np.abs(beta)))
        if tol <= 1e-6:
            beta = beta_new
            break
        beta = beta_new

    # Final stats
    resid_final = y - X_full @ beta
    sigma2_final = float(resid_final @ resid_final) / N

    return {
        "df": df,
        "beta_oracle": beta,
    }


def test_pa_coefficients_match_oracle(pa_data):
    """PA 系数应与 GEE oracle 一致"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        pa_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="pa", is_con=True,
    )
    assert np.allclose(result.coef, pa_data["beta_oracle"], atol=1e-4)


def test_pa_var_names_has_cons(pa_data):
    """PA 结果应含 _cons"""
    from tabra.models.estimate.panel import PanelModel

    model = PanelModel()
    result = model.fit(
        pa_data["df"], y="y", x=["x1", "x2"],
        panel_var="group", model="pa", is_con=True,
    )
    assert "_cons" in result.var_names
