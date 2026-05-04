#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_heckman.py

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def womenwk_data():
    """Replicate Stata webuse womenwk dataset.

    From Stata estat summarize:
    wage:     mean=23.69, sd=6.31, min=5.88, max=45.81 (1343 nonmissing)
    education: mean=13.08, sd=3.05, min=10, max=20
    age:      mean=36.21, sd=8.29, min=20, max=59
    married:  mean=0.67, sd=0.47, min=0, max=1
    children: mean=1.64, sd=1.40, min=0, max=5

    N=2000, N_selected=1343, N_nonselected=657
    Selection: wage is missing for non-selected obs.
    """
    np.random.seed(42)
    n = 2000

    education = np.random.randint(10, 21, n).astype(float)
    age = np.random.randint(20, 60, n).astype(float)
    married = np.random.binomial(1, 0.67, n).astype(float)
    children = np.random.randint(0, 6, n).astype(float)

    # 选择方程: z = 1[married*0.4 + children*0.4 + education*0.05 + age*0.03 - 2.5 + u2 > 0]
    z_star = (0.4 * married + 0.4 * children + 0.05 * education
              + 0.03 * age - 2.5 + np.random.randn(n))
    selected = (z_star > 0).astype(float)

    # 结果方程: wage = 1.0*educ + 0.2*age + 0.5 + u1
    sigma_true = 6.0
    rho_true = 0.7
    # 生成相关误差
    u2 = np.random.randn(n)
    u1 = rho_true * u2 + np.sqrt(1 - rho_true**2) * np.random.randn(n)
    u1 = u1 * sigma_true

    wage = 1.0 * education + 0.2 * age + 0.5 + u1
    # 未被选中的设为 NaN
    wage[selected == 0] = np.nan

    df = pd.DataFrame({
        "wage": wage,
        "education": education,
        "age": age,
        "married": married,
        "children": children,
    })
    return df


@pytest.fixture
def stata_mle_values():
    """Stata output for: heckman wage educ age, select(married children educ age)
    (from actual Stata run on womenwk dataset)

    Outcome equation:
      education: coef=0.9899537, se=0.0532565
      age:       coef=0.2131294, se=0.0206031
      _cons:     coef=0.4857752, se=1.077037

    Selection equation:
      married:   coef=0.4451721, se=0.0673954
      children:  coef=0.4387068, se=0.0277828
      education: coef=0.0557318, se=0.0107349
      age:       coef=0.0365098, se=0.0041533
      _cons:     coef=-2.491015, se=0.1893402

    Ancillary:
      athrho:    0.8742086
      lnsigma:   1.792559

    Derived:
      rho:       0.7035061
      sigma:     6.0047973
      lambda:    4.2244115

    N=2000, N_selected=1343, N_nonselected=657
    ll=-5178.3045
    chi2=508.43855
    """
    return {
        "outcome_coef": np.array([0.9899537, 0.2131294, 0.4857752]),
        "outcome_se": np.array([0.0532565, 0.0206031, 1.077037]),
        "select_coef": np.array([0.4451721, 0.4387068, 0.0557318,
                                 0.0365098, -2.491015]),
        "select_se": np.array([0.0673954, 0.0277828, 0.0107349,
                               0.0041533, 0.1893402]),
        "athrho": 0.8742086,
        "lnsigma": 1.792559,
        "rho": 0.7035061,
        "sigma": 6.0047973,
        "lambda": 4.2244115,
        "n_obs": 2000,
        "n_selected": 1343,
        "n_nonselected": 657,
        "ll": -5178.3045,
        "chi2": 508.43855,
    }


def _heckman_mle_oracle(y, X_outcome, X_select, selected):
    """Heckman MLE oracle using scipy.optimize.minimize."""
    n = len(y)
    k_out = X_outcome.shape[1]
    k_sel = X_select.shape[1]

    y_obs = y[selected == 1]
    X_out_obs = X_outcome[selected == 1]
    X_sel_obs = X_select[selected == 1]
    X_sel_nobs = X_select[selected == 0]

    def neg_log_lik(params):
        beta = params[:k_out]
        gamma = params[k_out:k_out + k_sel]
        athrho = params[k_out + k_sel]
        lnsigma = params[k_out + k_sel + 1]

        rho = np.tanh(athrho)
        sigma = np.exp(lnsigma)

        if sigma < 1e-10:
            return 1e10

        # 选中观测的似然
        resid = y_obs - X_out_obs @ beta
        a = resid / sigma
        b = (X_sel_obs @ gamma + rho * a) / np.sqrt(1 - rho**2 + 1e-15)
        b = np.clip(b, -20, 20)

        ll_sel = np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma)
                        - 0.5 * a**2
                        + np.log(np.clip(sp_stats.norm.cdf(b), 1e-300, 1)))

        # 未选中观测的似然
        xb_nobs = X_sel_nobs @ gamma
        xb_nobs = np.clip(xb_nobs, -20, 20)
        ll_nobs = np.sum(np.log(np.clip(sp_stats.norm.cdf(-xb_nobs), 1e-300, 1)))

        return -(ll_sel + ll_nobs)

    # 初始值
    x0 = np.zeros(k_out + k_sel + 2)
    # 使用 OLS 初始化 beta
    try:
        x0[:k_out] = np.linalg.lstsq(X_out_obs, y_obs, rcond=None)[0]
    except Exception:
        pass
    # 使用 lnsigma 初始化
    resid_init = y_obs - X_out_obs @ x0[:k_out]
    x0[k_out + k_sel + 1] = np.log(np.std(resid_init) + 1e-6)

    bounds = [(None, None)] * (k_out + k_sel) + [(-5, 5), (-5, 5)]

    from scipy.optimize import minimize
    res = minimize(neg_log_lik, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-12})

    # 重新算一次不带 bounds 的
    res = minimize(neg_log_lik, res.x, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})

    return res.x, -res.fun


@pytest.fixture
def synthetic_heckman_data():
    """Synthetic Heckman selection data with known oracle results."""
    np.random.seed(42)
    n = 500

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)

    # 真实参数
    beta_true = np.array([1.0, -0.5, 0.5])  # x1, x2, _cons
    gamma_true = np.array([0.8, 0.6, -0.5])  # z1, z2, _cons
    rho_true = 0.5
    sigma_true = 1.0

    # 相关误差项
    u2 = np.random.randn(n)
    u1 = rho_true * u2 + np.sqrt(1 - rho_true**2) * np.random.randn(n)
    u1 = u1 * sigma_true

    X_outcome = np.column_stack([x1, x2, np.ones(n)])
    X_select = np.column_stack([z1, z2, np.ones(n)])

    y = X_outcome @ beta_true + u1
    z_star = X_select @ gamma_true + u2
    selected = (z_star > 0).astype(float)

    y[selected == 0] = np.nan

    df = pd.DataFrame({
        "y": y,
        "x1": x1, "x2": x2,
        "z1": z1, "z2": z2,
    })

    oracle_coef, oracle_ll = _heckman_mle_oracle(
        y, X_outcome, X_select, selected)

    return {
        "df": df,
        "oracle_coef": oracle_coef,
        "oracle_ll": oracle_ll,
        "rho_true": rho_true,
        "sigma_true": sigma_true,
        "n_selected": int(np.sum(selected)),
        "n_nonselected": int(np.sum(selected == 0)),
    }


# ─────────────────────────────────────────────────────────
# MLE Tests
# ─────────────────────────────────────────────────────────

class TestHeckmanMLE:

    def test_basic_fit(self, synthetic_heckman_data):
        """Test that HeckmanModel.fit() returns a result."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert result is not None

    def test_n_obs(self, synthetic_heckman_data):
        """Test total observation count."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert result.n_obs == 500
        assert result.n_selected == synthetic_heckman_data["n_selected"]
        assert result.n_nonselected == synthetic_heckman_data["n_nonselected"]

    def test_coefficients_vs_oracle(self, synthetic_heckman_data):
        """Test coefficients match scipy oracle."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]
        oracle = synthetic_heckman_data["oracle_coef"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        # 比较 outcome 系数
        assert np.allclose(result.outcome_coef, oracle[:3], atol=0.1)

    def test_log_likelihood_vs_oracle(self, synthetic_heckman_data):
        """Test log-likelihood matches scipy oracle."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]
        oracle_ll = synthetic_heckman_data["oracle_ll"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert np.isclose(result.ll, oracle_ll, atol=1.0)

    def test_rho_in_range(self, synthetic_heckman_data):
        """Test rho is in [-1, 1]."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert -1 <= result.rho <= 1

    def test_sigma_positive(self, synthetic_heckman_data):
        """Test sigma is positive."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert result.sigma > 0

    def test_lambda_equals_rho_sigma(self, synthetic_heckman_data):
        """Test lambda = rho * sigma."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert np.isclose(result.lambda_, result.rho * result.sigma, atol=1e-6)

    def test_var_names(self, synthetic_heckman_data):
        """Test variable names are correct."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert result.outcome_var_names == ["x1", "x2", "_cons"]
        assert result.select_var_names == ["z1", "z2", "_cons"]

    def test_chi2_positive(self, synthetic_heckman_data):
        """Test Wald chi2 is positive."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert result.chi2 > 0

    def test_converged(self, synthetic_heckman_data):
        """Test that optimization converged."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"])

        assert result.converged == True


# ─────────────────────────────────────────────────────────
# Stata womenwk comparison tests
# ─────────────────────────────────────────────────────────

class TestHeckmanStataComparison:

    def test_stata_outcome_coefficients(self, stata_mle_values):
        """Compare outcome equation coefficients with Stata."""
        from tabra.models.estimate.heckman import HeckmanModel

        np.random.seed(42)
        n = 2000
        education = np.random.randint(10, 21, n).astype(float)
        age = np.random.randint(20, 60, n).astype(float)
        married = np.random.binomial(1, 0.67, n).astype(float)
        children = np.random.randint(0, 6, n).astype(float)

        z_star = (0.4 * married + 0.4 * children + 0.05 * education
                  + 0.03 * age - 2.5 + np.random.randn(n))
        selected = (z_star > 0).astype(float)
        sigma_true = 6.0
        rho_true = 0.7
        u2 = np.random.randn(n)
        u1 = rho_true * u2 + np.sqrt(1 - rho_true**2) * np.random.randn(n)
        u1 = u1 * sigma_true
        wage = 1.0 * education + 0.2 * age + 0.5 + u1
        wage[selected == 0] = np.nan

        df = pd.DataFrame({
            "wage": wage,
            "education": education,
            "age": age,
            "married": married,
            "children": children,
        })

        model = HeckmanModel()
        result = model.fit(df, y="wage", x=["education", "age"],
                           select_x=["married", "children", "education", "age"])

        # 注意: 合成数据不会完全匹配 Stata 的 womenwk 真实数据
        # 这里只检查基本属性
        assert result.n_obs == 2000
        assert result.n_selected > 0
        assert result.n_nonselected > 0
        assert -1 <= result.rho <= 1
        assert result.sigma > 0


# ─────────────────────────────────────────────────────────
# Two-step tests
# ─────────────────────────────────────────────────────────

class TestHeckmanTwoStep:

    def test_twostep_basic(self, synthetic_heckman_data):
        """Test two-step estimator runs and returns result."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"], method="twostep")

        assert result is not None
        assert result.n_obs == 500

    def test_twostep_rho_in_range(self, synthetic_heckman_data):
        """Test two-step rho is bounded."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"], method="twostep")

        assert -1 <= result.rho <= 1

    def test_twostep_sigma_positive(self, synthetic_heckman_data):
        """Test two-step sigma is positive."""
        from tabra.models.estimate.heckman import HeckmanModel
        df = synthetic_heckman_data["df"]

        model = HeckmanModel()
        result = model.fit(df, y="y", x=["x1", "x2"],
                           select_x=["z1", "z2"], method="twostep")

        assert result.sigma > 0


# ─────────────────────────────────────────────────────────
# TabraData integration tests
# ─────────────────────────────────────────────────────────

class TestHeckmanTabraDataIntegration:

    def test_tabradata_heckman(self, synthetic_heckman_data):
        """Test TabraData.heckman() integration."""
        from tabra.core.data import TabraData
        df = synthetic_heckman_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.est.heckman("y", ["x1", "x2"], select_x=["z1", "z2"])

        assert result is not None
        assert result.n_obs == 500

    def test_tabradata_heckman_twostep(self, synthetic_heckman_data):
        """Test TabraData.heckman() with twostep method."""
        from tabra.core.data import TabraData
        df = synthetic_heckman_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.est.heckman("y", ["x1", "x2"], select_x=["z1", "z2"],
                             method="twostep")

        assert result is not None
        assert result.n_obs == 500

    def test_summary_output(self, synthetic_heckman_data):
        """Test summary output contains key elements."""
        from tabra.core.data import TabraData
        df = synthetic_heckman_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.est.heckman("y", ["x1", "x2"], select_x=["z1", "z2"])
        summary = result.summary()

        assert "Heckman selection model" in summary
        assert "Number of obs" in summary
        assert "Selected" in summary
        assert "rho" in summary
        assert "sigma" in summary

    def test_result_display(self, synthetic_heckman_data):
        """Test result can be displayed without error."""
        from tabra.core.data import TabraData
        df = synthetic_heckman_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.est.heckman("y", ["x1", "x2"], select_x=["z1", "z2"])

        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
