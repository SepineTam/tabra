#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_tobit.py

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────
# Oracle: scipy-based tobit MLE
# ─────────────────────────────────────────────────────────

def _scipy_tobit_oracle(y, X, ll=None, ul=None):
    """Tobit MLE oracle using scipy.optimize.minimize."""
    n, k = X.shape

    def neg_log_lik(params):
        beta = params[:k]
        ln_sigma = params[k]
        sigma = np.exp(ln_sigma)
        xb = X @ beta

        ll_val = 0.0
        for i in range(n):
            if ll is not None and y[i] <= ll:
                # 左删失
                z = (ll - xb[i]) / sigma
                z = np.clip(z, -30, 30)
                ll_val += np.log(sp_stats.norm.cdf(z) + 1e-300)
            elif ul is not None and y[i] >= ul:
                # 右删失
                z = (ul - xb[i]) / sigma
                z = np.clip(z, -30, 30)
                ll_val += np.log(1 - sp_stats.norm.cdf(z) + 1e-300)
            else:
                # 无删失
                ll_val += sp_stats.norm.logpdf(y[i], loc=xb[i], scale=sigma)

        return -ll_val

    # 初始值: OLS
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_init
    sigma_init = np.std(resid, ddof=X.shape[1])
    sigma_init = max(sigma_init, 0.1)
    params_init = np.append(beta_init, np.log(sigma_init))

    res = minimize(neg_log_lik, params_init, method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})

    beta = res.x[:k]
    sigma = np.exp(res.x[k])

    # 数值 Hessian 求标准误
    eps = 1e-5
    p = len(res.x)
    hess = np.zeros((p, p))
    f0 = neg_log_lik(res.x)
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p)
            ej = np.zeros(p)
            ei[i] = eps
            ej[j] = eps
            hess[i, j] = (neg_log_lik(res.x + ei + ej) -
                          neg_log_lik(res.x + ei - ej) -
                          neg_log_lik(res.x - ei + ej) +
                          neg_log_lik(res.x - ei - ej)) / (4 * eps * eps)
            hess[j, i] = hess[i, j]

    try:
        V = np.linalg.inv(hess)
        se = np.sqrt(np.abs(np.diag(V)))
    except np.linalg.LinAlgError:
        se = np.ones(p) * np.nan

    return beta, sigma, se[:k], se[k], -res.fun


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def auto_data():
    """Stata auto dataset loaded from web."""
    try:
        import urllib.request
        import tempfile
        import os
        url = "https://www.stata-press.com/data/r19/auto2.dta"
        tmp = tempfile.NamedTemporaryFile(suffix=".dta", delete=False)
        try:
            urllib.request.urlretrieve(url, tmp.name)
            df = pd.read_stata(tmp.name)
        finally:
            os.unlink(tmp.name)
        df["wgt"] = df["weight"] / 1000
        return df
    except Exception:
        pytest.skip("Cannot download Stata auto dataset")


@pytest.fixture
def synthetic_tobit_data():
    """Synthetic left-censored data with known oracle."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    sigma_true = 2.0
    y_star = 1.0 + 0.5 * x1 - 0.3 * x2 + sigma_true * np.random.randn(n)
    ll = 0.0
    y = np.maximum(y_star, ll)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    X = np.column_stack([x1, x2, np.ones(n)])
    coef, sigma, se_beta, se_sigma, ll_val = _scipy_tobit_oracle(y, X, ll=ll)

    return {
        "df": df,
        "ll": ll,
        "ul": None,
        "coef": coef,
        "sigma": sigma,
        "se_beta": se_beta,
        "se_sigma": se_sigma,
        "ll_val": ll_val,
    }


@pytest.fixture
def synthetic_right_censored_data():
    """Synthetic right-censored data."""
    np.random.seed(123)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    sigma_true = 1.5
    y_star = 2.0 + 1.0 * x1 + 0.5 * x2 + sigma_true * np.random.randn(n)
    ul = 5.0
    y = np.minimum(y_star, ul)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    X = np.column_stack([x1, x2, np.ones(n)])
    coef, sigma, se_beta, se_sigma, ll_val = _scipy_tobit_oracle(y, X, ul=ul)

    return {
        "df": df,
        "ll": None,
        "ul": ul,
        "coef": coef,
        "sigma": sigma,
        "se_beta": se_beta,
        "se_sigma": se_sigma,
        "ll_val": ll_val,
    }


@pytest.fixture
def synthetic_two_limit_data():
    """Synthetic two-limit censored data."""
    np.random.seed(456)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    sigma_true = 1.5
    y_star = 1.0 + 0.8 * x1 - 0.4 * x2 + sigma_true * np.random.randn(n)
    ll_val = -1.0
    ul_val = 3.0
    y = np.clip(y_star, ll_val, ul_val)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    X = np.column_stack([x1, x2, np.ones(n)])
    coef, sigma, se_beta, se_sigma, loglik = _scipy_tobit_oracle(
        y, X, ll=ll_val, ul=ul_val)

    return {
        "df": df,
        "ll": ll_val,
        "ul": ul_val,
        "coef": coef,
        "sigma": sigma,
        "se_beta": se_beta,
        "se_sigma": se_sigma,
        "ll_val": loglik,
    }


# ─────────────────────────────────────────────────────────
# Stata comparison values (from actual Stata runs)
# ─────────────────────────────────────────────────────────

@pytest.fixture
def stata_left_censored():
    """tobit mpg wgt, ll(17) on auto data"""
    return {
        "coef": np.array([-6.87305, 41.49856]),
        "var_e": 14.78942,
        "ll": -164.25438,
        "N_unc": 56,
        "N_lc": 18,
        "N_rc": 0,
        "chi2": 72.85,
        "pseudo_r2": 0.1815,
    }


@pytest.fixture
def stata_right_censored():
    """tobit mpg wgt, ul(24) on auto data"""
    return {
        "coef": np.array([-5.080645, 36.08037]),
        "var_e": 5.689927,
        "ll": -129.8279,
        "N_unc": 51,
        "N_lc": 0,
        "N_rc": 23,
        "chi2": 90.72,
        "pseudo_r2": 0.2589,
    }


@pytest.fixture
def stata_two_limit():
    """tobit mpg wgt, ll(17) ul(24) on auto data"""
    return {
        "coef": np.array([-5.764448, 38.07469]),
        "var_e": 8.330943,
        "ll": -104.25976,
        "N_unc": 33,
        "N_lc": 18,
        "N_rc": 23,
        "chi2": 77.60,
        "pseudo_r2": 0.2712,
    }


@pytest.fixture
def stata_multi_var():
    """tobit mpg wgt length displacement, ll(17) on auto data"""
    return {
        "coef": np.array([-4.990222, -0.0751196, 0.0022593, 49.52994]),
        "var_e": 14.27064,
        "ll": -163.58407,
    }


# ─────────────────────────────────────────────────────────
# Tests: left-censored tobit
# ─────────────────────────────────────────────────────────

class TestTobitLeftCensored:

    def test_coefficients_vs_oracle(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert np.allclose(result.coef, d["coef"], atol=0.05)

    def test_sigma_vs_oracle(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert np.isclose(result.sigma, d["sigma"], atol=0.1)

    def test_stata_coefficients(self, auto_data, stata_left_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_left_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17)

        assert np.allclose(result.coef, sv["coef"], atol=0.01)

    def test_stata_sigma(self, auto_data, stata_left_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_left_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17)

        expected_sigma = np.sqrt(sv["var_e"])
        assert np.isclose(result.sigma, expected_sigma, atol=0.05)

    def test_stata_log_likelihood(self, auto_data, stata_left_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_left_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17)

        assert np.isclose(result.ll, sv["ll"], atol=0.1)

    def test_stata_censoring_counts(self, auto_data, stata_left_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_left_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17)

        assert result.n_unc == sv["N_unc"]
        assert result.n_lc == sv["N_lc"]
        assert result.n_rc == sv["N_rc"]

    def test_stata_chi2(self, auto_data, stata_left_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_left_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17)

        assert np.isclose(result.chi2, sv["chi2"], atol=0.5)

    def test_stata_pseudo_r2(self, auto_data, stata_left_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_left_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17)

        assert np.isclose(result.pseudo_r2, sv["pseudo_r2"], atol=0.005)


# ─────────────────────────────────────────────────────────
# Tests: right-censored tobit
# ─────────────────────────────────────────────────────────

class TestTobitRightCensored:

    def test_stata_coefficients(self, auto_data, stata_right_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_right_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ul=24)

        assert np.allclose(result.coef, sv["coef"], atol=0.01)

    def test_stata_sigma(self, auto_data, stata_right_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_right_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ul=24)

        expected_sigma = np.sqrt(sv["var_e"])
        assert np.isclose(result.sigma, expected_sigma, atol=0.05)

    def test_stata_censoring_counts(self, auto_data, stata_right_censored):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_right_censored
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ul=24)

        assert result.n_unc == sv["N_unc"]
        assert result.n_lc == sv["N_lc"]
        assert result.n_rc == sv["N_rc"]


# ─────────────────────────────────────────────────────────
# Tests: two-limit tobit
# ─────────────────────────────────────────────────────────

class TestTobitTwoLimit:

    def test_stata_coefficients(self, auto_data, stata_two_limit):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_two_limit
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17, ul=24)

        assert np.allclose(result.coef, sv["coef"], atol=0.01)

    def test_stata_sigma(self, auto_data, stata_two_limit):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_two_limit
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17, ul=24)

        expected_sigma = np.sqrt(sv["var_e"])
        assert np.isclose(result.sigma, expected_sigma, atol=0.05)

    def test_stata_censoring_counts(self, auto_data, stata_two_limit):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_two_limit
        model = TobitModel()
        result = model.fit(auto_data, y="mpg", x=["wgt"], ll=17, ul=24)

        assert result.n_unc == sv["N_unc"]
        assert result.n_lc == sv["N_lc"]
        assert result.n_rc == sv["N_rc"]

    def test_vs_oracle(self, synthetic_two_limit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_two_limit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"],
                           ll=d["ll"], ul=d["ul"])

        assert np.allclose(result.coef, d["coef"], atol=0.05)


# ─────────────────────────────────────────────────────────
# Tests: multi-var tobit
# ─────────────────────────────────────────────────────────

class TestTobitMultiVar:

    def test_stata_coefficients(self, auto_data, stata_multi_var):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_multi_var
        model = TobitModel()
        result = model.fit(auto_data, y="mpg",
                           x=["wgt", "length", "displacement"], ll=17)

        assert np.allclose(result.coef, sv["coef"], atol=0.01)

    def test_stata_log_likelihood(self, auto_data, stata_multi_var):
        from tabra.models.estimate.tobit import TobitModel
        sv = stata_multi_var
        model = TobitModel()
        result = model.fit(auto_data, y="mpg",
                           x=["wgt", "length", "displacement"], ll=17)

        assert np.isclose(result.ll, sv["ll"], atol=0.1)


# ─────────────────────────────────────────────────────────
# Tests: general properties
# ─────────────────────────────────────────────────────────

class TestTobitGeneral:

    def test_var_names_with_constant(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert result.var_names == ["x1", "x2", "_cons"]

    def test_var_names_noconstant(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"],
                           ll=d["ll"], is_con=False)

        assert result.var_names == ["x1", "x2"]

    def test_n_obs(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert result.n_obs == 300

    def test_convergence(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert result.converged

    def test_pseudo_r2_range(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert 0 <= result.pseudo_r2 <= 1

    def test_chi2_positive(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        assert result.chi2 > 0

    def test_t_stat_computation(self, synthetic_tobit_data):
        from tabra.models.estimate.tobit import TobitModel
        d = synthetic_tobit_data
        model = TobitModel()
        result = model.fit(d["df"], y="y", x=["x1", "x2"], ll=d["ll"])

        expected_t = result.coef / result.std_err
        assert np.allclose(result.t_stat, expected_t, atol=1e-8)


# ─────────────────────────────────────────────────────────
# Tests: TabraData integration
# ─────────────────────────────────────────────────────────

class TestTobitTabraDataIntegration:

    def test_tabradata_tobit(self, synthetic_tobit_data):
        from tabra.core.data import TabraData
        d = synthetic_tobit_data
        tab = TabraData(d["df"], is_display_result=False)
        result = tab.est.tobit("y", ["x1", "x2"], ll=d["ll"])

        assert result is not None
        assert result.n_obs == 300
        assert len(result.coef) == 3

    def test_tabradata_tobit_two_limit(self, synthetic_two_limit_data):
        from tabra.core.data import TabraData
        d = synthetic_two_limit_data
        tab = TabraData(d["df"], is_display_result=False)
        result = tab.est.tobit("y", ["x1", "x2"], ll=d["ll"], ul=d["ul"])

        assert result is not None
        assert result.n_obs == 300

    def test_summary_output(self, synthetic_tobit_data):
        from tabra.core.data import TabraData
        d = synthetic_tobit_data
        tab = TabraData(d["df"], is_display_result=False)
        result = tab.est.tobit("y", ["x1", "x2"], ll=d["ll"])
        summary = result.summary()

        assert "Tobit regression" in summary
        assert "Number of obs" in summary
        assert "Uncensored" in summary
        assert "Left-censored" in summary
        assert "Pseudo R2" in summary
