#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_binary_choice.py

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def auto_data():
    """Replicate Stata auto dataset: sysuse auto, gen highprice = (price > 6000)"""
    # 使用 scipy.stats logistic probit 数据来模拟
    # 这里使用确定性的 auto 数据近似值来与 Stata 输出对照
    np.random.seed(123)
    n = 74
    weight = np.random.uniform(2000, 5000, n)
    mpg = np.random.uniform(12, 40, n)
    xb = 0.0002 * weight - 0.1 * mpg - 0.5
    p = 1 / (1 + np.exp(-xb))
    highprice = (p > 0.5).astype(float)

    df = pd.DataFrame({
        "highprice": highprice,
        "weight": weight,
        "mpg": mpg,
    })
    return df


@pytest.fixture
def probit_stata_values():
    """Stata output for: probit highprice weight mpg
    (from actual Stata run on auto dataset)

    Coefficients:  weight=0.0001935, mpg=-0.0961735, _cons=0.8621834
    Std errors:     weight=0.0003782, mpg=0.0636704, _cons=2.338241
    Log likelihood: -39.131679
    Pseudo R2:      0.1467
    LR chi2(2):     13.46
    N:              74
    """
    return {
        "coef": np.array([0.0001935, -0.0961735, 0.8621834]),
        "std_err": np.array([0.0003782, 0.0636704, 2.338241]),
        "ll": -39.131679,
        "pseudo_r2": 0.1467,
        "chi2": 13.46,
        "n_obs": 74,
        "df_m": 2,
    }


@pytest.fixture
def logit_stata_values():
    """Stata output for: logit highprice weight mpg
    (from actual Stata run on auto dataset)

    Coefficients:  weight=0.0003668, mpg=-0.1613354, _cons=1.290534
    Std errors:     weight=0.0006598, mpg=0.1112337, _cons=4.064619
    Log likelihood: -39.04196
    Pseudo R2:      0.1487
    LR chi2(2):     13.64
    N:              74
    """
    return {
        "coef": np.array([0.0003668, -0.1613354, 1.290534]),
        "std_err": np.array([0.0006598, 0.1112337, 4.064619]),
        "ll": -39.04196,
        "pseudo_r2": 0.1487,
        "chi2": 13.64,
        "n_obs": 74,
        "df_m": 2,
    }


@pytest.fixture
def auto_stata_data():
    """Exact Stata auto data (sysuse auto) for comparison with Stata output.
    We load the actual auto dataset values that Stata uses.
    """
    try:
        from tabra.io.importers import read_stata
        # 尝试加载 Stata 自带的 auto 数据
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
        df["highprice"] = (df["price"] > 6000).astype(float)
        return df
    except Exception:
        pytest.skip("Cannot download Stata auto dataset")


def _scipy_probit_oracle(y, X):
    """Probit MLE oracle using scipy.optimize.minimize"""
    n, k = X.shape

    def neg_log_lik(beta):
        xb = X @ beta
        # clamp to avoid log(0)
        xb = np.clip(xb, -20, 20)
        p = sp_stats.norm.cdf(xb)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return -ll

    def neg_score(beta):
        xb = X @ beta
        xb = np.clip(xb, -20, 20)
        p = sp_stats.norm.cdf(xb)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        phi = sp_stats.norm.pdf(xb)
        q = (y - p) * phi / (p * (1 - p))
        return -(X.T @ q)

    res = minimize(neg_log_lik, np.zeros(k), jac=neg_score, method='Newton-CG',
                   options={'maxiter': 100, 'xtol': 1e-12})
    beta = res.x

    # 标准误 from observed information matrix
    xb = X @ beta
    xb = np.clip(xb, -20, 20)
    p = sp_stats.norm.cdf(xb)
    phi = sp_stats.norm.pdf(xb)
    # Hessian for probit
    lam = phi**2 / (p * (1 - p)) + (y - p) * (-xb) * phi / (p * (1 - p))
    W = np.diag(lam)
    H = -X.T @ W @ X
    V = np.linalg.inv(-H)
    se = np.sqrt(np.diag(V))

    ll = -res.fun
    return beta, se, ll


def _scipy_logit_oracle(y, X):
    """Logit MLE oracle using scipy.optimize.minimize"""
    n, k = X.shape

    def neg_log_lik(beta):
        xb = X @ beta
        xb = np.clip(xb, -20, 20)
        ll = np.sum(y * xb - np.log(1 + np.exp(xb)))
        return -ll

    def neg_score(beta):
        xb = X @ beta
        xb = np.clip(xb, -20, 20)
        p = 1 / (1 + np.exp(-xb))
        return -(X.T @ (y - p))

    res = minimize(neg_log_lik, np.zeros(k), jac=neg_score, method='Newton-CG',
                   options={'maxiter': 100, 'xtol': 1e-12})
    beta = res.x

    # 标准误 from Hessian
    xb = X @ beta
    xb = np.clip(xb, -20, 20)
    p = 1 / (1 + np.exp(-xb))
    W = np.diag(p * (1 - p))
    H = -X.T @ W @ X
    V = np.linalg.inv(-H)
    se = np.sqrt(np.diag(V))

    ll = -res.fun
    return beta, se, ll


@pytest.fixture
def synthetic_binary_data():
    """Synthetic binary choice data with known oracle results"""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    xb = 1.0 + 0.5 * x1 - 0.3 * x2
    # probit link
    p = sp_stats.norm.cdf(xb)
    y = (np.random.rand(n) < p).astype(float)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    # X order: [x1, x2, constant] to match our implementation
    X = np.column_stack([x1, x2, np.ones(n)])

    probit_coef, probit_se, probit_ll = _scipy_probit_oracle(y, X)
    logit_coef, logit_se, logit_ll = _scipy_logit_oracle(y, X)

    return {
        "df": df,
        "probit": {"coef": probit_coef, "se": probit_se, "ll": probit_ll},
        "logit": {"coef": logit_coef, "se": logit_se, "ll": logit_ll},
    }


# ─────────────────────────────────────────────────────────
# Probit tests
# ─────────────────────────────────────────────────────────

class TestProbit:

    def test_coefficients_vs_oracle(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]
        oracle = synthetic_binary_data["probit"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.allclose(result.coef, oracle["coef"], atol=1e-4)

    def test_standard_errors_vs_oracle(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]
        oracle = synthetic_binary_data["probit"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.allclose(result.std_err, oracle["se"], atol=1e-4)

    def test_log_likelihood_vs_oracle(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]
        oracle = synthetic_binary_data["probit"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.isclose(result.ll, oracle["ll"], atol=1e-3)

    def test_stata_coefficients(self, auto_stata_data, probit_stata_values):
        """Compare with actual Stata output"""
        from tabra.models.estimate.binary_choice import ProbitModel
        df = auto_stata_data
        sv = probit_stata_values

        model = ProbitModel()
        result = model.fit(df, y="highprice", x=["weight", "mpg"])

        assert np.allclose(result.coef, sv["coef"], atol=1e-4)
        assert np.allclose(result.std_err, sv["std_err"], atol=1e-4)
        assert np.isclose(result.ll, sv["ll"], atol=1e-3)
        assert np.isclose(result.pseudo_r2, sv["pseudo_r2"], atol=1e-3)

    def test_var_names(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert result.var_names == ["x1", "x2", "_cons"]

    def test_var_names_noconstant(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"], is_con=False)

        assert result.var_names == ["x1", "x2"]

    def test_n_obs(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert result.n_obs == 200

    def test_pseudo_r2_range(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert 0 <= result.pseudo_r2 <= 1

    def test_z_stat(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        expected_z = result.coef / result.std_err
        assert np.allclose(result.z_stat, expected_z, atol=1e-8)

    def test_chi2_positive(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert result.chi2 > 0

    def test_noconstant(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import ProbitModel
        df = synthetic_binary_data["df"]

        model = ProbitModel()
        result = model.fit(df, y="y", x=["x1", "x2"], is_con=False)

        assert result.k_vars == 2
        assert len(result.coef) == 2


# ─────────────────────────────────────────────────────────
# Logit tests
# ─────────────────────────────────────────────────────────

class TestLogit:

    def test_coefficients_vs_oracle(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import LogitModel
        df = synthetic_binary_data["df"]
        oracle = synthetic_binary_data["logit"]

        model = LogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.allclose(result.coef, oracle["coef"], atol=1e-4)

    def test_standard_errors_vs_oracle(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import LogitModel
        df = synthetic_binary_data["df"]
        oracle = synthetic_binary_data["logit"]

        model = LogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.allclose(result.std_err, oracle["se"], atol=1e-4)

    def test_log_likelihood_vs_oracle(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import LogitModel
        df = synthetic_binary_data["df"]
        oracle = synthetic_binary_data["logit"]

        model = LogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.isclose(result.ll, oracle["ll"], atol=1e-3)

    def test_stata_coefficients(self, auto_stata_data, logit_stata_values):
        """Compare with actual Stata output"""
        from tabra.models.estimate.binary_choice import LogitModel
        df = auto_stata_data
        sv = logit_stata_values

        model = LogitModel()
        result = model.fit(df, y="highprice", x=["weight", "mpg"])

        assert np.allclose(result.coef, sv["coef"], atol=1e-4)
        assert np.allclose(result.std_err, sv["std_err"], atol=1e-4)
        assert np.isclose(result.ll, sv["ll"], atol=1e-3)
        assert np.isclose(result.pseudo_r2, sv["pseudo_r2"], atol=1e-3)

    def test_var_names(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import LogitModel
        df = synthetic_binary_data["df"]

        model = LogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert result.var_names == ["x1", "x2", "_cons"]

    def test_noconstant(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import LogitModel
        df = synthetic_binary_data["df"]

        model = LogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"], is_con=False)

        assert result.k_vars == 2
        assert len(result.coef) == 2

    def test_n_obs(self, synthetic_binary_data):
        from tabra.models.estimate.binary_choice import LogitModel
        df = synthetic_binary_data["df"]

        model = LogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert result.n_obs == 200


# ─────────────────────────────────────────────────────────
# TabraData integration tests
# ─────────────────────────────────────────────────────────

class TestTabraDataIntegration:

    def test_tabradata_probit(self, synthetic_binary_data):
        from tabra.core.data import TabraData
        df = synthetic_binary_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.probit("y", ["x1", "x2"])

        assert result is not None
        assert result.n_obs == 200
        assert len(result.coef) == 3

    def test_tabradata_logit(self, synthetic_binary_data):
        from tabra.core.data import TabraData
        df = synthetic_binary_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.logit("y", ["x1", "x2"])

        assert result is not None
        assert result.n_obs == 200
        assert len(result.coef) == 3

    def test_summary_output(self, synthetic_binary_data):
        from tabra.core.data import TabraData
        df = synthetic_binary_data["df"]

        tab = TabraData(df, is_display_result=False)
        result = tab.probit("y", ["x1", "x2"])
        summary = result.summary()

        assert "Probit regression" in summary
        assert "Number of obs" in summary
        assert "Pseudo R2" in summary
        assert "z" in summary
        assert "P>|z|" in summary
