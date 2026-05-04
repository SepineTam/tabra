#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_glm.py

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def gaussian_data():
    # 连续Y，用于 Gaussian + identity（应接近OLS）
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + np.random.randn(n) * 0.5
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


@pytest.fixture
def binomial_data():
    # 二元Y，用于 Binomial + logit / probit
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    xb = 1.0 + 0.5 * x1 - 0.3 * x2
    p = 1 / (1 + np.exp(-xb))
    y = (np.random.rand(n) < p).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


@pytest.fixture
def poisson_data():
    # 计数Y，用于 Poisson + log
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    lam = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
    y = np.random.poisson(lam)
    df = pd.DataFrame({"y": y.astype(float), "x1": x1, "x2": x2})
    return df


@pytest.fixture
def gamma_data():
    # 正连续Y，用于 Gamma + log
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    mu = np.exp(1.0 + 0.2 * x1 - 0.1 * x2)
    y = np.random.gamma(shape=2.0, scale=mu / 2.0)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


# ─────────────────────────────────────────────────────────
# Helper: 用 numpy lstsq 作为 oracle
# ─────────────────────────────────────────────────────────

def _ols_oracle(df, y_col, x_cols, is_con=True):
    y = df[y_col].values
    X = df[x_cols].values.astype(float)
    if is_con:
        X = np.column_stack([X, np.ones(X.shape[0])])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ─────────────────────────────────────────────────────────
# Gaussian + identity ≈ OLS
# ─────────────────────────────────────────────────────────

class TestGaussianIdentity:

    def test_coef_vs_ols(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        model = GLMModel()
        result = model.fit(gaussian_data, y="y", x=["x1", "x2"],
                           family="gaussian", link="identity")
        oracle = _ols_oracle(gaussian_data, "y", ["x1", "x2"])
        assert np.allclose(result.coef, oracle, atol=1e-6)

    def test_coef_shape(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.coef.shape == (3,)

    def test_std_err_positive(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert np.all(result.std_err > 0)

    def test_p_values_in_range(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert np.all(result.p_value >= 0)
        assert np.all(result.p_value <= 1)

    def test_pseudo_r2(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert 0 <= result.pseudo_r2 <= 1

    def test_ll_and_ll0(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.ll < 0 or result.ll >= 0  # just exists
        assert result.ll_0 is not None
        assert result.ll >= result.ll_0  # full model logLik >= null

    def test_oracle_lstsq(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        oracle = _ols_oracle(gaussian_data, "y", ["x1", "x2"])
        assert np.allclose(result.coef, oracle, atol=1e-5)

    def test_n_obs(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.n_obs == 200

    def test_converged(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.converged is True

    def test_var_names(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.var_names == ["x1", "x2", "_cons"]

    def test_model_name(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert "GLM" in result.model_name


# ─────────────────────────────────────────────────────────
# Binomial + logit
# ─────────────────────────────────────────────────────────

class TestBinomialLogit:

    def test_coef_shape(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="logit")
        assert result.coef.shape == (3,)

    def test_std_err_positive(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="logit")
        assert np.all(result.std_err > 0)

    def test_p_values_in_range(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="logit")
        assert np.all(result.p_value >= 0)
        assert np.all(result.p_value <= 1)

    def test_converged(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="logit")
        assert result.converged is True

    def test_pseudo_r2(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="logit")
        assert 0 <= result.pseudo_r2 <= 1

    def test_vs_logit_model(self, binomial_data):
        # GLM binomial+logit 应该和 LogitModel 的结果接近
        from tabra.models.estimate.glm import GLMModel
        from tabra.models.estimate.binary_choice import LogitModel
        glm_result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                    family="binomial", link="logit")
        logit_result = LogitModel().fit(binomial_data, y="y", x=["x1", "x2"])
        assert np.allclose(glm_result.coef, logit_result.coef, atol=1e-3)


# ─────────────────────────────────────────────────────────
# Binomial + probit
# ─────────────────────────────────────────────────────────

class TestBinomialProbit:

    def test_coef_shape(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="probit")
        assert result.coef.shape == (3,)

    def test_std_err_positive(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="probit")
        assert np.all(result.std_err > 0)

    def test_converged(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                                family="binomial", link="probit")
        assert result.converged is True


# ─────────────────────────────────────────────────────────
# Poisson + log
# ─────────────────────────────────────────────────────────

class TestPoissonLog:

    def test_coef_shape(self, poisson_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                                family="poisson", link="log")
        assert result.coef.shape == (3,)

    def test_std_err_positive(self, poisson_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                                family="poisson", link="log")
        assert np.all(result.std_err > 0)

    def test_p_values_in_range(self, poisson_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                                family="poisson", link="log")
        assert np.all(result.p_value >= 0)
        assert np.all(result.p_value <= 1)

    def test_converged(self, poisson_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                                family="poisson", link="log")
        assert result.converged is True

    def test_pseudo_r2(self, poisson_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                                family="poisson", link="log")
        assert 0 <= result.pseudo_r2 <= 1


# ─────────────────────────────────────────────────────────
# Gamma + log
# ─────────────────────────────────────────────────────────

class TestGammaLog:

    def test_coef_shape(self, gamma_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                                family="gamma", link="log")
        assert result.coef.shape == (3,)

    def test_std_err_positive(self, gamma_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                                family="gamma", link="log")
        assert np.all(result.std_err > 0)

    def test_p_values_in_range(self, gamma_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                                family="gamma", link="log")
        assert np.all(result.p_value >= 0)
        assert np.all(result.p_value <= 1)

    def test_converged(self, gamma_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                                family="gamma", link="log")
        assert result.converged is True

    def test_pseudo_r2(self, gamma_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                                family="gamma", link="log")
        assert 0 <= result.pseudo_r2 <= 1


# ─────────────────────────────────────────────────────────
# Default link (link=None) auto-match
# ─────────────────────────────────────────────────────────

class TestDefaultLink:

    def test_gaussian_default_is_identity(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        r1 = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                            family="gaussian")
        r2 = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                            family="gaussian", link="identity")
        assert np.allclose(r1.coef, r2.coef, atol=1e-10)

    def test_binomial_default_is_logit(self, binomial_data):
        from tabra.models.estimate.glm import GLMModel
        r1 = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                            family="binomial")
        r2 = GLMModel().fit(binomial_data, y="y", x=["x1", "x2"],
                            family="binomial", link="logit")
        assert np.allclose(r1.coef, r2.coef, atol=1e-10)

    def test_poisson_default_is_log(self, poisson_data):
        from tabra.models.estimate.glm import GLMModel
        r1 = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                            family="poisson")
        r2 = GLMModel().fit(poisson_data, y="y", x=["x1", "x2"],
                            family="poisson", link="log")
        assert np.allclose(r1.coef, r2.coef, atol=1e-10)

    def test_gamma_default_is_log(self, gamma_data):
        from tabra.models.estimate.glm import GLMModel
        r1 = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                            family="gamma")
        r2 = GLMModel().fit(gamma_data, y="y", x=["x1", "x2"],
                            family="gamma", link="log")
        assert np.allclose(r1.coef, r2.coef, atol=1e-10)


# ─────────────────────────────────────────────────────────
# Result properties
# ─────────────────────────────────────────────────────────

class TestResultProperties:

    def test_family_and_link(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.family == "gaussian"
        assert result.link == "identity"

    def test_deviance_positive(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.deviance >= 0

    def test_null_deviance_positive(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.null_deviance >= 0

    def test_V_shape(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.V.shape == (3, 3)

    def test_y_name(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.y_name == "y"

    def test_df_m(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.df_m == 2

    def test_k_vars(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity")
        assert result.k_vars == 3


# ─────────────────────────────────────────────────────────
# TabraData integration
# ─────────────────────────────────────────────────────────

class TestTabraDataGLM:

    def test_tabradata_glm(self, gaussian_data):
        from tabra.core.data import TabraData
        tab = TabraData(gaussian_data, is_display_result=False)
        result = tab.est.glm("y", ["x1", "x2"], family="gaussian", link="identity")
        assert result is not None
        assert result.n_obs == 200

    def test_tabradata_glm_summary(self, gaussian_data):
        from tabra.core.data import TabraData
        tab = TabraData(gaussian_data, is_display_result=False)
        result = tab.est.glm("y", ["x1", "x2"], family="gaussian", link="identity")
        summary = result.summary()
        assert "GLM" in summary
        assert "Number of obs" in summary

    def test_noconstant(self, gaussian_data):
        from tabra.models.estimate.glm import GLMModel
        result = GLMModel().fit(gaussian_data, y="y", x=["x1", "x2"],
                                family="gaussian", link="identity", is_con=False)
        assert result.k_vars == 2
        assert len(result.coef) == 2
