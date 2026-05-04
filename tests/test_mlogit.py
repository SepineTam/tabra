#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_mlogit.py

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def mlogit_data():
    """Synthetic 3-category multinomial logit data."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # True parameters: base category = 0, two equation sets
    beta_1 = np.array([0.5, 1.0, -0.5])   # category 1 vs base(0)
    beta_2 = np.array([-0.3, 0.0, 1.0])   # category 2 vs base(0)

    # Compute probabilities via softmax
    XB = np.column_stack([np.zeros(n), X @ beta_1, X @ beta_2])
    XB -= XB.max(axis=1, keepdims=True)
    exp_XB = np.exp(XB)
    probs = exp_XB / exp_XB.sum(axis=1, keepdims=True)

    # Draw y from categorical
    cum_probs = np.cumsum(probs, axis=1)
    u = np.random.rand(n)
    y = np.array([np.searchsorted(cum_probs[i], u[i]) for i in range(n)])

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


@pytest.fixture
def mlogit_data_4cat():
    """Synthetic 4-category multinomial logit data."""
    np.random.seed(123)
    n = 400
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    beta_1 = np.array([0.5, 1.0, -0.5])
    beta_2 = np.array([-0.3, 0.0, 1.0])
    beta_3 = np.array([1.0, -1.0, 0.5])

    XB = np.column_stack([np.zeros(n), X @ beta_1, X @ beta_2, X @ beta_3])
    XB -= XB.max(axis=1, keepdims=True)
    exp_XB = np.exp(XB)
    probs = exp_XB / exp_XB.sum(axis=1, keepdims=True)

    cum_probs = np.cumsum(probs, axis=1)
    u = np.random.rand(n)
    y = np.array([np.searchsorted(cum_probs[i], u[i]) for i in range(n)])

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


# ─────────────────────────────────────────────────────────
# Oracle: direct scipy optimization for mlogit
# ─────────────────────────────────────────────────────────

def _scipy_mlogit_oracle(y, X, categories, base_idx):
    """Multinomial logit oracle via scipy L-BFGS-B."""
    cats = list(categories)
    n_cats = len(cats)
    n, k = X.shape
    non_base = [c for i, c in enumerate(cats) if i != base_idx]
    n_eq = len(non_base)

    def neg_log_lik(params_flat):
        beta_mat = np.zeros((n_cats, k))
        for eq_i, cat_i in enumerate(non_base):
            beta_mat[cat_i] = params_flat[eq_i * k:(eq_i + 1) * k]

        XB = X @ beta_mat.T  # (n, n_cats)
        XB -= XB.max(axis=1, keepdims=True)
        exp_XB = np.exp(XB)
        log_sum_exp = np.log(exp_XB.sum(axis=1))
        ll = np.sum(XB[np.arange(n), y] - log_sum_exp)
        return -ll

    def neg_score(params_flat):
        beta_mat = np.zeros((n_cats, k))
        for eq_i, cat_i in enumerate(non_base):
            beta_mat[cat_i] = params_flat[eq_i * k:(eq_i + 1) * k]

        XB = X @ beta_mat.T
        XB -= XB.max(axis=1, keepdims=True)
        exp_XB = np.exp(XB)
        probs = exp_XB / exp_XB.sum(axis=1, keepdims=True)

        grad = np.zeros_like(params_flat)
        for eq_i, cat_i in enumerate(non_base):
            resid = (y == cat_i).astype(float) - probs[:, cat_i]
            grad[eq_i * k:(eq_i + 1) * k] = -(X.T @ resid)
        return grad

    x0 = np.zeros(n_eq * k)
    res = minimize(neg_log_lik, x0, jac=neg_score, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})

    params_flat = res.x
    ll = -res.fun

    # VCE via numerical Hessian
    eps = 1e-5
    p = len(params_flat)
    H = np.zeros((p, p))
    f0 = neg_log_lik(params_flat)
    for i in range(p):
        ei = np.zeros(p)
        ei[i] = eps
        H[i, i] = (neg_log_lik(params_flat + ei) - 2 * f0 + neg_log_lik(params_flat - ei)) / (eps ** 2)
        for j in range(i + 1, p):
            ej = np.zeros(p)
            ej[j] = eps
            H[i, j] = (neg_log_lik(params_flat + ei + ej) - neg_log_lik(params_flat + ei - ej)
                        - neg_log_lik(params_flat - ei + ej) + neg_log_lik(params_flat - ei - ej)) / (4 * eps ** 2)
            H[j, i] = H[i, j]

    V = np.linalg.inv(H)
    se = np.sqrt(np.maximum(np.diag(V), 0))

    # Reshape coefficients
    beta_dict = {}
    beta_dict[cats[base_idx]] = np.zeros(k)
    for eq_i, cat_i in enumerate(non_base):
        beta_dict[cat_i] = params_flat[eq_i * k:(eq_i + 1) * k]

    return {
        "coef": beta_dict,
        "se": se,
        "ll": ll,
        "params_flat": params_flat,
    }


# ─────────────────────────────────────────────────────────
# Basic model tests
# ─────────────────────────────────────────────────────────

class TestMLogitBasic:

    def test_basic_fit_3cat(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result is not None

    def test_n_obs(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.n_obs == len(mlogit_data)

    def test_coef_shape(self, mlogit_data):
        """coef shape: k_vars * (n_cat - 1), base category has zeros."""
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        k_vars = result.k_vars
        k_cat = result.k_cat
        # Each non-base category has k_vars coefficients
        for cat in result.categories:
            if cat == result.base_outcome:
                assert np.allclose(result.coef[cat], 0.0)
            else:
                assert len(result.coef[cat]) == k_vars

    def test_std_err_positive(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        for cat in result.categories:
            if cat != result.base_outcome:
                assert np.all(result.std_err[cat] > 0)

    def test_p_values_in_range(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        for cat in result.categories:
            if cat != result.base_outcome:
                assert np.all(result.p_value[cat] >= 0)
                assert np.all(result.p_value[cat] <= 1)

    def test_pseudo_r2_in_range(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert 0 < result.pseudo_r2 < 1

    def test_ll_greater_than_ll0(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.ll > result.ll_0

    def test_converged(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.converged is True

    def test_var_names_includes_cons(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert "_cons" in result.var_names

    def test_k_cat(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.k_cat == 3

    def test_4cat(self, mlogit_data_4cat):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data_4cat, y="y", x=["x1", "x2"])
        assert result.k_cat == 4
        base_count = sum(1 for c in result.categories if c == result.base_outcome)
        assert base_count == 1


# ─────────────────────────────────────────────────────────
# Oracle comparison tests
# ─────────────────────────────────────────────────────────

class TestMLogitOracle:

    def test_coef_vs_oracle(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        df = mlogit_data
        y = df["y"].values.astype(int)
        X = np.column_stack([df["x1"].values, df["x2"].values, np.ones(len(df))])

        categories = np.sort(np.unique(y))
        oracle = _scipy_mlogit_oracle(y, X, categories, base_idx=0)

        model = MultinomialLogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        for cat in categories:
            if cat == result.base_outcome:
                continue
            assert np.allclose(result.coef[cat], oracle["coef"][cat], atol=1e-3)

    def test_ll_vs_oracle(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        df = mlogit_data
        y = df["y"].values.astype(int)
        X = np.column_stack([df["x1"].values, df["x2"].values, np.ones(len(df))])

        categories = np.sort(np.unique(y))
        oracle = _scipy_mlogit_oracle(y, X, categories, base_idx=0)

        model = MultinomialLogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        assert np.isclose(result.ll, oracle["ll"], atol=1e-2)

    def test_se_vs_oracle(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        df = mlogit_data
        y = df["y"].values.astype(int)
        X = np.column_stack([df["x1"].values, df["x2"].values, np.ones(len(df))])

        categories = np.sort(np.unique(y))
        oracle = _scipy_mlogit_oracle(y, X, categories, base_idx=0)

        model = MultinomialLogitModel()
        result = model.fit(df, y="y", x=["x1", "x2"])

        # Compare flattened SE
        k = X.shape[1]
        oracle_se_flat = oracle["se"]
        our_se_flat = np.concatenate(
            [result.std_err[cat] for cat in categories if cat != result.base_outcome]
        )
        assert np.allclose(our_se_flat, oracle_se_flat, atol=1e-3)


# ─────────────────────────────────────────────────────────
# base_outcome parameter test
# ─────────────────────────────────────────────────────────

class TestMLogitBaseOutcome:

    def test_custom_base_outcome(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"], base_outcome=1)
        assert result.base_outcome == 1
        assert np.allclose(result.coef[1], 0.0)

    def test_default_base_outcome(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.base_outcome == 0


# ─────────────────────────────────────────────────────────
# Result class tests
# ─────────────────────────────────────────────────────────

class TestMLogitResult:

    def test_summary_output(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        summary = result.summary()

        assert "Multinomial" in summary or "mlogit" in summary.lower()
        assert "Number of obs" in summary
        assert "Pseudo R2" in summary
        assert "x1" in summary
        assert "x2" in summary
        assert "_cons" in summary

    def test_chi2_positive(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.chi2 > 0

    def test_chi2_pval_in_range(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert 0 <= result.chi2_pval <= 1

    def test_v_matrix_exists(self, mlogit_data):
        from tabra.models.estimate.mlogit import MultinomialLogitModel
        model = MultinomialLogitModel()
        result = model.fit(mlogit_data, y="y", x=["x1", "x2"])
        assert result.V is not None
        assert result.V.shape[0] == result.V.shape[1]


# ─────────────────────────────────────────────────────────
# TabraData integration test
# ─────────────────────────────────────────────────────────

class TestMLogitIntegration:

    def test_tabradata_mlogit(self, mlogit_data):
        from tabra.core.data import TabraData
        tab = TabraData(mlogit_data, is_display_result=False)
        result = tab.est.mlogit("y", ["x1", "x2"])
        assert result is not None
        assert result.n_obs == len(mlogit_data)
        assert result.k_cat == 3

    def test_tabradata_mlogit_base_outcome(self, mlogit_data):
        from tabra.core.data import TabraData
        tab = TabraData(mlogit_data, is_display_result=False)
        result = tab.est.mlogit("y", ["x1", "x2"], base_outcome=2)
        assert result.base_outcome == 2
