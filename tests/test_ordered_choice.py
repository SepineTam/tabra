#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Tests for ordered choice models (oprobit/ologit)

import numpy as np
import pandas as pd
import pytest

from tabra.models.estimate.ordered_choice import (
    OrderedProbitModel,
    OrderedLogitModel,
)


def _make_ordered_data(n=1000, k=3, n_cat=4, seed=42):
    """Generate ordered choice data with known structure."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, k)
    beta = np.array([1.0, -0.5, 0.8])

    # Latent variable: y* = X @ beta + eps
    xb = X @ beta
    eps = rng.randn(n)  # standard normal for probit
    y_star = xb + eps

    # Cutpoints evenly spaced
    cuts = np.linspace(-1, 2, n_cat - 1)
    y = np.zeros(n, dtype=int)
    for i, c in enumerate(cuts):
        if i == 0:
            y[y_star <= c] = 0
        elif i == len(cuts) - 1:
            y[(y_star > cuts[i - 1]) & (y_star <= c)] = i
            y[y_star > c] = i + 1
        else:
            y[(y_star > cuts[i - 1]) & (y_star <= c)] = i
    # Fix: simple assignment
    y = np.zeros(n, dtype=int)
    for j in range(n):
        y[j] = np.searchsorted(cuts, y_star[j])

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(k)])
    df["y"] = y
    return df, beta, cuts


class TestOrderedProbit:

    def test_basic_fit(self):
        df, beta_true, cuts_true = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"], is_con=True)
        assert result is not None

    def test_n_obs(self):
        df, _, _ = _make_ordered_data(n=500)
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.n_obs == 500

    def test_coef_shape(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        # 3 x-vars + 1 constant = 4 coefficients
        assert len(result.coef) == 4

    def test_cutpoints(self):
        df, _, cuts_true = _make_ordered_data(n_cat=4)
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        # Should have n_cat - 1 = 3 cutpoints
        assert len(result.cutpoints) == 3
        # Cutpoints should be ordered
        cuts = result.cutpoints
        for i in range(len(cuts) - 1):
            assert cuts[i] < cuts[i + 1]

    def test_coef_reasonable(self):
        """Coefficients should be close to true values."""
        df, beta_true, _ = _make_ordered_data(n=5000, seed=42)
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        # Check first 3 coefficients (x1, x2, x3) are within 0.3 of truth
        for i in range(3):
            assert abs(result.coef[i] - beta_true[i]) < 0.3, \
                f"coef[{i}]: {result.coef[i]} vs true {beta_true[i]}"

    def test_std_err_positive(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert np.all(result.std_err > 0)

    def test_p_value_range(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert np.all(result.p_value >= 0)
        assert np.all(result.p_value <= 1)

    def test_pseudo_r2(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert 0 < result.pseudo_r2 < 1

    def test_k_cat(self):
        df, _, _ = _make_ordered_data(n_cat=4)
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.k_cat == 4

    def test_ll_and_ll0(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.ll < 0
        assert result.ll_0 < 0
        # Full model should fit better than null
        assert result.ll > result.ll_0

    def test_var_names(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert "_cons" in result.var_names

    def test_converged(self):
        df, _, _ = _make_ordered_data()
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.converged is True


class TestOrderedLogit:

    def test_basic_fit(self):
        df, _, _ = _make_ordered_data()
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result is not None

    def test_coef_shape(self):
        df, _, _ = _make_ordered_data()
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert len(result.coef) == 4

    def test_cutpoints_ordered(self):
        df, _, _ = _make_ordered_data(n_cat=5)
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        cuts = result.cutpoints
        assert len(cuts) == 4
        for i in range(len(cuts) - 1):
            assert cuts[i] < cuts[i + 1]

    def test_coef_reasonable(self):
        """Logit coefficients should be ~1.6x probit (scaling factor)."""
        df, beta_true, _ = _make_ordered_data(n=5000, seed=42)
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        # Logit scale ≈ probit * pi/sqrt(3) ≈ 1.81
        logit_scale = np.pi / np.sqrt(3)
        for i in range(3):
            expected = beta_true[i] * logit_scale
            assert abs(result.coef[i] - expected) < 0.5, \
                f"coef[{i}]: {result.coef[i]} vs expected {expected}"

    def test_pseudo_r2(self):
        df, _, _ = _make_ordered_data()
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert 0 < result.pseudo_r2 < 1

    def test_n_obs(self):
        df, _, _ = _make_ordered_data(n=300)
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.n_obs == 300

    def test_converged(self):
        df, _, _ = _make_ordered_data()
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.converged is True

    def test_chi2_positive(self):
        df, _, _ = _make_ordered_data()
        model = OrderedLogitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])
        assert result.chi2 > 0


class TestOracleConsistency:
    """Compare tabra results against direct scipy optimization."""

    def test_probit_vs_numpy(self):
        """Tabra oprobit should match direct MLE."""
        from scipy import stats as sp_stats
        from scipy.optimize import minimize

        df, beta_true, cuts_true = _make_ordered_data(n=2000, n_cat=3, seed=123)
        model = OrderedProbitModel()
        result = model.fit(df, "y", ["x1", "x2", "x3"])

        # Direct optimization
        y = df["y"].values.astype(int)
        X = df[["x1", "x2", "x3"]].values
        X = np.column_stack([X, np.ones(len(X))])
        k = X.shape[1]
        n_cat = len(np.unique(y))

        def neg_ll(params):
            beta = params[:k]
            cuts = params[k:k + n_cat - 1]
            # Ensure ordered cutpoints
            for i in range(1, len(cuts)):
                if cuts[i] <= cuts[i - 1]:
                    return 1e15
            xb = X @ beta
            ll = 0.0
            for i in range(len(y)):
                if y[i] == 0:
                    ll += np.log(sp_stats.norm.cdf(cuts[0] - xb[i]) + 1e-300)
                elif y[i] == n_cat - 1:
                    ll += np.log(1 - sp_stats.norm.cdf(cuts[-1] - xb[i]) + 1e-300)
                else:
                    p = (sp_stats.norm.cdf(cuts[y[i]] - xb[i])
                         - sp_stats.norm.cdf(cuts[y[i] - 1] - xb[i]))
                    ll += np.log(max(p, 1e-300))
            return -ll

        x0 = np.zeros(k + n_cat - 1)
        x0[k] = -1
        x0[k + 1] = 1
        res_direct = minimize(neg_ll, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12})

        # Compare coefficients
        for i in range(k):
            assert abs(result.coef[i] - res_direct.x[i]) < 0.2, \
                f"coef[{i}]: tabra={result.coef[i]} vs direct={res_direct.x[i]}"
