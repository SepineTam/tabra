#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : binary_choice.py

import numpy as np
from abc import abstractmethod
from scipy import stats as sp_stats
from scipy.special import expit

from tabra.models.estimate.base import BaseModel
from tabra.results.binary_choice_result import BinaryChoiceResult


class BinaryChoiceModel(BaseModel):
    """Base class for binary choice models (probit/logit) estimated via MLE."""

    def fit(self, df, y, x, is_con=True, max_iter=100, tol=1e-8):
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)

        if is_con:
            X = np.column_stack([X, np.ones(X.shape[0])])
            var_names = var_names + ["_cons"]

        n, k = X.shape

        # Step 1: Fit constant-only model for ll_0
        ll_0 = self._fit_null_model(y_vec, is_con)

        # Step 2: Newton-Raphson / IRLS for full model
        beta = np.zeros(k)
        converged = False

        for iteration in range(max_iter):
            p = self._link(X @ beta)
            p = np.clip(p, 1e-15, 1 - 1e-15)

            # Gradient and Hessian
            g = self._gradient(X, y_vec, p, beta)
            H = self._hessian(X, y_vec, p, beta)

            # Newton step: delta = -H^{-1} g
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break

            beta_new = beta - delta

            if np.max(np.abs(delta)) < tol:
                beta = beta_new
                converged = True
                break

            beta = beta_new
            n_iter = iteration + 1

        # Step 3: Compute final quantities
        p = self._link(X @ beta)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        ll = self._log_likelihood(y_vec, p)

        # Variance-covariance matrix from observed information
        H_final = self._hessian(X, y_vec, p, beta)
        try:
            V = np.linalg.inv(-H_final)
        except np.linalg.LinAlgError:
            V = np.linalg.pinv(-H_final)

        std_err = np.sqrt(np.diag(V))
        z_stat = beta / std_err
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # Pseudo R-squared
        pseudo_r2 = 1 - ll / ll_0 if ll_0 != 0 else 0.0

        # LR chi-squared test
        df_m = k - 1 if is_con else k
        chi2 = 2 * (ll - ll_0)
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        return BinaryChoiceResult(
            coef=beta,
            std_err=std_err,
            z_stat=z_stat,
            p_value=p_value,
            ll=ll,
            ll_0=ll_0,
            pseudo_r2=pseudo_r2,
            chi2=chi2,
            chi2_pval=chi2_pval,
            n_obs=n,
            k_vars=k,
            df_m=df_m,
            var_names=var_names,
            y_name=y,
            model_name=self._model_name(),
            converged=converged,
            n_iter=n_iter,
            vce_type="OIM",
        )

    def _fit_null_model(self, y, is_con):
        """Fit constant-only model to get ll_0."""
        p_bar = np.mean(y)
        p_bar = np.clip(p_bar, 1e-15, 1 - 1e-15)
        ll_0 = np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar))
        return ll_0

    def _log_likelihood(self, y, p):
        """Compute log-likelihood."""
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    @abstractmethod
    def _link(self, xb):
        """Link function: linear predictor -> probability."""
        raise NotImplementedError

    @abstractmethod
    def _gradient(self, X, y, p, beta):
        """Compute gradient (score) vector."""
        raise NotImplementedError

    @abstractmethod
    def _hessian(self, X, y, p, beta):
        """Compute Hessian matrix (should be negative definite)."""
        raise NotImplementedError

    @abstractmethod
    def _model_name(self):
        """Return model name for display."""
        raise NotImplementedError

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for binary choice estimation")


class ProbitModel(BinaryChoiceModel):
    """Probit regression: P(y=1|x) = Phi(x'b)"""

    def _link(self, xb):
        xb = np.clip(xb, -20, 20)
        return sp_stats.norm.cdf(xb)

    def _gradient(self, X, y, p, beta):
        xb = X @ beta
        xb = np.clip(xb, -20, 20)
        phi = sp_stats.norm.pdf(xb)
        # score = X' * (y - p) * phi / (p * (1-p))
        q = (y - p) * phi / (p * (1 - p))
        return X.T @ q

    def _hessian(self, X, y, p, beta):
        xb = X @ beta
        xb = np.clip(xb, -20, 20)
        phi = sp_stats.norm.pdf(xb)
        # Hessian diagonal weights
        # lambda = phi^2 / (p*(1-p)) - (y-p) * xb * phi / (p*(1-p))
        lam = (phi ** 2) / (p * (1 - p)) - (y - p) * xb * phi / (p * (1 - p))
        W = np.diag(lam)
        return -X.T @ W @ X

    def _model_name(self):
        return "Probit regression"


class LogitModel(BinaryChoiceModel):
    """Logistic regression: P(y=1|x) = 1/(1+exp(-x'b))"""

    def _link(self, xb):
        xb = np.clip(xb, -500, 500)
        return expit(xb)

    def _gradient(self, X, y, p, beta):
        # score = X'(y - p)
        return X.T @ (y - p)

    def _hessian(self, X, y, p, beta):
        # H = -X'WX where W = diag(p*(1-p))
        W = np.diag(p * (1 - p))
        return -X.T @ W @ X

    def _model_name(self):
        return "Logistic regression"
