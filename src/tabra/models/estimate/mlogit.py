#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : mlogit.py

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from tabra.models.estimate.base import BaseModel
from tabra.results.mlogit_result import MLogitResult


class MultinomialLogitModel(BaseModel):
    """Multinomial logistic regression estimated via MLE."""

    def fit(self, df, y, x, base_outcome=None, is_con=True, max_iter=200,
            tol=1e-8):
        """Fit a multinomial logistic regression via MLE.

        Args:
            df (pd.DataFrame): Input dataset.
            y (str): Categorical dependent variable name.
            x (list[str]): Independent variable names.
            base_outcome: Base category value. Default None (uses first category).
            is_con (bool): Whether to include a constant term. Default True.
            max_iter (int): Maximum optimizer iterations. Default 200.
            tol (float): Convergence tolerance. Default 1e-8.

        Returns:
            MLogitResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = MultinomialLogitModel().fit(dta._df, "rep78", ["price", "weight"])
        """
        df = self._prepare_df(df, y, x)
        y_vec = df[y].values.astype(int)
        X = df[x].values.astype(float)
        var_names = list(x)

        if is_con:
            X = np.column_stack([X, np.ones(X.shape[0])])
            var_names = var_names + ["_cons"]

        n, k = X.shape
        categories = np.sort(np.unique(y_vec))
        k_cat = len(categories)

        # Map categories to 0..k_cat-1
        cat_to_idx = {c: i for i, c in enumerate(categories)}

        # Determine base category
        if base_outcome is None:
            base_cat = categories[0]
        else:
            if base_outcome not in categories:
                raise ValueError(
                    f"base_outcome {base_outcome} not in categories {categories}")
            base_cat = base_outcome
        base_idx = cat_to_idx[base_cat]

        non_base = [c for c in categories if c != base_cat]
        n_eq = len(non_base)

        # Fit null model (constant-only) for ll_0
        ll_0 = self._fit_null_model(y_vec, categories, base_cat)

        # Full model MLE via L-BFGS-B
        def neg_log_lik(params_flat):
            beta_mat = np.zeros((k_cat, k))
            for eq_i, cat in enumerate(non_base):
                cat_idx = cat_to_idx[cat]
                beta_mat[cat_idx] = params_flat[eq_i * k:(eq_i + 1) * k]

            XB = X @ beta_mat.T  # (n, k_cat)
            # Numerically stable log-softmax
            XB_max = XB.max(axis=1, keepdims=True)
            XB_stable = XB - XB_max
            log_sum_exp = np.log(np.exp(XB_stable).sum(axis=1))
            ll = np.sum(XB_stable[np.arange(n), y_vec] - log_sum_exp)
            return -ll

        def neg_score(params_flat):
            beta_mat = np.zeros((k_cat, k))
            for eq_i, cat in enumerate(non_base):
                cat_idx = cat_to_idx[cat]
                beta_mat[cat_idx] = params_flat[eq_i * k:(eq_i + 1) * k]

            XB = X @ beta_mat.T
            XB_max = XB.max(axis=1, keepdims=True)
            XB_stable = XB - XB_max
            exp_XB = np.exp(XB_stable)
            probs = exp_XB / exp_XB.sum(axis=1, keepdims=True)

            grad = np.zeros_like(params_flat)
            for eq_i, cat in enumerate(non_base):
                cat_idx = cat_to_idx[cat]
                resid = (y_vec == cat).astype(float) - probs[:, cat_idx]
                grad[eq_i * k:(eq_i + 1) * k] = -(X.T @ resid)
            return grad

        x0 = np.zeros(n_eq * k)
        result_opt = minimize(neg_log_lik, x0, jac=neg_score, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': 1e-12,
                                       'gtol': tol})

        params_flat = result_opt.x
        ll = -result_opt.fun
        converged = result_opt.success

        # Build coefficient dict
        coef = {}
        coef[base_cat] = np.zeros(k)
        for eq_i, cat in enumerate(non_base):
            coef[cat] = params_flat[eq_i * k:(eq_i + 1) * k].copy()

        # VCE via numerical Hessian
        V = self._compute_vce(params_flat, neg_log_lik)

        n_params = len(params_flat)
        se_flat = np.sqrt(np.maximum(np.diag(V), 0))
        z_flat = params_flat / se_flat
        p_flat = 2 * (1 - sp_stats.norm.cdf(np.abs(z_flat)))

        # Split into per-category dicts
        std_err = {}
        z_stat = {}
        p_value = {}
        std_err[base_cat] = np.zeros(k)
        z_stat[base_cat] = np.zeros(k)
        p_value[base_cat] = np.ones(k)
        for eq_i, cat in enumerate(non_base):
            std_err[cat] = se_flat[eq_i * k:(eq_i + 1) * k]
            z_stat[cat] = z_flat[eq_i * k:(eq_i + 1) * k]
            p_value[cat] = p_flat[eq_i * k:(eq_i + 1) * k]

        # Pseudo R2
        pseudo_r2 = 1 - ll / ll_0 if ll_0 != 0 else 0.0

        # LR chi2
        df_m = n_eq * k - (1 if is_con else 0) * n_eq
        # In Stata, df_m = n_eq * (k - 1) if is_con, else n_eq * k
        # Actually Stata uses: df_m = k_cat * (k - 1) if constant
        # but the standard is: df_m = (k_cat - 1) * k minus (k_cat-1) for constant
        # Simplify: df_m = n_eq * (k - 1) if is_con, else n_eq * k
        if is_con:
            df_m = n_eq * (k - 1)
        else:
            df_m = n_eq * k

        chi2 = 2 * (ll - ll_0)
        chi2 = max(chi2, 0)
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        return MLogitResult(
            coef=coef,
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
            k_cat=k_cat,
            df_m=df_m,
            var_names=var_names,
            y_name=y,
            categories=list(categories),
            base_outcome=base_cat,
            converged=converged,
            model_name="Multinomial logistic regression",
            V=V,
        )

    def _fit_null_model(self, y, categories, base_cat):
        """Fit constant-only model for ll_0."""
        n = len(y)
        counts = {}
        for c in categories:
            counts[c] = np.sum(y == c)

        # Null model: intercept-only multinomial logit
        # P(y=j) = exp(alpha_j) / sum_k exp(alpha_k), alpha_base = 0
        non_base = [c for c in categories if c != base_cat]
        n_eq = len(non_base)

        def neg_ll_null(alphas):
            log_probs = np.zeros(len(categories))
            for i, c in enumerate(non_base):
                idx = list(categories).index(c)
                log_probs[idx] = alphas[i]
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()

            ll = 0.0
            for i, c in enumerate(categories):
                if counts[c] > 0:
                    ll += counts[c] * np.log(max(probs[i], 1e-300))
            return -ll

        from scipy.optimize import minimize as scipy_minimize
        x0 = np.zeros(n_eq)
        res = scipy_minimize(neg_ll_null, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-15})
        return -res.fun

    def _compute_vce(self, params, neg_log_lik):
        """Compute VCE via numerical Hessian."""
        eps = 1e-5
        p = len(params)
        H = np.zeros((p, p))
        f0 = neg_log_lik(params)

        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps
            f_plus = neg_log_lik(params + ei)
            f_minus = neg_log_lik(params - ei)
            H[i, i] = (f_plus - 2 * f0 + f_minus) / (eps ** 2)

            for j in range(i + 1, p):
                ej = np.zeros(p)
                ej[j] = eps
                f_pp = neg_log_lik(params + ei + ej)
                f_pm = neg_log_lik(params + ei - ej)
                f_mp = neg_log_lik(params - ei + ej)
                f_mm = neg_log_lik(params - ei - ej)
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
                H[j, i] = H[i, j]

        try:
            V = np.linalg.inv(H)
            if not np.all(np.diag(V) > 0):
                V = np.abs(V)
        except np.linalg.LinAlgError:
            V = np.abs(np.linalg.pinv(H))
        return V

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for multinomial logit estimation")
