#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ordered_choice.py

import numpy as np
from abc import abstractmethod
from scipy import stats as sp_stats
from scipy.optimize import minimize
from scipy.special import expit

from tabra.models.estimate.base import BaseModel
from tabra.results.ordered_choice_result import OrderedChoiceResult


class OrderedChoiceModel(BaseModel):
    """Base class for ordered choice models (oprobit/ologit) estimated via MLE."""

    def fit(self, df, y, x, is_con=True, max_iter=300, tol=1e-8):
        """Fit an ordered choice model via MLE.

        Args:
            df (pd.DataFrame): Input dataset.
            y (str): Ordinal dependent variable name.
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.
            max_iter (int): Maximum optimizer iterations. Default 300.
            tol (float): Convergence tolerance. Default 1e-8.

        Returns:
            OrderedChoiceResult: Estimation result.
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
        n_cat = len(categories)
        # Map y to 0, 1, 2, ...
        cat_map = {c: i for i, c in enumerate(categories)}
        y_mapped = np.array([cat_map[v] for v in y_vec])

        # Fit null model (cutpoints only, no x)
        ll_0 = self._fit_null_model(y_mapped, n_cat)

        # Fit full model via MLE
        # Params: [beta(k), cutpoints(n_cat-1)]
        n_cuts = n_cat - 1

        # Initialize: OLS on midpoints
        midpoints = np.array([(i + 0.5) / n_cat for i in range(n_cat)])
        y_mid = midpoints[y_mapped]
        beta_init = np.linalg.lstsq(X, y_mid, rcond=None)[0]
        resid = y_mid - X @ beta_init
        sd = max(np.std(resid), 0.1)

        # Initialize cutpoints from residual quantiles
        cuts_init = np.quantile(resid, np.linspace(0.2, 0.8, n_cuts))
        # Ensure strict ordering
        for i in range(1, len(cuts_init)):
            if cuts_init[i] <= cuts_init[i - 1]:
                cuts_init[i] = cuts_init[i - 1] + 0.1

        params_init = np.concatenate([beta_init, cuts_init])

        # Optimize
        result_opt = minimize(
            fun=lambda p: self._neg_log_likelihood(p, y_mapped, X, k, n_cat),
            x0=params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
        )

        params = result_opt.x
        beta = params[:k]
        cutpoints = params[k:k + n_cuts]

        ll = -result_opt.fun
        converged = result_opt.success

        # Standard errors via numerical Hessian
        V = self._compute_vce(params, y_mapped, X, k, n_cat)
        n_params = len(params)
        std_err = np.sqrt(np.maximum(np.diag(V), 0))

        # Split SE
        se_beta = std_err[:k]
        se_cuts = std_err[k:k + n_cuts]

        z_stat = params / std_err
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # Pseudo R2
        pseudo_r2 = 1 - ll / ll_0 if ll_0 != 0 else 0.0

        # LR chi2
        df_m = k - 1 if is_con else k
        chi2 = 2 * (ll - ll_0)
        chi2 = max(chi2, 0)
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        return OrderedChoiceResult(
            coef=beta,
            std_err=se_beta,
            z_stat=z_stat[:k],
            p_value=p_value[:k],
            cutpoints=cutpoints,
            cutpoint_se=se_cuts,
            cutpoint_z=z_stat[k:k + n_cuts],
            cutpoint_p=p_value[k:k + n_cuts],
            ll=ll,
            ll_0=ll_0,
            pseudo_r2=pseudo_r2,
            chi2=chi2,
            chi2_pval=chi2_pval,
            n_obs=n,
            k_vars=k,
            k_cat=n_cat,
            df_m=df_m,
            var_names=var_names,
            y_name=y,
            categories=categories,
            converged=converged,
            model_name=self._model_name(),
            V=V,
        )

    def _neg_log_likelihood(self, params, y, X, k, n_cat):
        beta = params[:k]
        cuts = params[k:k + n_cat - 1]

        # Penalize non-ordered cutpoints
        for i in range(1, len(cuts)):
            if cuts[i] <= cuts[i - 1]:
                return 1e15

        xb = X @ beta
        ll = 0.0
        for i in range(len(y)):
            prob = self._interval_prob(y[i], cuts, xb[i])
            ll += np.log(max(prob, 1e-300))
        return -ll

    def _fit_null_model(self, y, n_cat):
        """Fit cutpoints-only model for ll_0."""
        n = len(y)
        counts = np.bincount(y, minlength=n_cat)
        probs = counts / n

        # For null model, xb = 0 for all obs
        # Just use empirical probabilities
        ll_0 = 0.0
        for i in range(n):
            ll_0 += np.log(max(probs[y[i]], 1e-300))
        return ll_0

    def _compute_vce(self, params, y, X, k, n_cat):
        """Numerical Hessian for VCE."""
        eps = 1e-5
        p = len(params)
        H = np.zeros((p, p))
        f0 = self._neg_log_likelihood(params, y, X, k, n_cat)

        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps
            fi_plus = self._neg_log_likelihood(params + ei, y, X, k, n_cat)
            fi_minus = self._neg_log_likelihood(params - ei, y, X, k, n_cat)
            H[i, i] = (fi_plus - 2 * f0 + fi_minus) / (eps ** 2)

            for j in range(i + 1, p):
                ej = np.zeros(p)
                ej[j] = eps
                f_pp = self._neg_log_likelihood(params + ei + ej, y, X, k, n_cat)
                f_pm = self._neg_log_likelihood(params + ei - ej, y, X, k, n_cat)
                f_mp = self._neg_log_likelihood(params - ei + ej, y, X, k, n_cat)
                f_mm = self._neg_log_likelihood(params - ei - ej, y, X, k, n_cat)
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
                H[j, i] = H[i, j]

        try:
            V = np.linalg.inv(H)
            if not np.all(np.diag(V) > 0):
                V = np.abs(V)
        except np.linalg.LinAlgError:
            V = np.abs(np.linalg.pinv(H))
        return V

    @abstractmethod
    def _interval_prob(self, j, cuts, xb):
        """P(y = j | xb) for ordered outcome j."""
        raise NotImplementedError

    @abstractmethod
    def _model_name(self):
        raise NotImplementedError


class OrderedProbitModel(OrderedChoiceModel):
    """Ordered probit: P(y=j|x) = Phi(cut_j - xb) - Phi(cut_{j-1} - xb)"""

    def _interval_prob(self, j, cuts, xb):
        n_cat = len(cuts) + 1
        if j == 0:
            return sp_stats.norm.cdf(cuts[0] - xb)
        elif j == n_cat - 1:
            return 1 - sp_stats.norm.cdf(cuts[-1] - xb)
        else:
            return sp_stats.norm.cdf(cuts[j] - xb) - sp_stats.norm.cdf(cuts[j - 1] - xb)

    def _model_name(self):
        return "Ordered probit regression"

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for ordered probit estimation")


class OrderedLogitModel(OrderedChoiceModel):
    """Ordered logit: P(y=j|x) = F(cut_j - xb) - F(cut_{j-1} - xb) where F = logistic CDF"""

    def _interval_prob(self, j, cuts, xb):
        n_cat = len(cuts) + 1
        if j == 0:
            return expit(cuts[0] - xb)
        elif j == n_cat - 1:
            return 1 - expit(cuts[-1] - xb)
        else:
            return expit(cuts[j] - xb) - expit(cuts[j - 1] - xb)

    def _model_name(self):
        return "Ordered logistic regression"

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for ordered logit estimation")
