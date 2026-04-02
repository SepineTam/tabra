#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : tobit.py

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from tabra.models.estimate.base import BaseModel
from tabra.results.tobit_result import TobitResult


class TobitModel(BaseModel):
    """Tobit (censored regression) model estimated via MLE."""

    def fit(self, df, y, x, ll=None, ul=None, is_con=True,
            max_iter=200, tol=1e-8):
        df = self._prepare_df(df, y, x)
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)

        if is_con:
            X = np.column_stack([X, np.ones(X.shape[0])])
            var_names = var_names + ["_cons"]

        n, k = X.shape

        # 确定删失类型
        if ll is not None:
            left_censored = y_vec <= ll
        else:
            left_censored = np.zeros(n, dtype=bool)
        if ul is not None:
            right_censored = y_vec >= ul
        else:
            right_censored = np.zeros(n, dtype=bool)
        uncensored = ~left_censored & ~right_censored

        n_lc = int(np.sum(left_censored))
        n_rc = int(np.sum(right_censored))
        n_unc = int(np.sum(uncensored))

        # Step 1: 拟合仅常数项模型用于 ll_0
        ll_0 = self._fit_null_model(y_vec, n, ll, ul, is_con)

        # Step 2: 优化完整模型
        # 初始值: OLS
        beta_init = np.linalg.lstsq(X, y_vec, rcond=None)[0]
        resid = y_vec - X @ beta_init
        sigma_init = np.std(resid, ddof=k)
        sigma_init = max(sigma_init, 0.1)

        params_init = np.append(beta_init, np.log(sigma_init))

        result_opt = minimize(
            fun=lambda p: self._neg_log_lik(p, y_vec, X, n, ll, ul),
            x0=params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
        )

        params = result_opt.x
        beta = params[:k]
        sigma = np.exp(params[k])
        converged = result_opt.success

        ll_val = -result_opt.fun

        # Step 3: 标准误 via 数值 Hessian
        V = self._compute_vce(params, y_vec, X, n, ll, ul, k)
        std_err = np.sqrt(np.abs(np.diag(V)))
        se_beta = std_err[:k]
        se_lnsigma = std_err[k]

        # sigma 的标准误通过 delta method: se(sigma) = sigma * se(ln_sigma)
        se_sigma = sigma * se_lnsigma

        # t 统计量
        t_stat = beta / se_beta
        p_value = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), df=n - k))

        # LR 检验
        df_m = k - 1 if is_con else k
        chi2 = 2 * (ll_val - ll_0)
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        # 伪 R²
        pseudo_r2 = 1 - ll_val / ll_0 if ll_0 != 0 else 0.0

        return TobitResult(
            coef=beta,
            std_err=se_beta,
            t_stat=t_stat,
            p_value=p_value,
            sigma=sigma,
            var_e=sigma ** 2,
            se_sigma=se_sigma,
            ll=ll_val,
            ll_0=ll_0,
            pseudo_r2=pseudo_r2,
            chi2=chi2,
            chi2_pval=chi2_pval,
            n_obs=n,
            n_unc=n_unc,
            n_lc=n_lc,
            n_rc=n_rc,
            k_vars=k,
            df_m=df_m,
            var_names=var_names,
            y_name=y,
            converged=converged,
            ll_limit=ll,
            ul_limit=ul,
        )

    def _neg_log_lik(self, params, y, X, n, ll, ul):
        """Compute negative log-likelihood for tobit model."""
        k = X.shape[1]
        beta = params[:k]
        ln_sigma = params[k]
        sigma = np.exp(ln_sigma)

        xb = X @ beta

        ll_val = 0.0

        if ll is not None:
            mask_lc = y <= ll
            if np.any(mask_lc):
                z_lc = (ll - xb[mask_lc]) / sigma
                z_lc = np.clip(z_lc, -30, 30)
                ll_val += np.sum(np.log(sp_stats.norm.cdf(z_lc) + 1e-300))

        if ul is not None:
            mask_rc = y >= ul
            if np.any(mask_rc):
                z_rc = (ul - xb[mask_rc]) / sigma
                z_rc = np.clip(z_rc, -30, 30)
                ll_val += np.sum(np.log(1 - sp_stats.norm.cdf(z_rc) + 1e-300))

        # 无删失的观测
        mask_unc = np.ones(n, dtype=bool)
        if ll is not None:
            mask_unc &= (y > ll)
        if ul is not None:
            mask_unc &= (y < ul)

        if np.any(mask_unc):
            ll_val += np.sum(
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma)
                - 0.5 * ((y[mask_unc] - xb[mask_unc]) / sigma) ** 2
            )

        return -ll_val

    def _fit_null_model(self, y, n, ll, ul, is_con):
        """Fit constant-only model to get ll_0."""
        # 仅常数项模型：只有截距，X = np.ones(n)
        X_null = np.ones((n, 1))

        # 初始值
        y_mean = np.mean(y)
        resid = y - y_mean
        sigma_init = np.std(resid, ddof=1)
        sigma_init = max(sigma_init, 0.1)

        params_init = np.array([y_mean, np.log(sigma_init)])

        def neg_ll_null(params):
            beta0 = params[0]
            sigma = np.exp(params[1])

            ll_val = 0.0
            if ll is not None:
                mask_lc = y <= ll
                if np.any(mask_lc):
                    z = (ll - beta0) / sigma
                    z = np.clip(z, -30, 30)
                    ll_val += np.sum(mask_lc) * np.log(
                        sp_stats.norm.cdf(z) + 1e-300)

            if ul is not None:
                mask_rc = y >= ul
                if np.any(mask_rc):
                    z = (ul - beta0) / sigma
                    z = np.clip(z, -30, 30)
                    ll_val += np.sum(mask_rc) * np.log(
                        1 - sp_stats.norm.cdf(z) + 1e-300)

            mask_unc = np.ones(n, dtype=bool)
            if ll is not None:
                mask_unc &= (y > ll)
            if ul is not None:
                mask_unc &= (y < ul)
            n_unc = np.sum(mask_unc)
            if n_unc > 0:
                ll_val += np.sum(
                    -0.5 * np.log(2 * np.pi)
                    - np.log(sigma)
                    - 0.5 * ((y[mask_unc] - beta0) / sigma) ** 2
                )

            return -ll_val

        result = minimize(neg_ll_null, params_init, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12})
        return -result.fun

    def _compute_vce(self, params, y, X, n, ll, ul, k):
        """Compute variance-covariance matrix via numerical Hessian."""
        eps = 1e-5
        p = len(params)
        hess = np.zeros((p, p))

        f0 = self._neg_log_lik(params, y, X, n, ll, ul)

        for i in range(p):
            for j in range(i, p):
                ei = np.zeros(p)
                ej = np.zeros(p)
                ei[i] = eps
                ej[j] = eps
                f_pp = self._neg_log_lik(params + ei + ej, y, X, n, ll, ul)
                f_pm = self._neg_log_lik(params + ei - ej, y, X, n, ll, ul)
                f_mp = self._neg_log_lik(params - ei + ej, y, X, n, ll, ul)
                f_mm = self._neg_log_lik(params - ei - ej, y, X, n, ll, ul)
                hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
                hess[j, i] = hess[i, j]

        try:
            V = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            V = np.eye(p) * np.nan

        return V

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for tobit estimation")
