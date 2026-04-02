#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : panel.py

import numpy as np
from tabra.models.estimate.base import BaseModel
from tabra.ops.linalg import mat_mul, mat_transpose, mat_inv
from tabra.ops.stats import t_pval, f_pval


class PanelModel(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(self, df, y, x, panel_var, model="fe", is_con=True):
        df = self._prepare_df(df, y, x, extra_cols=[panel_var])
        dispatch = {
            "fe": self._fit_fe,
            "re": self._fit_re,
            "be": self._fit_be,
            "mle": self._fit_mle,
            "pa": self._fit_pa,
        }
        fitter = dispatch.get(model)
        if fitter is None:
            raise ValueError(f"Unknown model: {model}")
        return fitter(df, y, x, panel_var, is_con=is_con)

    def _fit_fe(self, df, y, x, panel_var, is_con=True):
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        groups = df[panel_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n, k = X.shape

        # Within transformation: demean by group
        y_dm = np.zeros(n)
        X_dm = np.zeros((n, k))
        for g in unique_groups:
            mask = groups == g
            y_dm[mask] = y_vec[mask] - y_vec[mask].mean()
            for j in range(k):
                X_dm[mask, j] = X[mask, j] - X[mask, j].mean()

        # OLS on demeaned data (no constant — FE absorbs it)
        XtX = mat_mul(mat_transpose(X_dm), X_dm)
        Xty = mat_mul(mat_transpose(X_dm), y_dm.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta = mat_mul(XtX_inv, Xty).flatten()

        resid_within = y_dm - X_dm @ beta
        SSR_w = float(resid_within @ resid_within)
        df_resid = n - n_groups - k
        sigma2_e = SSR_w / df_resid
        var_beta = sigma2_e * XtX_inv
        std_err = np.sqrt(np.diag(var_beta))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        # Fitted values and residuals (overall)
        fitted = X_dm @ beta
        resid = y_dm - fitted

        # Overall SST on demeaned y
        y_dm_mean = y_dm.mean()
        SST_w = float((y_dm - y_dm_mean) @ (y_dm - y_dm_mean))
        SSE_w = SST_w - SSR_w
        r_squared = 1 - SSR_w / SST_w if SST_w > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0

        df_model = k
        f_stat = (SSE_w / df_model) / (SSR_w / df_resid) if df_model > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        # sigma_u: between-group variance component
        group_means_y = np.array([y_vec[groups == g].mean() for g in unique_groups])
        group_means_X = np.array([X[groups == g].mean(axis=0) for g in unique_groups])
        X_between = np.column_stack([np.ones(n_groups), group_means_X])
        beta_b = np.linalg.lstsq(X_between, group_means_y, rcond=None)[0]
        resid_b = group_means_y - X_between @ beta_b
        SSR_b = float(resid_b @ resid_b)
        T_bar = n / n_groups
        sigma2_u = max(0.0, SSR_b / (n_groups - k - 1) - sigma2_e / T_bar)
        sigma_u = np.sqrt(sigma2_u)
        sigma_e = np.sqrt(sigma2_e)
        rho = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0.0

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="fe",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid, fitted=fitted, n_obs=n, k_vars=k,
            var_names=var_names,
            SSR=SSR_w, SSE=SSE_w, SST=SST_w,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2_e, root_mse=sigma_e,
            sigma_u=sigma_u, sigma_e=sigma_e, rho=rho,
            y_name=y, n_groups=n_groups,
        )

    def _fit_re(self, df, y, x, panel_var, is_con=True):
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        groups = df[panel_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n, k = X.shape

        # Step 1: Within (FE) variance
        y_dm = np.zeros(n)
        X_dm = np.zeros((n, k))
        for g in unique_groups:
            mask = groups == g
            y_dm[mask] = y_vec[mask] - y_vec[mask].mean()
            for j in range(k):
                X_dm[mask, j] = X[mask, j] - X[mask, j].mean()

        beta_fe = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        resid_w = y_dm - X_dm @ beta_fe
        SSR_w = float(resid_w @ resid_w)
        df_w = n - n_groups - k
        sigma2_e = SSR_w / df_w

        # Step 2: Between variance
        gm_y = np.array([y_vec[groups == g].mean() for g in unique_groups])
        gm_X = np.array([X[groups == g].mean(axis=0) for g in unique_groups])
        X_b = np.column_stack([np.ones(n_groups), gm_X])
        beta_b = np.linalg.lstsq(X_b, gm_y, rcond=None)[0]
        resid_b = gm_y - X_b @ beta_b
        SSR_b = float(resid_b @ resid_b)
        T_bar = n / n_groups
        sigma2_u = max(0.0, SSR_b / (n_groups - k - 1) - sigma2_e / T_bar)

        # Step 3: Theta (Swamy-Arora)
        theta = 1.0 - np.sqrt(sigma2_e / (sigma2_e + T_bar * sigma2_u))

        # Step 4: Quasi-demeaned data
        y_qd = np.zeros(n)
        X_qd = np.zeros((n, k))
        for i, g in enumerate(unique_groups):
            mask = groups == g
            y_qd[mask] = y_vec[mask] - theta * gm_y[i]
            for j in range(k):
                X_qd[mask, j] = X[mask, j] - theta * gm_X[i, j]

        if is_con:
            X_qd_full = np.column_stack([np.ones(n), X_qd])
            var_names = var_names + ["_cons"]
        else:
            X_qd_full = X_qd

        n_qd, k_qd = X_qd_full.shape
        XtX = mat_mul(mat_transpose(X_qd_full), X_qd_full)
        Xty = mat_mul(mat_transpose(X_qd_full), y_qd.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta = mat_mul(XtX_inv, Xty).flatten()

        resid_re = y_qd - X_qd_full @ beta
        SSR_re = float(resid_re @ resid_re)
        df_resid = n_qd - k_qd
        sigma2_re = SSR_re / df_resid
        var_beta = sigma2_re * XtX_inv
        std_err = np.sqrt(np.diag(var_beta))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        y_qd_mean = y_qd.mean()
        SST_re = float((y_qd - y_qd_mean) @ (y_qd - y_qd_mean))
        SSE_re = SST_re - SSR_re
        r_squared = 1 - SSR_re / SST_re if SST_re > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0

        df_model = k_qd - 1 if is_con else k_qd
        f_stat = (SSE_re / df_model) / (SSR_re / df_resid) if df_model > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        sigma_u = np.sqrt(sigma2_u)
        sigma_e = np.sqrt(sigma2_e)
        rho = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0.0

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="re",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid_re, fitted=X_qd_full @ beta, n_obs=n, k_vars=k_qd,
            var_names=var_names,
            SSR=SSR_re, SSE=SSE_re, SST=SST_re,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2_re, root_mse=np.sqrt(sigma2_re),
            sigma_u=sigma_u, sigma_e=sigma_e, rho=rho,
            y_name=y, theta=theta,
        )

    def _fit_be(self, df, y, x, panel_var, is_con=True):
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        groups = df[panel_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Compute group means
        gm_y = np.array([y_vec[groups == g].mean() for g in unique_groups])
        gm_X = np.array([X[groups == g].mean(axis=0) for g in unique_groups])

        # OLS on group means
        if is_con:
            X_b = np.column_stack([np.ones(n_groups), gm_X])
            var_names = var_names + ["_cons"]
        else:
            X_b = gm_X

        n_b, k_b = X_b.shape

        XtX = mat_mul(mat_transpose(X_b), X_b)
        Xty = mat_mul(mat_transpose(X_b), gm_y.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta = mat_mul(XtX_inv, Xty).flatten()

        resid_b = gm_y - X_b @ beta
        SSR_b = float(resid_b @ resid_b)
        df_resid = n_groups - k_b
        sigma2 = SSR_b / df_resid
        var_beta = sigma2 * XtX_inv
        std_err = np.sqrt(np.diag(var_beta))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        y_mean = gm_y.mean()
        SST_b = float((gm_y - y_mean) @ (gm_y - y_mean))
        SSE_b = SST_b - SSR_b
        r_squared = 1 - SSR_b / SST_b if SST_b > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n_groups - 1) / df_resid if df_resid > 0 else 0.0

        df_model = k_b - 1 if is_con else k_b
        f_stat = (SSE_b / df_model) / (SSR_b / df_resid) if df_model > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="be",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid_b, fitted=X_b @ beta, n_obs=n_groups, k_vars=k_b,
            var_names=var_names,
            SSR=SSR_b, SSE=SSE_b, SST=SST_b,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2, root_mse=np.sqrt(sigma2),
            sigma_u=None, sigma_e=None, rho=None,
            y_name=y, n_groups=n_groups,
        )

    def _fit_mle(self, df, y, x, panel_var, is_con=True):
        from scipy.optimize import minimize_scalar

        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        groups = df[panel_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n, k = X.shape
        T_bar = n / n_groups

        gm_y = np.array([y_vec[groups == g].mean() for g in unique_groups])
        gm_X = np.array([X[groups == g].mean(axis=0) for g in unique_groups])

        def _neg_cloglik(theta):
            if theta <= 0 or theta >= 1:
                return 1e20
            y_qd = np.zeros(n)
            X_qd = np.zeros((n, k))
            for i, g in enumerate(unique_groups):
                mask = groups == g
                y_qd[mask] = y_vec[mask] - theta * gm_y[i]
                for j in range(k):
                    X_qd[mask, j] = X[mask, j] - theta * gm_X[i, j]
            X_qd_f = np.column_stack([np.ones(n), X_qd])
            beta = np.linalg.lstsq(X_qd_f, y_qd, rcond=None)[0]
            resid = y_qd - X_qd_f @ beta
            sigma2 = float(resid @ resid) / n
            return n / 2 * np.log(sigma2) - n_groups / 2 * np.log(1 - theta)

        res = minimize_scalar(_neg_cloglik, bounds=(1e-6, 1 - 1e-6), method="bounded")
        theta = res.x

        # GLS with optimal theta
        y_qd = np.zeros(n)
        X_qd = np.zeros((n, k))
        for i, g in enumerate(unique_groups):
            mask = groups == g
            y_qd[mask] = y_vec[mask] - theta * gm_y[i]
            for j in range(k):
                X_qd[mask, j] = X[mask, j] - theta * gm_X[i, j]

        if is_con:
            X_qd_f = np.column_stack([np.ones(n), X_qd])
            var_names = var_names + ["_cons"]
        else:
            X_qd_f = X_qd

        n_qd, k_qd = X_qd_f.shape
        XtX = mat_mul(mat_transpose(X_qd_f), X_qd_f)
        Xty = mat_mul(mat_transpose(X_qd_f), y_qd.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta = mat_mul(XtX_inv, Xty).flatten()

        resid_mle = y_qd - X_qd_f @ beta
        SSR = float(resid_mle @ resid_mle)
        sigma2 = SSR / n
        df_resid = n_qd - k_qd
        var_beta = sigma2 * XtX_inv
        std_err = np.sqrt(np.diag(var_beta))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        y_qd_mean = y_qd.mean()
        SST = float((y_qd - y_qd_mean) @ (y_qd - y_qd_mean))
        SSE = SST - SSR
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0

        df_model = k_qd - 1 if is_con else k_qd
        f_stat = (SSE / df_model) / (SSR / df_resid) if df_model > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        sigma2_e = sigma2 * (1 - theta)
        sigma2_u = sigma2 * theta / T_bar / (1 - theta) if (1 - theta) > 0 else 0.0
        sigma_u = np.sqrt(max(0, sigma2_u))
        sigma_e = np.sqrt(sigma2_e)
        rho = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0.0

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="mle",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid_mle, fitted=X_qd_f @ beta, n_obs=n, k_vars=k_qd,
            var_names=var_names,
            SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2, root_mse=np.sqrt(sigma2),
            sigma_u=sigma_u, sigma_e=sigma_e, rho=rho,
            y_name=y, theta=theta,
        )

    def _fit_pa(self, df, y, x, panel_var, is_con=True):
        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        groups = df[panel_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n, k = X.shape

        if is_con:
            X_full = np.column_stack([np.ones(n), X])
            var_names = var_names + ["_cons"]
        else:
            X_full = X

        k_full = X_full.shape[1]

        # Get T per group
        T_arr = np.array([np.sum(groups == g) for g in unique_groups])

        # Initial OLS
        beta = np.linalg.lstsq(X_full, y_vec, rcond=None)[0]

        for _ in range(100):
            resid = y_vec - X_full @ beta
            sigma2 = float(resid @ resid) / n

            # Estimate exchangeable alpha
            num = 0.0
            den = 0.0
            for i, g in enumerate(unique_groups):
                mask = groups == g
                ri = resid[mask]
                Ti = ri.shape[0]
                for a in range(Ti):
                    for b in range(a + 1, Ti):
                        num += ri[a] * ri[b]
                        den += 1.0
            alpha_corr = num / den / sigma2 if den > 0 and sigma2 > 0 else 0.0
            alpha_corr = max(-1.0 / (max(T_arr) - 1), min(alpha_corr, 0.999))

            # GLS update
            rho_inv = alpha_corr / (1 - alpha_corr + T_arr.mean() * alpha_corr)
            XtX = np.zeros((k_full, k_full))
            Xty = np.zeros(k_full)
            for i, g in enumerate(unique_groups):
                mask = groups == g
                Xi = X_full[mask]
                yi = y_vec[mask]
                Ti = mask.sum()
                Xi_sum = Xi.sum(axis=0)
                Xi_tilde = Xi - rho_inv * np.outer(np.ones(Ti), Xi_sum)
                yi_sum = yi.sum()
                yi_tilde = yi - rho_inv * yi_sum * np.ones(Ti)
                XtX += Xi_tilde.T @ Xi
                Xty += Xi_tilde.T @ yi_tilde
            beta_new = np.linalg.solve(XtX, Xty)
            if np.allclose(beta_new, beta, atol=1e-10):
                beta = beta_new
                break
            beta = beta_new

        # Final stats on quasi-demeaned scale
        resid_final = y_vec - X_full @ beta
        sigma2_final = float(resid_final @ resid_final) / n

        df_resid = n - k_full
        SST = float((y_vec - y_vec.mean()) @ (y_vec - y_vec.mean()))
        SSR = float(resid_final @ resid_final)
        SSE = SST - SSR
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0

        df_model = k_full - 1 if is_con else k_full
        f_stat = (SSE / df_model) / (SSR / df_resid) if df_model > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        # Robust SE: sandwich estimator approach for PA
        # For simplicity, use OLS-like SE on the full data
        XtX_full = X_full.T @ X_full
        XtX_inv_full = np.linalg.inv(XtX_full)
        std_err = np.sqrt(sigma2_final * np.diag(XtX_inv_full))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="pa",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid_final, fitted=X_full @ beta, n_obs=n, k_vars=k_full,
            var_names=var_names,
            SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2_final, root_mse=np.sqrt(sigma2_final),
            sigma_u=None, sigma_e=None, rho=None,
            y_name=y, n_groups=n_groups,
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for panel estimation")
