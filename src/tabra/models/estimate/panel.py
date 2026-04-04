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
        """Fit a panel data model.

        Args:
            df (pd.DataFrame): Panel dataset in long format.
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            panel_var (str): Column name for the entity identifier.
            model (str): Estimation method. One of "fe", "re", "be", "mle", "pa". Default "fe".
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            PanelResult: Panel estimation result.

        Example:
            >>> dta = load_data("nlswork")
            >>> dta.xtset("idcode", "year")
            >>> result = PanelModel().fit(dta._df, "ln_wage", ["age", "tenure"], "idcode", model="fe")
        """
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

        # sigma_u: standard deviation of estimated fixed effects
        group_means_y = np.array([y_vec[groups == g].mean() for g in unique_groups])
        group_means_X = np.array([X[groups == g].mean(axis=0) for g in unique_groups])
        alpha_hat = group_means_y - group_means_X @ beta
        sigma2_u = float(np.var(alpha_hat, ddof=1))
        sigma_u = np.sqrt(sigma2_u)
        sigma_e = np.sqrt(sigma2_e)
        rho = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0.0

        # FE _cons: average fixed effect (grand mean)
        overall_y_mean = float(y_vec.mean())
        overall_X_mean = X.mean(axis=0)
        cons_fe = overall_y_mean - overall_X_mean @ beta
        var_names = list(x) + ["_cons"]
        beta = np.append(beta, cons_fe)

        # _cons SE
        se_cons = np.sqrt(sigma2_e / n + overall_X_mean @ var_beta @ overall_X_mean)
        std_err = np.append(std_err, se_cons)
        t_cons = cons_fe / se_cons
        t_stat = np.append(t_stat, t_cons)
        p_cons = t_pval(t_cons, df_resid)
        p_value = np.append(p_value, p_cons)

        k_vars = k + 1  # include _cons

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="fe",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid, fitted=fitted, n_obs=n, k_vars=k_vars,
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
        T_i = np.array([np.sum(groups == g) for g in unique_groups])
        T_bar_harmonic = n_groups / np.sum(1.0 / T_i)
        sigma2_u = max(0.0, SSR_b / (n_groups - k - 1) - sigma2_e / T_bar_harmonic)

        # Step 3: Theta (Swamy-Arora) — per-group for unbalanced panels
        theta_i = 1.0 - np.sqrt(sigma2_e / (sigma2_e + T_i * sigma2_u))

        # Step 4: Quasi-demeaned data (per-group theta)
        y_qd = np.zeros(n)
        X_qd = np.zeros((n, k))
        for i, g in enumerate(unique_groups):
            mask = groups == g
            y_qd[mask] = y_vec[mask] - theta_i[i] * gm_y[i]
            for j in range(k):
                X_qd[mask, j] = X[mask, j] - theta_i[i] * gm_X[i, j]

        if is_con:
            ones_qd = np.zeros(n)
            for i, g in enumerate(unique_groups):
                mask = groups == g
                ones_qd[mask] = 1.0 - theta_i[i]
            X_qd_full = np.column_stack([ones_qd, X_qd])
            var_names = ["_cons"] + var_names
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

        # Wald chi2 (slopes only, full covariance matrix)
        if is_con:
            beta_slopes = beta[1:]
            V_slopes = var_beta[1:, 1:]
        else:
            beta_slopes = beta
            V_slopes = var_beta
        wald_chi2 = float(beta_slopes @ np.linalg.inv(V_slopes) @ beta_slopes) if k > 0 else 0.0

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
            y_name=y, theta=theta_i.mean(), n_groups=n_groups,
            chi2_stat=wald_chi2, chi2_label=f"Wald chi2({k})",
        )

    def _fit_be(self, df, y, x, panel_var, is_con=True):
        y_vec = df[y].values.astype(float)
        n = len(y_vec)
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
            var_names = ["_cons"] + var_names
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
            resid=resid_b, fitted=X_b @ beta, n_obs=n, k_vars=k_b,
            var_names=var_names,
            SSR=SSR_b, SSE=SSE_b, SST=SST_b,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2, root_mse=np.sqrt(sigma2),
            sigma_u=None, sigma_e=None, rho=None,
            y_name=y, n_groups=n_groups,
        )

    def _fit_mle(self, df, y, x, panel_var, is_con=True):
        from scipy.optimize import minimize

        y_vec = df[y].values.astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        groups = df[panel_var].values
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n, k = X.shape

        # Pre-compute per-group data
        T_i = np.array([np.sum(groups == g) for g in unique_groups])
        gm_y = np.array([y_vec[groups == g].mean() for g in unique_groups])
        gm_X = np.array([X[groups == g].mean(axis=0) for g in unique_groups])

        # Build index arrays for fast group access
        group_indices = []
        for g in unique_groups:
            group_indices.append(np.where(groups == g)[0])

        def _gls_solution(sigma2_u_val, sigma2_e_val):
            """Given variance components, compute GLS estimator."""
            phi_i = sigma2_u_val / (sigma2_e_val + T_i * sigma2_u_val)
            theta_i = 1.0 - np.sqrt(sigma2_e_val / (sigma2_e_val + T_i * sigma2_u_val))

            # Quasi-demean
            y_qd = np.zeros(n)
            X_qd = np.zeros((n, k))
            for idx in range(n_groups):
                gi = group_indices[idx]
                y_qd[gi] = y_vec[gi] - theta_i[idx] * gm_y[idx]
                for j in range(k):
                    X_qd[gi, j] = X[gi, j] - theta_i[idx] * gm_X[idx, j]

            if is_con:
                ones_qd = np.zeros(n)
                for idx in range(n_groups):
                    gi = group_indices[idx]
                    ones_qd[gi] = 1.0 - theta_i[idx]
                X_qd_full = np.column_stack([ones_qd, X_qd])
            else:
                X_qd_full = X_qd

            beta = np.linalg.lstsq(X_qd_full, y_qd, rcond=None)[0]
            return beta, X_qd_full, y_qd

        def _neg_profile_loglik(params):
            """Negative profile log-likelihood for RE MLE.

            Reference: Baltagi (2021), Econometric Analysis of Panel Data, Ch. 3.
            """
            log_su, log_se = params
            sigma2_u_val = np.exp(log_su)
            sigma2_e_val = np.exp(log_se)

            if sigma2_u_val < 1e-20 or sigma2_e_val < 1e-20:
                return 1e20

            beta, _, _ = _gls_solution(sigma2_u_val, sigma2_e_val)

            total_loglik = 0.0
            for idx in range(n_groups):
                gi = group_indices[idx]
                Ti = T_i[idx]
                yi = y_vec[gi]
                if is_con:
                    Xi = np.column_stack([np.ones(Ti), X[gi]])
                else:
                    Xi = X[gi]

                resid_i = yi - Xi @ beta

                log_det = (Ti - 1) * np.log(sigma2_e_val) + np.log(sigma2_e_val + Ti * sigma2_u_val)

                phi_i = sigma2_u_val / (sigma2_e_val + Ti * sigma2_u_val)
                resid_sum = resid_i.sum()
                quad_form = (resid_i @ resid_i - phi_i * resid_sum**2) / sigma2_e_val

                log_lik_i = -Ti / 2 * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad_form
                total_loglik += log_lik_i

            return -total_loglik

        # Initial values from Swamy-Arora
        y_dm = np.zeros(n)
        X_dm = np.zeros((n, k))
        for g_idx, g in enumerate(unique_groups):
            mask = groups == g
            y_dm[mask] = y_vec[mask] - y_vec[mask].mean()
            for j in range(k):
                X_dm[mask, j] = X[mask, j] - X[mask, j].mean()
        beta_fe = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        resid_w = y_dm - X_dm @ beta_fe
        SSR_w = float(resid_w @ resid_w)
        df_w = n - n_groups - k
        sigma2_e_init = SSR_w / df_w

        resid_b = gm_y - gm_X @ beta_fe
        SSR_b = float(resid_b @ resid_b)
        T_bar_harmonic = n_groups / np.sum(1.0 / T_i)
        sigma2_u_init = max(0.01, SSR_b / (n_groups - k - 1) - sigma2_e_init / T_bar_harmonic)

        x0 = [np.log(sigma2_u_init), np.log(sigma2_e_init)]
        result = minimize(_neg_profile_loglik, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        # Powell refinement for higher precision
        result = minimize(_neg_profile_loglik, result.x, method='Powell',
                          options={'maxiter': 10000, 'ftol': 1e-15, 'xtol': 1e-15})

        sigma2_u_final = np.exp(result.x[0])
        sigma2_e_final = np.exp(result.x[1])
        sigma_u = np.sqrt(sigma2_u_final)
        sigma_e = np.sqrt(sigma2_e_final)
        rho = sigma2_u_final / (sigma2_u_final + sigma2_e_final)

        # Final GLS with optimal variance components
        beta_final, X_qd_full, y_qd = _gls_solution(sigma2_u_final, sigma2_e_final)

        if is_con:
            var_names = ["_cons"] + var_names

        n_qd, k_qd = X_qd_full.shape
        XtX = mat_mul(mat_transpose(X_qd_full), X_qd_full)
        Xty = mat_mul(mat_transpose(X_qd_full), y_qd.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta = mat_mul(XtX_inv, Xty).flatten()

        resid_mle = y_qd - X_qd_full @ beta
        SSR = float(resid_mle @ resid_mle)
        df_resid = n_qd - k_qd
        sigma2 = SSR / df_resid
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

        # Log-likelihood for full model
        ll_full = -result.fun  # result.fun = neg_profile_loglik at optimum

        # Restricted model: intercept-only RE MLE
        # Stata optimizes mu as a free parameter alongside variance components
        y_mean_global = float(y_vec.mean())

        def _neg_loglik_restricted(params_r):
            log_su, log_se, mu = params_r
            s2u = np.exp(log_su)
            s2e = np.exp(log_se)
            if s2u < 1e-20 or s2e < 1e-20:
                return 1e20
            total_ll = 0.0
            for idx in range(n_groups):
                gi = group_indices[idx]
                Ti = T_i[idx]
                yi = y_vec[gi]
                resid_i = yi - mu
                log_det = (Ti - 1) * np.log(s2e) + np.log(s2e + Ti * s2u)
                phi_i = s2u / (s2e + Ti * s2u)
                resid_sum = resid_i.sum()
                quad = (resid_i @ resid_i - phi_i * resid_sum ** 2) / s2e
                total_ll += -Ti / 2 * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad
            return -total_ll

        x0_r = [np.log(sigma2_u_final), np.log(sigma2_e_final), y_mean_global]
        result_r = minimize(_neg_loglik_restricted, x0_r, method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        result_r = minimize(_neg_loglik_restricted, result_r.x, method='Powell',
                            options={'maxiter': 10000, 'ftol': 1e-15, 'xtol': 1e-15})
        ll_restricted = -result_r.fun
        lr_chi2 = max(0.0, -2.0 * (ll_restricted - ll_full))

        from tabra.results.panel_result import PanelResult
        return PanelResult(
            model_type="mle",
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid_mle, fitted=X_qd_full @ beta, n_obs=n, k_vars=k_qd,
            var_names=var_names,
            SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_resid,
            mse=sigma2, root_mse=np.sqrt(sigma2),
            sigma_u=sigma_u, sigma_e=sigma_e, rho=rho,
            y_name=y, theta=1.0 - np.sqrt(sigma2_e_final / (sigma2_e_final + np.mean(T_i) * sigma2_u_final)),
            n_groups=n_groups,
            chi2_stat=lr_chi2, chi2_label=f"LR chi2({k})",
            ll=ll_full,
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
            var_names = ["_cons"] + var_names
        else:
            X_full = X

        k_full = X_full.shape[1]
        T_i = np.array([np.sum(groups == g) for g in unique_groups])

        # Build index arrays
        group_indices = []
        for g in unique_groups:
            group_indices.append(np.where(groups == g)[0])

        # Initial OLS
        beta = np.linalg.lstsq(X_full, y_vec, rcond=None)[0]

        for iteration in range(500):
            resid = y_vec - X_full @ beta
            phi = float(resid @ resid) / n

            if phi <= 0:
                break

            # Estimate exchangeable correlation alpha (vectorized)
            num = 0.0
            den = 0.0
            for idx in range(n_groups):
                gi = group_indices[idx]
                ri = resid[gi]
                Ti = len(gi)
                ri_sum = ri.sum()
                ri_sq_sum = (ri ** 2).sum()
                num += (ri_sum ** 2 - ri_sq_sum) / 2.0
                den += Ti * (Ti - 1) / 2.0

            alpha_corr = num / den / phi if den > 0 and phi > 0 else 0.0
            T_max = T_i.max()
            alpha_corr = max(-1.0 / (T_max - 1) if T_max > 1 else 0.0,
                             min(alpha_corr, 0.9999))

            # GEE update: correct normal equation
            # sum_i X_i' [I - rho_inv * J] X_i and X_i' [I - rho_inv * J] y_i
            H = np.zeros((k_full, k_full))
            g = np.zeros(k_full)
            for idx in range(n_groups):
                gi = group_indices[idx]
                Xi = X_full[gi]
                yi = y_vec[gi]
                Ti = len(gi)

                rho_inv_i = alpha_corr / (1 - alpha_corr + Ti * alpha_corr)

                XiX = Xi.T @ Xi
                Xiy = Xi.T @ yi
                si = Xi.sum(axis=0)  # column sums = X'1
                yi_sum = yi.sum()    # 1'y

                H += XiX - rho_inv_i * np.outer(si, si)
                g += Xiy - rho_inv_i * si * yi_sum

            beta_new = np.linalg.solve(H, g)
            tol = np.max(np.abs(beta_new - beta) / np.maximum(1.0, np.abs(beta)))
            if tol <= 1e-10:
                beta = beta_new
                break
            beta = beta_new

        # Final statistics
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

        # SE from GEE normal equation (H = sum X_i'[I-rho_inv*J]X_i)
        H_inv = np.linalg.inv(H)
        V_beta = sigma2_final * (1 - alpha_corr) * H_inv
        std_err = np.sqrt(np.diag(V_beta))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        # Wald chi2 (slopes only, full covariance matrix)
        if is_con:
            beta_slopes = beta[1:]
            V_slopes = V_beta[1:, 1:]
        else:
            beta_slopes = beta
            V_slopes = V_beta
        wald_chi2 = float(beta_slopes @ np.linalg.inv(V_slopes) @ beta_slopes) if k > 0 else 0.0

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
            chi2_stat=wald_chi2, chi2_label=f"Wald chi2({k})",
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for panel estimation")
