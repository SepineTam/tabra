#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : heckman.py

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from tabra.models.estimate.base import BaseModel
from tabra.results.heckman_result import HeckmanResult


class HeckmanModel(BaseModel):
    """Heckman selection model estimated via MLE or two-step."""

    def fit(self, df, y, x, select_x, select_var=None,
            is_con=True, method="mle", max_iter=500, tol=1e-8):
        y_vec = df[y].values.astype(float)
        X_outcome = df[x].values.astype(float)
        X_select = df[select_x].values.astype(float)

        outcome_var_names = list(x)
        select_var_names = list(select_x)

        if is_con:
            X_outcome = np.column_stack([X_outcome, np.ones(X_outcome.shape[0])])
            outcome_var_names = outcome_var_names + ["_cons"]
            X_select = np.column_stack([X_select, np.ones(X_select.shape[0])])
            select_var_names = select_var_names + ["_cons"]

        # 确定选择指示变量
        if select_var is not None:
            selected = df[select_var].values.astype(float)
            selected = (selected != 0).astype(float)
        else:
            selected = (~np.isnan(y_vec)).astype(float)

        n = len(y_vec)
        n_selected = int(np.sum(selected))
        n_nonselected = n - n_selected

        if method == "twostep":
            return self._fit_twostep(
                y_vec, X_outcome, X_select, selected,
                n, n_selected, n_nonselected,
                outcome_var_names, select_var_names,
                y, is_con)
        else:
            return self._fit_mle(
                y_vec, X_outcome, X_select, selected,
                n, n_selected, n_nonselected,
                outcome_var_names, select_var_names,
                y, is_con, max_iter, tol)

    def _fit_mle(self, y_vec, X_outcome, X_select, selected,
                 n, n_selected, n_nonselected,
                 outcome_var_names, select_var_names,
                 y_name, is_con, max_iter, tol):
        k_out = X_outcome.shape[1]
        k_sel = X_select.shape[1]

        # 分离选中和未选中
        mask_sel = selected == 1
        mask_nsel = selected == 0
        y_obs = y_vec[mask_sel]
        X_out_obs = X_outcome[mask_sel]
        X_sel_obs = X_select[mask_sel]
        X_sel_nobs = X_select[mask_nsel]

        def log_likelihood(params):
            beta = params[:k_out]
            gamma = params[k_out:k_out + k_sel]
            athrho = params[k_out + k_sel]
            lnsigma = params[k_out + k_sel + 1]

            rho = np.tanh(athrho)
            sigma = np.exp(lnsigma)

            if sigma < 1e-10:
                return -1e15

            # 选中观测
            resid = y_obs - X_out_obs @ beta
            a = resid / sigma
            rho2 = rho ** 2
            denom = np.sqrt(max(1 - rho2, 1e-15))

            arg = (X_sel_obs @ gamma + rho * a) / denom
            arg = np.clip(arg, -20, 20)
            Phi_arg = sp_stats.norm.cdf(arg)
            Phi_arg = np.clip(Phi_arg, 1e-300, 1)

            ll_sel = np.sum(
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma)
                - 0.5 * a ** 2
                + np.log(Phi_arg)
            )

            # 未选中观测
            xb_nobs = X_sel_nobs @ gamma
            xb_nobs = np.clip(xb_nobs, -20, 20)
            Phi_neg = sp_stats.norm.cdf(-xb_nobs)
            Phi_neg = np.clip(Phi_neg, 1e-300, 1)
            ll_nobs = np.sum(np.log(Phi_neg))

            return ll_sel + ll_nobs

        def neg_ll(params):
            return -log_likelihood(params)

        def neg_gradient(params):
            beta = params[:k_out]
            gamma = params[k_out:k_out + k_sel]
            athrho = params[k_out + k_sel]
            lnsigma = params[k_out + k_sel + 1]

            rho = np.tanh(athrho)
            sigma = np.exp(lnsigma)
            rho2 = rho ** 2
            denom = np.sqrt(max(1 - rho2, 1e-15))

            resid = y_obs - X_out_obs @ beta
            a = resid / sigma

            # 选中观测的梯度
            arg = (X_sel_obs @ gamma + rho * a) / denom
            arg = np.clip(arg, -20, 20)
            Phi_arg = sp_stats.norm.cdf(arg)
            Phi_arg = np.clip(Phi_arg, 1e-300, 1)
            phi_arg = sp_stats.norm.pdf(arg)
            lambda_arg = phi_arg / Phi_arg

            # d(ll_sel)/d(beta)
            g_beta = np.sum(
                (a / sigma).reshape(-1, 1) * X_out_obs, axis=0
            ) - (rho / (sigma * denom)) * np.sum(
                lambda_arg.reshape(-1, 1) * X_out_obs, axis=0
            )

            # d(ll_sel)/d(gamma)
            g_gamma_sel = (1 / denom) * np.sum(
                lambda_arg.reshape(-1, 1) * X_sel_obs, axis=0
            )

            # d(ll_sel)/d(athrho)
            d_rho_d_athrho = 1 - rho2  # dtanh(x)/dx = 1 - tanh(x)^2
            g_athrho_sel = np.sum(lambda_arg * (
                (a * denom - rho * (X_sel_obs @ gamma + rho * a) * (-rho) / denom)
                / (denom ** 2 + 1e-15)
                * d_rho_d_athrho
                + rho * a / sigma * 0  # 简化
            ))

            # 更精确的 d(ll)/d(athrho)
            w_gamma = X_sel_obs @ gamma
            # d(arg)/d(rho) = a/d - rho*(w_gamma + rho*a)*rho/(d^3)
            # 但更简单:
            # arg = (w_gamma + rho*a) / d
            # d(arg)/d(rho) = (a*d - (w_gamma + rho*a)*(-rho)/d) / d^2
            #                = (a*d^2 + rho*(w_gamma + rho*a)) / d^3
            darg_drho = (a * denom**2 + rho * (w_gamma + rho * a)) / (denom**3 + 1e-15)
            g_athrho_sel = np.sum(lambda_arg * darg_drho * d_rho_d_athrho)

            # d(ll_sel)/d(lnsigma)
            darg_dsigma = -rho * a / (sigma * denom + 1e-15)
            g_lnsigma_sel = np.sum(
                -1 + a**2 + lambda_arg * darg_dsigma * sigma
            )
            g_lnsigma_sel += np.sum(lambda_arg * (-rho * a / (denom + 1e-15)) * (-a / sigma) * sigma)

            # 未选中观测的梯度
            xb_nobs = X_sel_nobs @ gamma
            xb_nobs_c = np.clip(xb_nobs, -20, 20)
            Phi_neg = sp_stats.norm.cdf(-xb_nobs_c)
            Phi_neg = np.clip(Phi_neg, 1e-300, 1)
            phi_neg = sp_stats.norm.pdf(-xb_nobs_c)
            lambda_neg = phi_neg / Phi_neg

            g_gamma_nsel = -np.sum(
                lambda_neg.reshape(-1, 1) * X_sel_nobs, axis=0
            )
            g_athrho_nsel = 0.0
            g_lnsigma_nsel = 0.0

            grad = np.zeros(len(params))
            grad[:k_out] = -g_beta
            grad[k_out:k_out + k_sel] = -(g_gamma_sel + g_gamma_nsel)
            grad[k_out + k_sel] = -(g_athrho_sel + g_athrho_nsel)
            grad[k_out + k_sel + 1] = -(g_lnsigma_sel + g_lnsigma_nsel)

            return grad

        # 初始化
        x0 = np.zeros(k_out + k_sel + 2)
        try:
            x0[:k_out] = np.linalg.lstsq(X_out_obs, y_obs, rcond=None)[0]
        except Exception:
            pass
        resid_init = y_obs - X_out_obs @ x0[:k_out]
        x0[k_out + k_sel + 1] = np.log(max(np.std(resid_init), 0.1))

        # 优化: 先用 L-BFGS-B 粗搜索
        bounds = [(None, None)] * (k_out + k_sel) + [(-5, 5), (-5, 5)]
        res1 = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-12})

        # 再用 Nelder-Mead 精细化
        res = minimize(neg_ll, res1.x, method='Nelder-Mead',
                       options={'maxiter': max_iter * 4, 'xatol': 1e-10,
                                'fatol': 1e-10})

        params_opt = res.x
        ll = -res.fun
        converged = res.fun < 1e14

        beta = params_opt[:k_out]
        gamma = params_opt[k_out:k_out + k_sel]
        athrho = params_opt[k_out + k_sel]
        lnsigma = params_opt[k_out + k_sel + 1]

        rho = np.tanh(athrho)
        sigma = np.exp(lnsigma)
        lambda_ = rho * sigma

        # 用数值 Hessian 计算标准误
        n_params = len(params_opt)
        h = 1e-5
        H = np.zeros((n_params, n_params))
        f0 = neg_ll(params_opt)
        for i in range(n_params):
            params_plus = params_opt.copy()
            params_plus[i] += h
            fi_plus = neg_ll(params_plus)

            params_minus = params_opt.copy()
            params_minus[i] -= h
            fi_minus = neg_ll(params_minus)

            H[i, i] = (fi_plus - 2 * f0 + fi_minus) / h**2

            for j in range(i + 1, n_params):
                params_pp = params_opt.copy()
                params_pp[i] += h
                params_pp[j] += h
                f_pp = neg_ll(params_pp)

                params_pm = params_opt.copy()
                params_pm[i] += h
                params_pm[j] -= h
                f_pm = neg_ll(params_pm)

                params_mp = params_opt.copy()
                params_mp[i] -= h
                params_mp[j] += h
                f_mp = neg_ll(params_mp)

                params_mm = params_opt.copy()
                params_mm[i] -= h
                params_mm[j] -= h
                f_mm = neg_ll(params_mm)

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
                H[j, i] = H[i, j]

        try:
            V = np.linalg.inv(H)
            if not np.all(np.diag(V) > 0):
                V = np.linalg.pinv(H)
        except np.linalg.LinAlgError:
            V = np.linalg.pinv(H)

        se_all = np.sqrt(np.maximum(np.diag(V), 0))
        z_stat = params_opt / se_all
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # 分离各部分标准误
        outcome_se = se_all[:k_out]
        select_se = se_all[k_out:k_out + k_sel]
        athrho_se = se_all[k_out + k_sel]
        lnsigma_se = se_all[k_out + k_sel + 1]

        outcome_z = z_stat[:k_out]
        select_z = z_stat[k_out:k_out + k_sel]

        outcome_p = p_value[:k_out]
        select_p = p_value[k_out:k_out + k_sel]

        # Wald chi2 检验（检验结果方程系数除常数项外是否为 0）
        df_m = k_out - 1 if is_con else k_out
        R = np.zeros((df_m, n_params))
        for i in range(df_m):
            R[i, i] = 1.0

        try:
            Rb = R @ params_opt
            RVR = R @ V @ R.T
            chi2 = float(Rb.T @ np.linalg.solve(RVR, Rb))
        except Exception:
            chi2 = 0.0
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        # LR test of independent equations (rho=0)
        # 约束模型: rho=0 即 athrho=0
        params_constrained = params_opt.copy()
        params_constrained[k_out + k_sel] = 0.0  # athrho = 0

        # 重跑约束优化
        def neg_ll_constrained(params_free):
            params_full = params_opt.copy()
            idx = 0
            for i in range(n_params):
                if i == k_out + k_sel:
                    continue
                params_full[i] = params_free[idx]
                idx += 1
            params_full[k_out + k_sel] = 0.0
            return -log_likelihood(params_full)

        x0_c = np.delete(params_opt, k_out + k_sel)
        res_c = minimize(neg_ll_constrained, x0_c, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-12})
        ll_constrained = -res_c.fun
        lr_chi2 = 2 * (ll - ll_constrained)
        lr_chi2 = max(lr_chi2, 0)
        lr_pval = 1 - sp_stats.chi2.cdf(lr_chi2, 1)

        # rho/sigma 的标准误用 delta method
        # rho = tanh(athrho), se(rho) = se(athrho) * (1 - rho^2)
        rho_se = athrho_se * (1 - rho**2)
        # sigma = exp(lnsigma), se(sigma) = se(lnsigma) * sigma
        sigma_se = lnsigma_se * sigma
        # lambda = rho * sigma
        # var(lambda) = sigma^2 * var(rho) + rho^2 * var(sigma) + 2*rho*sigma*cov(rho,sigma)
        # 近似: 从 V 矩阵获取
        idx_athrho = k_out + k_sel
        idx_lnsigma = k_out + k_sel + 1
        # delta method: lambda = tanh(a) * exp(s)
        # d(lambda)/d(a) = (1-tanh(a)^2) * exp(s) = (1-rho^2) * sigma
        # d(lambda)/d(s) = tanh(a) * exp(s) = rho * sigma
        J = np.array([(1 - rho**2) * sigma, rho * sigma])
        V_aux = V[np.ix_([idx_athrho, idx_lnsigma], [idx_athrho, idx_lnsigma])]
        lambda_se = np.sqrt(max(float(J @ V_aux @ J.T), 0))

        return HeckmanResult(
            outcome_coef=beta,
            outcome_se=outcome_se,
            outcome_z=outcome_z,
            outcome_p=outcome_p,
            select_coef=gamma,
            select_se=select_se,
            select_z=select_z,
            select_p=select_p,
            athrho=athrho,
            athrho_se=athrho_se,
            lnsigma=lnsigma,
            lnsigma_se=lnsigma_se,
            rho=rho,
            rho_se=rho_se,
            sigma=sigma,
            sigma_se=sigma_se,
            lambda_=lambda_,
            lambda_se=lambda_se,
            ll=ll,
            n_obs=n,
            n_selected=n_selected,
            n_nonselected=n_nonselected,
            chi2=chi2,
            chi2_pval=chi2_pval,
            df_m=df_m,
            outcome_var_names=outcome_var_names,
            select_var_names=select_var_names,
            y_name=y_name,
            converged=converged,
            method="mle",
            lr_chi2=lr_chi2,
            lr_pval=lr_pval,
            V=V,
        )

    def _fit_twostep(self, y_vec, X_outcome, X_select, selected,
                     n, n_selected, n_nonselected,
                     outcome_var_names, select_var_names,
                     y_name, is_con):
        k_out = X_outcome.shape[1]
        k_sel = X_select.shape[1]

        mask_sel = selected == 1
        y_obs = y_vec[mask_sel]
        X_out_obs = X_outcome[mask_sel]
        X_sel_obs = X_select[mask_sel]

        # 第一步: Probit
        from scipy.optimize import minimize as sp_minimize

        def probit_neg_ll(gamma):
            xb = X_select @ gamma
            xb = np.clip(xb, -20, 20)
            p = sp_stats.norm.cdf(xb)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.sum(selected * np.log(p) + (1 - selected) * np.log(1 - p))

        gamma_init = np.zeros(k_sel)
        res_probit = sp_minimize(probit_neg_ll, gamma_init, method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-12})
        gamma = res_probit.x

        # 计算 IMR
        xb_sel = X_sel_obs @ gamma
        xb_sel = np.clip(xb_sel, -20, 20)
        phi_sel = sp_stats.norm.pdf(xb_sel)
        Phi_sel = sp_stats.norm.cdf(xb_sel)
        Phi_sel = np.clip(Phi_sel, 1e-15, 1)
        imr = phi_sel / Phi_sel

        # 第二步: OLS (y_obs = X_out_obs * beta + theta * imr + eps)
        X_aug = np.column_stack([X_out_obs, imr])
        beta_aug = np.linalg.lstsq(X_aug, y_obs, rcond=None)[0]
        beta = beta_aug[:k_out]
        theta = beta_aug[k_out]

        resid = y_obs - X_aug @ beta_aug
        n_sel = len(y_obs)
        k_aug = X_aug.shape[1]
        sigma_resid = np.sqrt(np.sum(resid**2) / (n_sel - k_aug))

        # rho 和 sigma
        sigma_hat = np.sqrt(np.var(resid) + theta**2)
        rho_hat = theta / sigma_hat

        # 截断 rho 到 [-1, 1]
        rho_hat = np.clip(rho_hat, -1, 1)

        lambda_hat = rho_hat * sigma_hat

        # 标准误（简化版本）
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
        V_aug = sigma_resid**2 * XtX_inv
        se_aug = np.sqrt(np.maximum(np.diag(V_aug), 0))

        outcome_se = se_aug[:k_out]
        theta_se = se_aug[k_out]

        outcome_z = beta / np.maximum(outcome_se, 1e-15)
        outcome_p = 2 * (1 - sp_stats.norm.cdf(np.abs(outcome_z)))

        # 选择方程标准误 from probit Hessian
        xb_all = X_select @ gamma
        xb_all = np.clip(xb_all, -20, 20)
        p_all = sp_stats.norm.cdf(xb_all)
        p_all = np.clip(p_all, 1e-15, 1 - 1e-15)
        phi_all = sp_stats.norm.pdf(xb_all)
        lam = phi_all**2 / (p_all * (1 - p_all))
        W = np.diag(lam)
        H_probit = -X_select.T @ W @ X_select
        try:
            V_probit = np.linalg.inv(-H_probit)
        except np.linalg.LinAlgError:
            V_probit = np.linalg.pinv(-H_probit)
        select_se = np.sqrt(np.maximum(np.diag(V_probit), 0))
        select_z = gamma / np.maximum(select_se, 1e-15)
        select_p = 2 * (1 - sp_stats.norm.cdf(np.abs(select_z)))

        # Wald chi2
        df_m = k_out - 1 if is_con else k_out
        R = np.zeros((df_m, k_out))
        for i in range(df_m):
            R[i, i] = 1.0
        V_outcome = V_aug[:k_out, :k_out]
        try:
            Rb = R @ beta
            RVR = R @ V_outcome @ R.T
            chi2 = float(Rb.T @ np.linalg.solve(RVR, Rb))
        except Exception:
            chi2 = 0.0
        chi2_pval = 1 - sp_stats.chi2.cdf(chi2, df_m) if df_m > 0 else 1.0

        return HeckmanResult(
            outcome_coef=beta,
            outcome_se=outcome_se,
            outcome_z=outcome_z,
            outcome_p=outcome_p,
            select_coef=gamma,
            select_se=select_se,
            select_z=select_z,
            select_p=select_p,
            athrho=np.arctanh(rho_hat),
            athrho_se=0.0,
            lnsigma=np.log(sigma_hat),
            lnsigma_se=0.0,
            rho=rho_hat,
            rho_se=0.0,
            sigma=sigma_hat,
            sigma_se=0.0,
            lambda_=lambda_hat,
            lambda_se=0.0,
            ll=0.0,
            n_obs=n,
            n_selected=n_selected,
            n_nonselected=n_nonselected,
            chi2=chi2,
            chi2_pval=chi2_pval,
            df_m=df_m,
            outcome_var_names=outcome_var_names,
            select_var_names=select_var_names,
            y_name=y_name,
            converged=res_probit.fun < 1e14,
            method="twostep",
            lr_chi2=0.0,
            lr_pval=1.0,
            V=None,
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for Heckman estimation")
