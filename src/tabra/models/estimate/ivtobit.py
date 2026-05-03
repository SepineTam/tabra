#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ivtobit.py

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from tabra.models.estimate.base import BaseModel
from tabra.results.ivtobit_result import IVTobitResult


class IVTobitModel(BaseModel):
    """IV tobit regression (MLE and twostep)."""

    def fit(self, df, y, exog, endog, instruments,
            ll=None, ul=None, method="mle", vce="unadjusted", is_con=True):
        if method not in ("mle", "twostep"):
            raise ValueError(f"Unknown method '{method}', use 'mle' or 'twostep'")
        if vce not in ("unadjusted", "robust"):
            raise ValueError(f"Unknown vce '{vce}', use 'unadjusted' or 'robust'")

        exog = list(exog)
        endog = list(endog)
        instruments = list(instruments)

        if len(instruments) < len(endog):
            raise ValueError(
                f"Underidentified: {len(endog)} endogenous but "
                f"only {len(instruments)} instruments."
            )

        all_cols = [y] + exog + endog + instruments
        df = df[all_cols].dropna()

        y_vec = df[y].values.astype(float)
        X1 = df[exog].values.astype(float) if exog else np.empty((len(df), 0))
        X2 = df[endog].values.astype(float)
        Z = df[instruments].values.astype(float)

        n = len(y_vec)
        n_exog = X1.shape[1]
        n_endog = X2.shape[1]

        if is_con:
            const = np.ones((n, 1))
            # Stata order: [exog, instruments, const]
            Z_full = np.column_stack([X1, Z, const]) if n_exog > 0 \
                else np.column_stack([Z, const])
        else:
            Z_full = np.column_stack([X1, Z]) if n_exog > 0 else Z.copy()

        # Censoring masks
        left_censored = y_vec <= ll if ll is not None else np.zeros(n, dtype=bool)
        right_censored = y_vec >= ul if ul is not None else np.zeros(n, dtype=bool)
        uncensored = ~left_censored & ~right_censored
        n_lc = int(np.sum(left_censored))
        n_rc = int(np.sum(right_censored))
        n_unc = int(np.sum(uncensored))

        if method == "mle":
            return self._fit_mle(
                y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                is_con, exog, endog, y, vce,
                ll, ul, left_censored, right_censored, uncensored,
                n_lc, n_rc, n_unc
            )
        else:
            return self._fit_twostep(
                y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                is_con, exog, endog, y, vce,
                ll, ul, left_censored, right_censored, uncensored,
                n_lc, n_rc, n_unc
            )

    def _fit_mle(self, y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                 is_con, exog_names, endog_names, y_name, vce,
                 ll, ul, left_censored, right_censored, uncensored,
                 n_lc, n_rc, n_unc):
        """MLE for IV tobit."""
        k_z = Z_full.shape[1]
        n_b1 = n_exog
        n_b2 = n_endog
        n_pi = k_z * n_endog
        n_theta = n_endog
        n_b0 = 1 if is_con else 0  # explicit constant in structure eq
        # Params: [b1, b2, b0, pi, theta, ln_sigma_uv, ln_sigma_v]
        n_params = n_b1 + n_b2 + n_b0 + n_pi + n_theta + 2

        # Init from reduced-form OLS + tobit on augmented equation
        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        pi_init = np.zeros(n_pi)
        v_hat = np.zeros((n, n_endog))
        for j in range(n_endog):
            pi_j = ZtZ_inv @ Z_full.T @ X2[:, j]
            pi_init[j * k_z:(j + 1) * k_z] = pi_j
            v_hat[:, j] = X2[:, j] - Z_full @ pi_j

        X_aug_init = np.column_stack([X1, X2, v_hat]) if n_exog > 0 \
            else np.column_stack([X2, v_hat])
        if is_con:
            X_aug_init = np.column_stack([X_aug_init, np.ones(n)])

        beta_init = np.linalg.lstsq(X_aug_init, y_vec, rcond=None)[0]
        resid = y_vec - X_aug_init @ beta_init
        sigma_init = max(np.std(resid, ddof=X_aug_init.shape[1]), 0.1)

        theta0 = np.zeros(n_params)
        # Layout: [b1, b2, b0, pi, theta, ln_sigma_uv, ln_sigma_v]
        # beta_init layout: [X1_cols, X2_cols, vhat_cols, const]
        theta0[:n_b1] = beta_init[:n_b1]
        theta0[n_b1:n_b1 + n_b2] = beta_init[n_b1:n_b1 + n_b2]
        if is_con:
            theta0[n_b1 + n_b2] = beta_init[-1]  # b0 = OLS intercept
        theta0[n_b1 + n_b2 + n_b0:n_b1 + n_b2 + n_b0 + n_pi] = pi_init
        theta0[-2] = np.log(sigma_init)
        v_sigma = np.std(v_hat[:, 0], ddof=k_z) if n_endog >= 1 else 1.0
        theta0[-1] = np.log(max(v_sigma, 0.1))

        def neg_log_lik(theta):
            b1 = theta[:n_b1]
            b2 = theta[n_b1:n_b1 + n_b2]
            b0 = theta[n_b1 + n_b2] if is_con else 0.0
            offset_pi = n_b1 + n_b2 + n_b0
            pi_vec = theta[offset_pi:offset_pi + n_pi]
            offset_th = offset_pi + n_pi
            theta_vec = theta[offset_th:offset_th + n_theta]
            ln_sigma_uv = theta[-2]
            ln_sigma_v = theta[-1]

            sigma_uv = np.exp(ln_sigma_uv)
            sigma_v = np.exp(ln_sigma_v)

            v = np.zeros((n, n_endog))
            for j in range(n_endog):
                pi_j = pi_vec[j * k_z:(j + 1) * k_z]
                v[:, j] = X2[:, j] - Z_full @ pi_j

            m = (X1 @ b1 if n_exog > 0 else 0.0) + X2 @ b2 + b0 + v @ theta_vec

            # LL contributions
            ll_val = 0.0

            # Left-censored: P(y* <= ll)
            if n_lc > 0:
                z_lc = (ll - m[left_censored]) / sigma_uv
                z_lc = np.clip(z_lc, -30, 30)
                ll_val += np.sum(np.log(sp_stats.norm.cdf(z_lc) + 1e-300))

            # Right-censored: P(y* >= ul)
            if n_rc > 0:
                z_rc = (ul - m[right_censored]) / sigma_uv
                z_rc = np.clip(z_rc, -30, 30)
                ll_val += np.sum(np.log(1 - sp_stats.norm.cdf(z_rc) + 1e-300))

            # Uncensored: normal density
            if n_unc > 0:
                resid_u = y_vec[uncensored] - m[uncensored]
                ll_val += np.sum(
                    -0.5 * np.log(2 * np.pi) - np.log(sigma_uv)
                    - 0.5 * (resid_u / sigma_uv) ** 2
                )

            # Reduced-form LL
            ll_val += -n * n_endog * np.log(sigma_v) \
                - 0.5 * n_endog * n * np.log(2 * np.pi) \
                - 0.5 * np.sum(v ** 2) / sigma_v ** 2

            return -ll_val if np.isfinite(ll_val) else 1e20

        res = minimize(neg_log_lik, theta0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        theta_hat = res.x
        converged = res.success

        b1 = theta_hat[:n_b1]
        b2 = theta_hat[n_b1:n_b1 + n_b2]
        b0 = theta_hat[n_b1 + n_b2] if is_con else 0.0
        offset_th = n_b1 + n_b2 + n_b0 + n_pi
        theta_vec = theta_hat[offset_th:offset_th + n_theta]
        sigma_uv = np.exp(theta_hat[-2])

        # Display: [endog, exog, _cons]
        if is_con:
            var_names = endog_names + exog_names + ["_cons"]
            coef = np.concatenate([b2, b1, [b0]])
        else:
            var_names = endog_names + exog_names
            coef = np.concatenate([b2, b1])

        k = len(coef)

        # VCE
        V_full = self._numerical_hessian_inv(neg_log_lik, theta_hat, eps=1e-4)

        if V_full is not None:
            if vce == "robust":
                B = self._robust_vce_mle(
                    y_vec, X1, X2, Z_full, theta_hat, n, n_exog, n_endog,
                    n_b0, n_pi, n_theta, k_z, ll, ul,
                    left_censored, right_censored, uncensored, eps=1e-6
                )
                if B is not None:
                    V_full = V_full @ B @ V_full

            # Map: display order [b2, b1, b0] → theta indices [b1, b2, b0, ...]
            idx_map = list(range(n_b1, n_b1 + n_b2)) + list(range(n_b1))
            if is_con:
                idx_map.append(n_b1 + n_b2)  # b0 index in theta
            V_coef = V_full[np.ix_(idx_map, idx_map)]

            std_err = np.sqrt(np.maximum(np.diag(V_coef), 0))

            # Endogeneity test: Wald H0: rho = 0 (atanh(rho) = 0)
            # rho = theta * sigma_v / sqrt(sigma_uv^2 + theta^2 * sigma_v^2)
            # athrho = atanh(rho)
            # Use delta method to compute var(athrho) from V_full
            sigma_v = np.exp(theta_hat[-1])
            sigma_u = np.sqrt(sigma_uv**2 + theta_vec[0]**2 * sigma_v**2)
            if sigma_u > 0 and abs(theta_vec[0] * sigma_v / sigma_u) < 1:
                theta_val = float(theta_vec[0])
                s_uv = float(sigma_uv)
                s_v = float(sigma_v)
                s_u = float(sigma_u)
                rho = theta_val * s_v / s_u
                athrho = 0.5 * np.log((1 + rho) / (1 - rho))

                # Analytical gradient of athrho w.r.t. [theta, ln_sigma_uv, ln_sigma_v]
                # rho = theta * sigma_v / sqrt(sigma_uv^2 + theta^2 * sigma_v^2)
                # d(rho)/d(theta) = sigma_v * sigma_uv^2 / sigma_u^3
                # d(rho)/d(sigma_uv) = -theta^2 * sigma_v / sigma_u^3
                # d(rho)/d(sigma_v) = theta * sigma_uv^2 / sigma_u^3
                # d(athrho)/d(rho) = 1 / (1 - rho^2)
                drho_dtheta = s_v * s_uv**2 / s_u**3
                drho_dsigma_uv = -theta_val * s_v * s_uv / s_u**3
                drho_dsigma_v = theta_val * s_uv**2 / s_u**3
                dathrho_drho = 1.0 / (1.0 - rho**2)

                # Chain rule for ln-transformed parameters
                dathrho_dln_sigma_uv = dathrho_drho * drho_dsigma_uv * s_uv
                dathrho_dln_sigma_v = dathrho_drho * drho_dsigma_v * s_v
                dathrho_dtheta_val = dathrho_drho * drho_dtheta

                p = len(theta_hat)
                grad_athrho = np.zeros(p)
                grad_athrho[offset_th] = dathrho_dtheta_val
                grad_athrho[-2] = dathrho_dln_sigma_uv
                grad_athrho[-1] = dathrho_dln_sigma_v

                var_athrho = float(grad_athrho @ V_full @ grad_athrho)
                if var_athrho > 0:
                    endog_test_stat = float(athrho**2 / var_athrho)
                    endog_test_pval = float(1 - sp_stats.chi2.cdf(endog_test_stat, 1))
                else:
                    endog_test_stat = None
                    endog_test_pval = None
            else:
                endog_test_stat = None
                endog_test_pval = None
        else:
            V_coef = None
            std_err = np.ones(k) * np.nan
            endog_test_stat = None
            endog_test_pval = None

        z_stat = coef / np.where(std_err > 0, std_err, np.nan)
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        ll_val = -neg_log_lik(theta_hat)

        mask = [i for i, nm in enumerate(var_names) if nm != "_cons"]
        if mask and V_coef is not None:
            chi2 = float(coef[mask] @ np.linalg.solve(V_coef[np.ix_(mask, mask)], coef[mask]))
            chi2_pval = float(1 - sp_stats.chi2.cdf(chi2, len(mask)))
        else:
            chi2 = 0.0
            chi2_pval = 1.0

        # sigma from the conditional variance
        sigma = sigma_uv

        return IVTobitResult(
            coef=coef, std_err=std_err, z_stat=z_stat, p_value=p_value,
            n_obs=n, n_lc=n_lc, n_rc=n_rc, n_unc=n_unc,
            ll=ll_val, chi2=chi2, chi2_pval=chi2_pval,
            sigma=sigma, sigma_se=np.nan,
            endog_test_stat=endog_test_stat,
            endog_test_pval=endog_test_pval,
            var_names=var_names, y_name=y_name,
            method="mle", converged=converged,
            vce_type=vce, ll_limit=ll, ul_limit=ul,
        )

    def _fit_twostep(self, y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                     is_con, exog_names, endog_names, y_name, vce,
                     ll, ul, left_censored, right_censored, uncensored,
                     n_lc, n_rc, n_unc):
        """Newey two-step IV tobit."""
        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        v_hat = np.zeros((n, n_endog))
        for j in range(n_endog):
            pi_j = ZtZ_inv @ Z_full.T @ X2[:, j]
            v_hat[:, j] = X2[:, j] - Z_full @ pi_j

        # Step 2: Tobit y on [X1, X2, v_hat]
        X_aug = np.column_stack([X1, X2, v_hat]) if n_exog > 0 \
            else np.column_stack([X2, v_hat])
        if is_con:
            X_aug = np.column_stack([X_aug, np.ones(n)])

        k = X_aug.shape[1]

        # Simple tobit MLE on augmented X
        beta_init = np.linalg.lstsq(X_aug, y_vec, rcond=None)[0]
        resid = y_vec - X_aug @ beta_init
        sigma_init = max(np.std(resid, ddof=k), 0.1)
        params0 = np.append(beta_init, np.log(sigma_init))

        # Store pi_init for Newey correction
        pi_init_store = np.zeros(n_endog * Z_full.shape[1])
        for j in range(n_endog):
            pi_j = ZtZ_inv @ Z_full.T @ X2[:, j]
            pi_init_store[j * Z_full.shape[1]:(j + 1) * Z_full.shape[1]] = pi_j

        def tobit_neg_ll(params):
            beta = params[:k]
            sigma = np.exp(params[k])
            xb = X_aug @ beta
            ll_val = 0.0
            if n_lc > 0:
                z = (ll - xb[left_censored]) / sigma
                z = np.clip(z, -30, 30)
                ll_val += np.sum(np.log(sp_stats.norm.cdf(z) + 1e-300))
            if n_rc > 0:
                z = (ul - xb[right_censored]) / sigma
                z = np.clip(z, -30, 30)
                ll_val += np.sum(np.log(1 - sp_stats.norm.cdf(z) + 1e-300))
            if n_unc > 0:
                r = y_vec[uncensored] - xb[uncensored]
                ll_val += np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (r / sigma) ** 2)
            return -ll_val if np.isfinite(ll_val) else 1e20

        res = minimize(tobit_neg_ll, params0, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12})
        params = res.x
        beta = params[:k]
        sigma = np.exp(params[k])

        vhat_names = [f"vhat_{e}" for e in endog_names]
        # X_aug = [X1, X2, v_hat, const] => beta = [exog, endog, vhat, const]
        # Display order: [endog, exog, vhat, const]
        var_names = endog_names + exog_names + vhat_names + (["_cons"] if is_con else [])

        # Reorder beta: [exog, endog, vhat, const] -> [endog, exog, vhat, const]
        idx_map = []
        for j in range(n_endog):
            idx_map.append(n_exog + j)  # endog
        for j in range(n_exog):
            idx_map.append(j)  # exog
        for j in range(n_endog):
            idx_map.append(n_exog + n_endog + j)  # vhat
        if is_con:
            idx_map.append(n_exog + n_endog + n_endog)  # const

        coef = beta[idx_map]

        # VCE
        V = self._numerical_hessian_inv(tobit_neg_ll, params, eps=1e-4)

        # Store uncorrected V_2 for endogeneity test BEFORE Newey and robust correction
        V_2 = V.copy() if V is not None else None

        # Newey (1987) two-step correction for unadjusted VCE
        # V = V_2 + V_2 @ C @ V_1 @ C.T @ V_2
        if V is not None:
            k_z = Z_full.shape[1]
            n_pi = k_z * n_endog
            # Step 1 VCE: OLS of X2 on Z_full
            sigma_v_sq = np.sum(v_hat ** 2) / n
            V_1 = sigma_v_sq * ZtZ_inv  # VCE for pi_hat

            # Compute C = sum_i [d^2 LL_2i / d(beta) d(pi)]
            # Using numerical cross-derivatives
            # Tobit params: [beta, ln_sigma], pi params: [pi]
            p_tobit = len(params)
            C = np.zeros((p_tobit, n_pi))
            eps_t = 1e-5
            eps_vec_t = np.maximum(np.abs(params) * eps_t, 1e-10)
            eps_vec_p = np.maximum(np.abs(pi_init_store) * eps_t, 1e-10)

            for i in range(p_tobit):
                for j in range(n_pi):
                    ei = np.zeros(p_tobit)
                    ei[i] = eps_vec_t[i]
                    ej_pi = np.zeros(n_pi)
                    ej_pi[j] = eps_vec_p[j]

                    # LL at (params + ei, pi + ej)
                    ll_pp = self._tobit_ll_at_pi(
                        params + ei, pi_init_store + ej_pi, y_vec, X1, X2, Z_full,
                        n, n_exog, n_endog, is_con, ll, ul,
                        left_censored, right_censored, uncensored
                    )
                    # LL at (params + ei, pi - ej)
                    ll_pm = self._tobit_ll_at_pi(
                        params + ei, pi_init_store - ej_pi, y_vec, X1, X2, Z_full,
                        n, n_exog, n_endog, is_con, ll, ul,
                        left_censored, right_censored, uncensored
                    )
                    # LL at (params - ei, pi + ej)
                    ll_mp = self._tobit_ll_at_pi(
                        params - ei, pi_init_store + ej_pi, y_vec, X1, X2, Z_full,
                        n, n_exog, n_endog, is_con, ll, ul,
                        left_censored, right_censored, uncensored
                    )
                    # LL at (params - ei, pi - ej)
                    ll_mm = self._tobit_ll_at_pi(
                        params - ei, pi_init_store - ej_pi, y_vec, X1, X2, Z_full,
                        n, n_exog, n_endog, is_con, ll, ul,
                        left_censored, right_censored, uncensored
                    )

                    if all(np.isfinite(x) for x in [ll_pp, ll_pm, ll_mp, ll_mm]):
                        C[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * eps_vec_t[i] * eps_vec_p[j])

            # Newey correction
            V_corr = V @ C @ V_1 @ C.T @ V
            V = V + V_corr

        if V is not None and vce == "robust":
            B = self._robust_vce_twostep(
                y_vec, X1, X2, v_hat, Z_full, params, n, n_exog, n_endog,
                is_con, ll, ul, left_censored, right_censored, uncensored, eps=1e-5
            )
            if B is not None:
                V = V @ B @ V

        if V is not None:
            V_beta = V[:k, :k]
            V_coef = V_beta[np.ix_(idx_map, idx_map)]
            std_err = np.sqrt(np.maximum(np.diag(V_coef), 0))
        else:
            std_err = np.ones(k) * np.nan
            V_coef = None

        z_stat = coef / np.where(std_err > 0, std_err, np.nan)
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        ll_val = -tobit_neg_ll(params)

        mask = [i for i, nm in enumerate(var_names)
                if nm != "_cons" and not nm.startswith("vhat_")]
        if mask and V_coef is not None:
            chi2 = float(coef[mask] @ np.linalg.solve(V_coef[np.ix_(mask, mask)], coef[mask]))
            chi2_pval = float(1 - sp_stats.chi2.cdf(chi2, len(mask)))
        else:
            chi2 = 0.0
            chi2_pval = 1.0

        vhat_idx = [i for i, nm in enumerate(var_names) if nm.startswith("vhat_")]
        if vhat_idx and V_coef is not None:
            # Stata's twostep endogeneity test uses the uncorrected V_2 for vhat
            V_2_beta = V_2[:k, :k] if V_2 is not None else None
            if V_2_beta is not None:
                V_2_coef = V_2_beta[np.ix_(idx_map, idx_map)]
                endog_test_stat = float(
                    coef[vhat_idx] @ np.linalg.solve(V_2_coef[np.ix_(vhat_idx, vhat_idx)], coef[vhat_idx])
                )
                endog_test_pval = float(1 - sp_stats.chi2.cdf(endog_test_stat, len(vhat_idx)))
            else:
                endog_test_stat = None
                endog_test_pval = None
        else:
            endog_test_stat = None
            endog_test_pval = None

        return IVTobitResult(
            coef=coef, std_err=std_err, z_stat=z_stat, p_value=p_value,
            n_obs=n, n_lc=n_lc, n_rc=n_rc, n_unc=n_unc,
            ll=ll_val, chi2=chi2, chi2_pval=chi2_pval,
            sigma=sigma, sigma_se=np.nan,
            endog_test_stat=endog_test_stat,
            endog_test_pval=endog_test_pval,
            var_names=var_names, y_name=y_name,
            method="twostep", converged=True,
            vce_type=vce, ll_limit=ll, ul_limit=ul,
        )

    @staticmethod
    def _numerical_hessian_inv(neg_ll_fn, theta, eps=1e-5):
        p = len(theta)
        eps_vec = np.maximum(np.abs(theta) * eps, 1e-10)
        hess = np.zeros((p, p))
        f0 = neg_ll_fn(theta)
        if not np.isfinite(f0):
            return None
        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps_vec[i]
            fp = neg_ll_fn(theta + ei)
            fm = neg_ll_fn(theta - ei)
            if not (np.isfinite(fp) and np.isfinite(fm)):
                return None
            hess[i, i] = (fp - 2 * f0 + fm) / eps_vec[i] ** 2
            for j in range(i + 1, p):
                ej = np.zeros(p)
                ej[j] = eps_vec[j]
                fpp = neg_ll_fn(theta + ei + ej)
                fpm = neg_ll_fn(theta + ei - ej)
                fmp = neg_ll_fn(theta - ei + ej)
                fmm = neg_ll_fn(theta - ei - ej)
                if all(np.isfinite(x) for x in [fpp, fpm, fmp, fmm]):
                    hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps_vec[i] * eps_vec[j])
                    hess[j, i] = hess[i, j]
        # Symmetrize and regularize before inversion
        hess = 0.5 * (hess + hess.T)
        try:
            return np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            # Add small ridge for numerical stability
            eigvals = np.linalg.eigvalsh(hess)
            min_eig = np.min(eigvals)
            if min_eig <= 0:
                ridge = max(-min_eig + 1e-10, 1e-10)
                hess_reg = hess + ridge * np.eye(p)
                try:
                    return np.linalg.inv(hess_reg)
                except np.linalg.LinAlgError:
                    return np.linalg.pinv(hess_reg)
            return np.linalg.pinv(hess) if np.any(hess != 0) else None

    @staticmethod
    def _outer_product_of_scores(neg_ll_fn, theta, eps=1e-5):
        p = len(theta)
        eps_vec = np.maximum(np.abs(theta) * eps, 1e-10)
        grad = np.zeros(p)
        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps_vec[i]
            fp = neg_ll_fn(theta + ei)
            fm = neg_ll_fn(theta - ei)
            if np.isfinite(fp) and np.isfinite(fm):
                grad[i] = (fp - fm) / (2 * eps_vec[i])
            else:
                return None
        return np.outer(grad, grad)

    @staticmethod
    def _robust_vce_mle(y_vec, X1, X2, Z_full, theta_hat, n, n_exog, n_endog,
                        n_b0, n_pi, n_theta, k_z, ll, ul,
                        left_censored, right_censored, uncensored, eps=1e-5):
        """Compute robust VCE meat for MLE via per-observation numerical scores."""
        p = len(theta_hat)

        def _obs_ll(i, theta):
            b1 = theta[:n_exog]
            b2 = theta[n_exog:n_exog + n_endog]
            b0 = theta[n_exog + n_endog] if n_b0 > 0 else 0.0
            offset_pi = n_exog + n_endog + n_b0
            pi_vec = theta[offset_pi:offset_pi + n_pi]
            offset_th = offset_pi + n_pi
            theta_vec = theta[offset_th:offset_th + n_theta]
            ln_sigma_uv = theta[-2]
            ln_sigma_v = theta[-1]

            sigma_uv = np.exp(ln_sigma_uv)
            sigma_v = np.exp(ln_sigma_v)

            v_i = np.zeros(n_endog)
            for j in range(n_endog):
                pi_j = pi_vec[j * k_z:(j + 1) * k_z]
                v_i[j] = X2[i, j] - Z_full[i] @ pi_j

            m_i = (X1[i] @ b1 if n_exog > 0 else 0.0) + X2[i] @ b2 + b0 + v_i @ theta_vec

            ll_i = 0.0
            if left_censored[i]:
                z = (ll - m_i) / sigma_uv
                z = np.clip(z, -30, 30)
                ll_i += np.log(sp_stats.norm.cdf(z) + 1e-300)
            elif right_censored[i]:
                z = (ul - m_i) / sigma_uv
                z = np.clip(z, -30, 30)
                ll_i += np.log(1 - sp_stats.norm.cdf(z) + 1e-300)
            else:
                r = y_vec[i] - m_i
                ll_i += -0.5 * np.log(2 * np.pi) - np.log(sigma_uv) - 0.5 * (r / sigma_uv) ** 2

            ll_i += -n_endog * np.log(sigma_v) - 0.5 * n_endog * np.log(2 * np.pi) - 0.5 * np.sum(v_i ** 2) / sigma_v ** 2
            return ll_i

        B = np.zeros((p, p))
        eps_vec = np.maximum(np.abs(theta_hat) * eps, 1e-10)
        for i in range(n):
            grad_i = np.zeros(p)
            for j in range(p):
                ej = np.zeros(p)
                ej[j] = eps_vec[j]
                fp = _obs_ll(i, theta_hat + ej)
                fm = _obs_ll(i, theta_hat - ej)
                if np.isfinite(fp) and np.isfinite(fm):
                    grad_i[j] = (fp - fm) / (2 * eps_vec[j])
            B += np.outer(grad_i, grad_i)
        return B

    @staticmethod
    def _robust_vce_twostep(y_vec, X1, X2, v_hat, Z_full, params, n, n_exog, n_endog,
                            is_con, ll, ul, left_censored, right_censored, uncensored, eps=1e-5):
        """Compute robust VCE meat for twostep via per-observation numerical scores."""
        k_aug = len(params) - 1  # beta params, last is ln_sigma

        def _obs_ll_tobit(i, beta_sigma):
            beta = beta_sigma[:k_aug]
            sigma = np.exp(beta_sigma[k_aug])
            xb_i = 0.0
            idx = 0
            if n_exog > 0:
                xb_i += X1[i] @ beta[idx:idx + n_exog]
                idx += n_exog
            xb_i += X2[i] @ beta[idx:idx + n_endog]
            idx += n_endog
            xb_i += v_hat[i] @ beta[idx:idx + n_endog]
            idx += n_endog
            if is_con:
                xb_i += beta[idx]

            ll_i = 0.0
            if left_censored[i]:
                z = (ll - xb_i) / sigma
                z = np.clip(z, -30, 30)
                ll_i += np.log(sp_stats.norm.cdf(z) + 1e-300)
            elif right_censored[i]:
                z = (ul - xb_i) / sigma
                z = np.clip(z, -30, 30)
                ll_i += np.log(1 - sp_stats.norm.cdf(z) + 1e-300)
            else:
                r = y_vec[i] - xb_i
                ll_i += -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (r / sigma) ** 2
            return ll_i

        p = len(params)
        B = np.zeros((p, p))
        eps_vec = np.maximum(np.abs(params) * eps, 1e-10)
        for i in range(n):
            grad_i = np.zeros(p)
            for j in range(p):
                ej = np.zeros(p)
                ej[j] = eps_vec[j]
                fp = _obs_ll_tobit(i, params + ej)
                fm = _obs_ll_tobit(i, params - ej)
                if np.isfinite(fp) and np.isfinite(fm):
                    grad_i[j] = (fp - fm) / (2 * eps_vec[j])
            B += np.outer(grad_i, grad_i)
        return B

    @staticmethod
    def _tobit_ll_at_pi(params, pi_vec, y_vec, X1, X2, Z_full,
                        n, n_exog, n_endog, is_con, ll, ul,
                        left_censored, right_censored, uncensored):
        """Compute Tobit log-likelihood at given params and pi."""
        k_z = Z_full.shape[1]
        beta = params[:-1]
        sigma = np.exp(params[-1])

        # Reconstruct X_aug from pi
        v = np.zeros((n, n_endog))
        for j in range(n_endog):
            pi_j = pi_vec[j * k_z:(j + 1) * k_z]
            v[:, j] = X2[:, j] - Z_full @ pi_j

        X_aug = np.column_stack([X1, X2, v]) if n_exog > 0 \
            else np.column_stack([X2, v])
        if is_con:
            X_aug = np.column_stack([X_aug, np.ones(n)])

        xb = X_aug @ beta
        ll_val = 0.0
        if np.any(left_censored):
            z = (ll - xb[left_censored]) / sigma
            z = np.clip(z, -30, 30)
            ll_val += np.sum(np.log(sp_stats.norm.cdf(z) + 1e-300))
        if np.any(right_censored):
            z = (ul - xb[right_censored]) / sigma
            z = np.clip(z, -30, 30)
            ll_val += np.sum(np.log(1 - sp_stats.norm.cdf(z) + 1e-300))
        if np.any(uncensored):
            r = y_vec[uncensored] - xb[uncensored]
            ll_val += np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * (r / sigma) ** 2)
        return ll_val

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for IV tobit estimation")
