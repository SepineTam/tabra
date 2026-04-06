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
            Z_full = np.column_stack([const, X1, Z]) if n_exog > 0 \
                else np.column_stack([const, Z])
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
        theta0[-1] = 0.0

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
                       options={'maxiter': 500, 'ftol': 1e-12})
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
        V_full = self._numerical_hessian_inv(neg_log_lik, theta_hat)

        if V_full is not None:
            if vce == "robust":
                B = self._outer_product_of_scores(neg_log_lik, theta_hat)
                if B is not None:
                    V_full = V_full @ B @ V_full

            # Map: display order [b2, b1, b0] → theta indices [b1, b2, b0, ...]
            idx_map = list(range(n_b1, n_b1 + n_b2)) + list(range(n_b1))
            if is_con:
                idx_map.append(n_b1 + n_b2)  # b0 index in theta
            V_coef = V_full[np.ix_(idx_map, idx_map)]

            std_err = np.sqrt(np.maximum(np.diag(V_coef), 0))

            # Endogeneity test: Wald H0: theta = 0
            theta_idx = list(range(offset_th, offset_th + n_theta))
            if theta_idx:
                endog_test_stat = float(
                    theta_vec @ np.linalg.solve(
                        V_full[np.ix_(theta_idx, theta_idx)], theta_vec
                    )
                )
                endog_test_pval = float(1 - sp_stats.chi2.cdf(endog_test_stat, n_theta))
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
        var_names = endog_names + exog_names + vhat_names + (["_cons"] if is_con else [])

        # VCE
        V = self._numerical_hessian_inv(tobit_neg_ll, params)
        if V is not None and vce == "robust":
            B = self._outer_product_of_scores(tobit_neg_ll, params)
            if B is not None:
                V = V @ B @ V

        if V is not None:
            std_err = np.sqrt(np.maximum(np.diag(V[:k, :k]), 0))
        else:
            std_err = np.ones(k) * np.nan

        z_stat = beta / np.where(std_err > 0, std_err, np.nan)
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        ll_val = -tobit_neg_ll(params)

        mask = [i for i, nm in enumerate(var_names)
                if nm != "_cons" and not nm.startswith("vhat_")]
        if mask and V is not None:
            chi2 = float(beta[mask] @ np.linalg.solve(V[np.ix_(mask, mask)], beta[mask]))
            chi2_pval = float(1 - sp_stats.chi2.cdf(chi2, len(mask)))
        else:
            chi2 = 0.0
            chi2_pval = 1.0

        vhat_idx = [i for i, nm in enumerate(var_names) if nm.startswith("vhat_")]
        if vhat_idx and V is not None:
            endog_test_stat = float(
                beta[vhat_idx] @ np.linalg.solve(V[np.ix_(vhat_idx, vhat_idx)], beta[vhat_idx])
            )
            endog_test_pval = float(1 - sp_stats.chi2.cdf(endog_test_stat, len(vhat_idx)))
        else:
            endog_test_stat = None
            endog_test_pval = None

        return IVTobitResult(
            coef=beta, std_err=std_err, z_stat=z_stat, p_value=p_value,
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
        hess = np.zeros((p, p))
        f0 = neg_ll_fn(theta)
        if not np.isfinite(f0):
            return None
        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps
            fp = neg_ll_fn(theta + ei)
            fm = neg_ll_fn(theta - ei)
            if not (np.isfinite(fp) and np.isfinite(fm)):
                return None
            hess[i, i] = (fp - 2 * f0 + fm) / eps ** 2
            for j in range(i + 1, p):
                ej = np.zeros(p)
                ej[j] = eps
                fpp = neg_ll_fn(theta + ei + ej)
                fpm = neg_ll_fn(theta + ei - ej)
                fmp = neg_ll_fn(theta - ei + ej)
                fmm = neg_ll_fn(theta - ei - ej)
                if all(np.isfinite(x) for x in [fpp, fpm, fmp, fmm]):
                    hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)
                    hess[j, i] = hess[i, j]
        try:
            return np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(hess) if np.any(hess != 0) else None

    @staticmethod
    def _outer_product_of_scores(neg_ll_fn, theta, eps=1e-5):
        p = len(theta)
        grad = np.zeros(p)
        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps
            fp = neg_ll_fn(theta + ei)
            fm = neg_ll_fn(theta - ei)
            if np.isfinite(fp) and np.isfinite(fm):
                grad[i] = (fp - fm) / (2 * eps)
            else:
                return None
        return np.outer(grad, grad)

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for IV tobit estimation")
