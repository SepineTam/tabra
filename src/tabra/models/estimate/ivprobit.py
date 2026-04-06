#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ivprobit.py

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize
from scipy.special import ndtr

from tabra.models.estimate.base import BaseModel
from tabra.results.ivprobit_result import IVProbitResult


class IVProbitModel(BaseModel):
    """IV probit regression (MLE and twostep)."""

    def fit(self, df, y, exog, endog, instruments,
            method="mle", vce="unadjusted", is_con=True):
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

        if method == "mle":
            return self._fit_mle(
                y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                is_con, exog, endog, y, vce
            )
        else:
            return self._fit_twostep(
                y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                is_con, exog, endog, y, vce
            )

    def _fit_mle(self, y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                 is_con, exog_names, endog_names, y_name, vce):
        """Full-information MLE for IV probit."""
        k_z = Z_full.shape[1]
        n_b1 = n_exog
        n_b2 = n_endog
        n_pi = k_z * n_endog
        has_c = 1 if is_con else 0
        n_params = has_c + n_b1 + n_b2 + n_pi + 2  # b0?, b1, b2, pi, ath_rho, ln_sv

        # Parameter offsets
        off_b0 = 0                              # only valid when is_con
        off_b1 = has_c
        off_b2 = off_b1 + n_b1
        off_pi = off_b2 + n_b2
        off_ath_rho = off_pi + n_pi
        off_ln_sv = off_ath_rho + 1

        # Init from probit + reduced-form OLS
        X_all = np.column_stack([X1, X2]) if n_exog > 0 else X2.copy()
        if is_con:
            X_all = np.column_stack([X_all, np.ones(n)])
        beta_init = self._simple_probit(y_vec, X_all)

        pi_init = np.zeros(n_pi)
        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        for j in range(n_endog):
            pi_init[j * k_z:(j + 1) * k_z] = ZtZ_inv @ Z_full.T @ X2[:, j]

        theta0 = np.zeros(n_params)
        if is_con:
            theta0[off_b0] = beta_init[n_b1 + n_b2]   # probit intercept
        theta0[off_b1:off_b1 + n_b1] = beta_init[:n_b1]
        theta0[off_b2:off_b2 + n_b2] = beta_init[n_b1:n_b1 + n_b2]
        theta0[off_pi:off_pi + n_pi] = pi_init
        sigma_v_init = np.std(X2[:, 0] - Z_full @ pi_init)
        theta0[off_ln_sv] = np.log(max(sigma_v_init, 1e-6))
        # ath_rho = 0

        def neg_log_lik(theta):
            b0 = theta[off_b0] if is_con else 0.0
            b1 = theta[off_b1:off_b1 + n_b1]
            b2 = theta[off_b2:off_b2 + n_b2]
            pi_vec = theta[off_pi:off_pi + n_pi]
            ath_rho = theta[off_ath_rho]
            ln_sv = theta[off_ln_sv]

            rho = np.tanh(ath_rho)
            sigma_v = np.exp(ln_sv)

            v = X2[:, 0] - Z_full @ pi_vec

            mu = b0 + (X1 @ b1 if n_exog > 0 else 0.0) + X2 @ b2 \
                + (rho / sigma_v) * v
            tau2 = 1 - rho ** 2
            if tau2 <= 1e-10:
                return 1e20
            tau = np.sqrt(tau2)

            z_val = np.clip(mu / tau, -20, 20)
            Phi_z = np.clip(ndtr(z_val), 1e-15, 1 - 1e-15)

            ll_probit = np.sum(
                y_vec * np.log(Phi_z) + (1 - y_vec) * np.log(1 - Phi_z)
            )
            ll_rf = -n * np.log(sigma_v) - 0.5 * n * np.log(2 * np.pi) \
                - 0.5 * np.sum(v ** 2) / sigma_v ** 2

            ll = ll_probit + ll_rf
            return -ll if np.isfinite(ll) else 1e20

        def score_contributions(theta):
            """Return (n, p) matrix of per-observation score vectors."""
            b0 = theta[off_b0] if is_con else 0.0
            b1 = theta[off_b1:off_b1 + n_b1]
            b2 = theta[off_b2:off_b2 + n_b2]
            pi_vec = theta[off_pi:off_pi + n_pi]
            ath_rho = theta[off_ath_rho]
            ln_sv = theta[off_ln_sv]

            rho = np.tanh(ath_rho)
            sigma_v = np.exp(ln_sv)
            drho = 1 - rho ** 2  # d(tanh)/d(ath_rho)

            v = X2[:, 0] - Z_full @ pi_vec

            mu = b0 + (X1 @ b1 if n_exog > 0 else 0.0) + X2 @ b2 \
                + (rho / sigma_v) * v
            tau2 = 1 - rho ** 2
            if tau2 <= 1e-10:
                return np.zeros((n, n_params))
            tau = np.sqrt(tau2)

            z_val = np.clip(mu / tau, -20, 20)
            Phi_z = np.clip(ndtr(z_val), 1e-15, 1 - 1e-15)
            phi_z = sp_stats.norm.pdf(z_val)

            # d(ll_probit_i)/dz_i
            dll_probit_dz = y_vec * phi_z / Phi_z \
                - (1 - y_vec) * phi_z / (1 - Phi_z)

            S = np.zeros((n, n_params))

            # Probit contribution via chain rule: dz = dmu/tau - mu*dtau/tau^2
            inv_tau = 1.0 / tau

            # b0
            if is_con:
                S[:, off_b0] = dll_probit_dz * inv_tau

            # b1
            for j in range(n_b1):
                S[:, off_b1 + j] = dll_probit_dz * (X1[:, j] * inv_tau)

            # b2
            for j in range(n_b2):
                S[:, off_b2 + j] = dll_probit_dz * (X2[:, j] * inv_tau)

            # pi (via v = X2 - Z_full @ pi, dmu/dpi = -(rho/sigma_v) * Z_full)
            dmu_dpi_coeff = -(rho / sigma_v)
            dll_probit_dpi = dll_probit_dz * (dmu_dpi_coeff * inv_tau)
            for j in range(n_pi):
                S[:, off_pi + j] = dll_probit_dpi * Z_full[:, j] \
                    + v * Z_full[:, j] / sigma_v ** 2

            # ath_rho
            dmu_drho = drho / sigma_v * v
            dtau_drho = -rho * drho / tau
            dz_drho = (dmu_drho * tau - mu * dtau_drho) / (tau ** 2)
            S[:, off_ath_rho] = dll_probit_dz * dz_drho

            # ln_sv
            dmu_dlnsv = -(rho / sigma_v) * v  # rho/sigma_v * v, d/d(ln_sv) gives -rho*v/sigma_v
            dz_dlnsv = dmu_dlnsv * inv_tau
            dll_rf_dlnsv = -1 + v ** 2 / sigma_v ** 2
            S[:, off_ln_sv] = dll_probit_dz * dz_dlnsv + dll_rf_dlnsv

            return S

        # Two-stage optimization: Nelder-Mead for robustness, then L-BFGS-B
        res1 = minimize(neg_log_lik, theta0, method='Nelder-Mead',
                        options={'maxiter': 50000, 'xatol': 1e-10,
                                 'fatol': 1e-10})
        res = minimize(neg_log_lik, res1.x, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12})
        if res.fun > res1.fun:
            res = res1
        theta_hat = res.x
        converged = res.success or res1.success

        b0 = theta_hat[off_b0] if is_con else 0.0
        b1 = theta_hat[off_b1:off_b1 + n_b1]
        b2 = theta_hat[off_b2:off_b2 + n_b2]
        ath_rho = theta_hat[off_ath_rho]
        ln_sv = theta_hat[off_ln_sv]
        rho = np.tanh(ath_rho)
        sigma_v = np.exp(ln_sv)

        if is_con:
            var_names = endog_names + exog_names + ["_cons"]
            coef = np.concatenate([b2, b1, [b0]])
        else:
            var_names = endog_names + exog_names
            coef = np.concatenate([b2, b1])

        k = len(coef)

        # VCE: analytical Hessian (OIM) inverse, fallback to numerical
        V_full = self._analytical_hessian_inv(
            theta_hat, y_vec, X1, X2, Z_full, n, n_exog, n_endog,
            is_con, off_b0, off_b1, off_b2, off_pi, off_ath_rho,
            off_ln_sv, n_params, k_z)
        if V_full is None:
            V_full = self._numerical_hessian_inv(neg_log_lik, theta_hat,
                                                  n_params)

        if V_full is not None:
            if vce == "robust":
                S = score_contributions(theta_hat)
                B = S.T @ S
                V_full = (n / (n - 1)) * V_full @ B @ V_full

            # Map structural coefficients from full parameter vector
            if is_con:
                idx_map = ([off_b2 + j for j in range(n_b2)]
                           + [off_b1 + j for j in range(n_b1)]
                           + [off_b0])
            else:
                idx_map = ([off_b2 + j for j in range(n_b2)]
                           + [off_b1 + j for j in range(n_b1)])
            V_coef = V_full[np.ix_(idx_map, idx_map)]
            std_err = np.sqrt(np.maximum(np.diag(V_coef), 0))
        else:
            V_coef = None
            std_err = np.ones(k) * np.nan

        z_stat = coef / np.where(std_err > 0, std_err, np.nan)
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        ll = -neg_log_lik(theta_hat)

        df_m = k - 1 if is_con else k
        mask = [i for i, nm in enumerate(var_names) if nm != "_cons"]
        if mask and V_coef is not None:
            V_mask = V_coef[np.ix_(mask, mask)]
            try:
                chi2 = float(coef[mask] @ np.linalg.solve(V_mask, coef[mask]))
            except (np.linalg.LinAlgError, ValueError):
                chi2 = float(coef[mask] @ np.linalg.pinv(V_mask) @ coef[mask])
            chi2_pval = float(1 - sp_stats.chi2.cdf(chi2, len(mask)))
        else:
            chi2 = 0.0
            chi2_pval = 1.0

        rho_idx = off_ath_rho
        if V_full is not None:
            rho_se = np.sqrt(max(V_full[rho_idx, rho_idx], 0)) * (1 - rho ** 2)
            endog_test_stat = float(ath_rho ** 2 / max(V_full[rho_idx, rho_idx], 1e-20))
            endog_test_pval = float(1 - sp_stats.chi2.cdf(endog_test_stat, 1))
        else:
            rho_se = np.nan
            endog_test_stat = None
            endog_test_pval = None

        return IVProbitResult(
            coef=coef, std_err=std_err, z_stat=z_stat, p_value=p_value,
            n_obs=n, ll=ll, chi2=chi2, chi2_pval=chi2_pval,
            rho=rho, rho_se=rho_se,
            endog_test_stat=endog_test_stat,
            endog_test_pval=endog_test_pval,
            var_names=var_names, y_name=y_name,
            method="mle", converged=converged,
            vce_type=vce,
        )

    def _fit_twostep(self, y_vec, X1, X2, Z_full, n, n_exog, n_endog,
                     is_con, exog_names, endog_names, y_name, vce):
        """Newey (1987) minimum chi-squared two-step IV probit."""
        k_x = Z_full.shape[1]  # number of exogenous vars (incl. const & instruments)

        # --- Step 1: OLS each endog var on all exog vars ---
        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        pi_hat = np.zeros((k_x, n_endog))
        v_hat = np.zeros((n, n_endog))
        for j in range(n_endog):
            pi_hat[:, j] = ZtZ_inv @ Z_full.T @ X2[:, j]
            v_hat[:, j] = X2[:, j] - Z_full @ pi_hat[:, j]

        # Construct D(pi_hat) matrix: k_x rows, k_delta columns
        # k_delta = n_endog + n_exog + (1 if is_con else 0)
        k_delta = n_endog + n_exog + (1 if is_con else 0)
        D = np.zeros((k_x, k_delta))
        # Column 0..n_endog-1: endog columns (pi_hat for each endog var)
        for j in range(n_endog):
            D[:, j] = pi_hat[:, j]
        # Column n_endog..n_endog+n_exog-1: exog (X1) indicator columns
        # X1 vars appear in Z_full at indices [1 : 1+n_exog] when is_con,
        # or [0 : n_exog] when not is_con
        x1_start = 1 if is_con else 0
        for j in range(n_exog):
            D[x1_start + j, n_endog + j] = 1.0
        # Last column: constant indicator (const is at index 0 in Z_full)
        if is_con:
            D[0, k_delta - 1] = 1.0

        # --- Step 2: Reduced-form probit: probit y on [x, v_hat] ---
        X_rf = np.column_stack([Z_full, v_hat])
        theta_rf = self._simple_probit(y_vec, X_rf)
        tilde_delta = theta_rf[:k_x]  # coefficients on x
        alpha_vhat = theta_rf[k_x:]   # coefficients on v_hat

        # Compute V_probit using observed information (exact Hessian)
        # Stata uses observed info, not expected (Fisher) info
        xb = np.clip(X_rf @ theta_rf, -20, 20)
        p_rf = np.clip(ndtr(xb), 1e-15, 1 - 1e-15)
        phi_rf = sp_stats.norm.pdf(xb)
        r_rf = y_vec - p_rf
        # Exact d²l/dη² = -φ²/(p(1-p)) + (y-p)[-ηφ/(p(1-p)) - (1-2p)φ²/(p(1-p))²]
        pmc = p_rf * (1 - p_rf)
        q_rf = (-phi_rf ** 2 / pmc
                + r_rf * (-xb * phi_rf / pmc
                          - (1 - 2 * p_rf) * phi_rf ** 2 / pmc ** 2))
        H_rf = X_rf.T @ (q_rf[:, None] * X_rf)
        try:
            V_probit_full = np.linalg.inv(-H_rf)
        except np.linalg.LinAlgError:
            V_probit_full = np.linalg.pinv(-H_rf)
        V_probit = V_probit_full[:k_x, :k_x]

        # --- Step 3: 2SIV probit to extract endogenous coefficients ---
        X_z = np.column_stack([X2, X1]) if n_exog > 0 else X2.copy()
        if is_con:
            X_z = np.column_stack([X_z, np.ones(n)])
        X_zv = np.column_stack([X_z, v_hat])
        theta_zv = self._simple_probit(y_vec, X_zv)
        alpha_hat = theta_zv[:n_endog]  # coefficients on endogenous vars

        # --- Step 4: Correction for Omega (Newey 1987) ---
        # Stata manual: reg y2i*(b_hat - b) on xi, add covariance to J^{-1}
        # b_hat = endog coef from 2SIV, b = v_hat coef from RF probit
        # For just-identified: diff = 0, correction vanishes
        diff_val = theta_zv[0] - alpha_vhat[0]

        y_corr = X2[:, 0] * diff_val
        beta_corr = ZtZ_inv @ Z_full.T @ y_corr
        resid_corr = y_corr - Z_full @ beta_corr
        s2_corr = np.sum(resid_corr ** 2) / (n - k_x)
        V_corr = s2_corr * ZtZ_inv

        Omega_hat = V_probit + V_corr

        # --- Step 5: Newey efficient estimator ---
        Omega_inv = np.linalg.inv(Omega_hat)
        DtOiD = D.T @ Omega_inv @ D
        delta_hat = np.linalg.solve(DtOiD, D.T @ Omega_inv @ tilde_delta)

        # --- Step 6: VCE ---
        V_delta = np.linalg.inv(DtOiD)
        se_delta = np.sqrt(np.maximum(np.diag(V_delta), 0))

        # Build output in [endog, exog, cons] order
        coef = delta_hat
        std_err = se_delta
        var_names = endog_names + exog_names + (["_cons"] if is_con else [])

        z_stat = coef / np.where(std_err > 0, std_err, np.nan)
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # Log-likelihood from reduced-form probit
        ll = float(np.sum(
            y_vec * np.log(p_rf) + (1 - y_vec) * np.log(1 - p_rf)
        ))

        # Wald test on structural coefficients (exclude constant)
        k = len(coef)
        mask = [i for i, nm in enumerate(var_names) if nm != "_cons"]
        if mask:
            V_mask = V_delta[np.ix_(mask, mask)]
            try:
                chi2 = float(coef[mask] @ np.linalg.solve(V_mask, coef[mask]))
            except (np.linalg.LinAlgError, ValueError):
                chi2 = float(coef[mask] @ np.linalg.pinv(V_mask) @ coef[mask])
            chi2_pval = float(1 - sp_stats.chi2.cdf(chi2, len(mask)))
        else:
            chi2 = 0.0
            chi2_pval = 1.0

        # Endogeneity test: Wald test on v_hat from 2SIV probit (Step 3)
        # Stata tests v_hat coefficient in probit y on [endog, exog, cons, v_hat]
        if n_endog > 0:
            X_zv_xb = np.clip(X_zv @ theta_zv, -20, 20)
            p_zv = np.clip(ndtr(X_zv_xb), 1e-15, 1 - 1e-15)
            phi_zv = sp_stats.norm.pdf(X_zv_xb)
            r_zv = y_vec - p_zv
            pmc_zv = p_zv * (1 - p_zv)
            q_zv = (-phi_zv ** 2 / pmc_zv
                     + r_zv * (-X_zv_xb * phi_zv / pmc_zv
                               - (1 - 2 * p_zv) * phi_zv ** 2 / pmc_zv ** 2))
            H_zv = X_zv.T @ (q_zv[:, None] * X_zv)
            try:
                V_zv = np.linalg.inv(-H_zv)
            except np.linalg.LinAlgError:
                V_zv = np.linalg.pinv(-H_zv)
            # v_hat is the last n_endog columns
            vhat_idx = list(range(X_zv.shape[1] - n_endog, X_zv.shape[1]))
            V_vhat = V_zv[np.ix_(vhat_idx, vhat_idx)]
            alpha_zv = theta_zv[vhat_idx]
            try:
                endog_test_stat = float(
                    alpha_zv @ np.linalg.solve(V_vhat, alpha_zv)
                )
            except (np.linalg.LinAlgError, ValueError):
                endog_test_stat = float(
                    alpha_zv @ np.linalg.pinv(V_vhat) @ alpha_zv
                )
            endog_test_pval = float(
                1 - sp_stats.chi2.cdf(endog_test_stat, n_endog)
            )
        else:
            endog_test_stat = None
            endog_test_pval = None

        return IVProbitResult(
            coef=coef, std_err=std_err, z_stat=z_stat, p_value=p_value,
            n_obs=n, ll=ll, chi2=chi2, chi2_pval=chi2_pval,
            rho=None, rho_se=None,
            endog_test_stat=endog_test_stat,
            endog_test_pval=endog_test_pval,
            var_names=var_names, y_name=y_name,
            method="twostep", converged=True,
            vce_type=vce,
        )

    @staticmethod
    def _simple_probit(y, X, max_iter=500, tol=1e-12):
        """Newton-Raphson probit, returns beta."""
        n, k = X.shape
        beta = np.zeros(k)
        for _ in range(max_iter):
            xb = np.clip(X @ beta, -20, 20)
            p = np.clip(ndtr(xb), 1e-15, 1 - 1e-15)
            phi = sp_stats.norm.pdf(xb)
            score = X.T @ ((y - p) * phi / (p * (1 - p)))
            W = phi ** 2 / (p * (1 - p))
            H = -X.T @ (W[:, None] * X)
            try:
                delta = np.linalg.solve(H, score)
            except np.linalg.LinAlgError:
                break
            beta = beta - delta
            if np.max(np.abs(delta)) < tol:
                break
        return beta

    @staticmethod
    def _analytical_hessian_inv(
            theta, y_vec, X1, X2, Z_full, n, n_exog, n_endog,
            is_con, off_b0, off_b1, off_b2, off_pi, off_ath_rho,
            off_ln_sv, n_params, k_z):
        """Compute inverse of analytical Hessian (OIM) for IV probit MLE.

        Returns V = (-H)^{-1} where H is the Hessian of log-likelihood.
        """
        b0 = theta[off_b0] if is_con else 0.0
        b1 = theta[off_b1:off_b1 + n_exog]
        b2 = theta[off_b2:off_b2 + n_endog]
        n_pi = k_z * n_endog
        pi_vec = theta[off_pi:off_pi + n_pi]
        ath_rho = theta[off_ath_rho]
        ln_sv = theta[off_ln_sv]

        rho = np.tanh(ath_rho)
        drho = 1 - rho ** 2
        sigma_v = np.exp(ln_sv)
        inv_sv = 1.0 / sigma_v
        tau2 = 1 - rho ** 2
        if tau2 <= 1e-10:
            return None
        tau = np.sqrt(tau2)
        itau = 1.0 / tau
        itau3 = itau ** 3

        v = X2[:, 0] - Z_full @ pi_vec
        rho_sv = rho * inv_sv

        mu = b0 + (X1 @ b1 if n_exog > 0 else 0.0) + X2 @ b2 + rho_sv * v
        z_val = np.clip(mu * itau, -20, 20)
        Phi_z = np.clip(ndtr(z_val), 1e-15, 1 - 1e-15)
        phi_z = sp_stats.norm.pdf(z_val)

        # w_i = d(ll_probit_i)/dz_i
        w = y_vec * phi_z / Phi_z - (1 - y_vec) * phi_z / (1 - Phi_z)
        # w2_i = d2(ll_probit_i)/dz_i^2
        lam = phi_z / Phi_z
        lam1 = phi_z / (1 - Phi_z)
        # More numerically stable: dw/dz = -z*w - w^2
        w2 = -z_val * w - w ** 2

        # === Build g = dz/dtheta, (n, n_params) ===
        g = np.zeros((n, n_params))
        if is_con:
            g[:, off_b0] = itau
        for j in range(n_exog):
            g[:, off_b1 + j] = X1[:, j] * itau
        for j in range(n_endog):
            g[:, off_b2 + j] = X2[:, j] * itau
        for j in range(n_pi):
            g[:, off_pi + j] = -rho_sv * Z_full[:, j] * itau

        # dz/d(ath_rho) = dmu/d(ath_rho) * itau + mu * d(itau)/d(ath_rho)
        # dmu/d(ath_rho) = drho/sigma_v * v
        # d(itau)/d(ath_rho) = rho * drho * itau^3
        ditau_dath_rho = rho * drho * itau3
        g[:, off_ath_rho] = drho * inv_sv * v * itau + mu * ditau_dath_rho
        g[:, off_ln_sv] = -rho_sv * v * itau

        # Probit part of Hessian: sum_i [w2_i * g_i * g_i' + w_i * h_i]
        # First term: w2 * g*g'
        H = (g * w2[:, None]).T @ g

        # Second term: w * h_jk for non-zero second derivatives of z
        # z = mu * itau, so d2z = d2mu*itau + 2*dmu*ditau + mu*d2itau
        # ditau only depends on ath_rho, d2itau only for (ath_rho,ath_rho)

        # h(b_j, ath_rho) = X_j * ditau_dath_rho  (for b-type params)
        for j_idx, off_j in (
            [(-1, off_b0)] if is_con else []
        ) + [(j, off_b1 + j) for j in range(n_exog)] \
          + [(j, off_b2 + j) for j in range(n_endog)]:
            x_col = np.ones(n) if off_j == off_b0 else (
                X1[:, j_idx] if off_b1 <= off_j < off_b1 + n_exog
                else X2[:, j_idx])
            h_val = x_col * ditau_dath_rho
            H[off_j, off_ath_rho] += np.sum(w * h_val)
            H[off_ath_rho, off_j] += np.sum(w * h_val)

        # h(pi_j, ath_rho)
        for j in range(n_pi):
            h_val = (-drho * inv_sv * Z_full[:, j] * itau
                     - rho_sv * Z_full[:, j] * ditau_dath_rho)
            H[off_pi + j, off_ath_rho] += np.sum(w * h_val)
            H[off_ath_rho, off_pi + j] += np.sum(w * h_val)

        # h(ath_rho, ath_rho) = d2mu*itau + 2*dmu*ditau + mu*d2itau
        d2rho_val = -2 * rho * drho
        d2mu_arar = d2rho_val * inv_sv * v
        d2itau_arar = (drho**2 + rho * d2rho_val) * itau3 \
            + 3 * rho * drho * itau**2 * ditau_dath_rho
        h_arar = d2mu_arar * itau + 2 * (drho * inv_sv * v) * ditau_dath_rho \
            + mu * d2itau_arar
        H[off_ath_rho, off_ath_rho] += np.sum(w * h_arar)

        # h(ath_rho, ln_sv) = d2mu * itau + dmu(ln_sv) * ditau
        # d2mu/(ar, ln_sv) = d(drho*inv_sv * v)/d(ln_sv) = -drho*inv_sv * v
        # dmu/d(ln_sv) = -rho_sv * v
        h_ar_ls = -drho * inv_sv * v * itau + (-rho_sv * v) * ditau_dath_rho
        H[off_ath_rho, off_ln_sv] += np.sum(w * h_ar_ls)
        H[off_ln_sv, off_ath_rho] += np.sum(w * h_ar_ls)

        # h(ln_sv, ln_sv) = d2mu(ln_sv,ln_sv) * itau
        # d2mu/d(ln_sv)^2 = d(-rho_sv*v)/d(ln_sv) = rho_sv*v
        H[off_ln_sv, off_ln_sv] += np.sum(w * rho_sv * v * itau)

        # h(ln_sv, pi_j): d(-rho_sv*Z_j*itau)/d(ln_sv) = rho_sv*Z_j*itau
        for j in range(n_pi):
            h_val = rho_sv * Z_full[:, j] * itau
            H[off_ln_sv, off_pi + j] += np.sum(w * h_val)
            H[off_pi + j, off_ln_sv] += np.sum(w * h_val)

        # h(ln_sv, b_j) = 0 (ditau doesn't depend on ln_sv)

        # === Reduced-form Hessian ===
        # ll_rf = -n*ln(sigma_v) - 0.5*sum(v^2)/sigma_v^2
        sv2 = sigma_v ** 2
        isv2 = 1.0 / sv2
        v2_sum = np.sum(v ** 2)

        # d2(ll_rf)/dpi_j dpi_k = -Z_j'Z_k / sigma_v^2
        ZtZ = Z_full.T @ Z_full
        for j in range(n_pi):
            for k_idx in range(j, n_pi):
                val = -ZtZ[j, k_idx] * isv2
                H[off_pi + j, off_pi + k_idx] += val
                if j != k_idx:
                    H[off_pi + k_idx, off_pi + j] += val

        # d2(ll_rf)/dpi_j d(ln_sv) = -2 * v'Z_j / sigma_v^2
        vZ = v @ Z_full  # (n_pi,)
        for j in range(n_pi):
            val = -2 * vZ[j] * isv2
            H[off_pi + j, off_ln_sv] += val
            H[off_ln_sv, off_pi + j] += val

        # d2(ll_rf)/d(ln_sv)^2 = -2 * v2_sum/sigma_v^2
        H[off_ln_sv, off_ln_sv] += -2 * v2_sum * isv2

        # VCE = (-H)^{-1}
        try:
            return np.linalg.inv(-H)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(-H) if np.any(H != 0) else None

    @staticmethod
    def _numerical_hessian_inv(neg_ll_fn, theta, n_params, eps_scale=1e-5):
        """Numerical Hessian inverse using relative step sizes."""
        p = n_params
        eps = np.maximum(eps_scale, np.abs(theta) * eps_scale)
        H = np.zeros((p, p))
        f0 = neg_ll_fn(theta)
        if not np.isfinite(f0):
            return None
        for i in range(p):
            ei = np.zeros(p)
            ei[i] = eps[i]
            fp = neg_ll_fn(theta + ei)
            fm = neg_ll_fn(theta - ei)
            if not (np.isfinite(fp) and np.isfinite(fm)):
                return None
            H[i, i] = (fp - 2 * f0 + fm) / eps[i] ** 2
            for j in range(i + 1, p):
                ej = np.zeros(p)
                ej[j] = eps[j]
                fpp = neg_ll_fn(theta + ei + ej)
                fpm = neg_ll_fn(theta + ei - ej)
                fmp = neg_ll_fn(theta - ei + ej)
                fmm = neg_ll_fn(theta - ei - ej)
                if not all(np.isfinite(x) for x in (fpp, fpm, fmp, fmm)):
                    return None
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps[i] * eps[j])
                H[j, i] = H[i, j]
        # H = Hessian of neg_log_lik (positive definite at MLE)
        # VCE = inv(H), since -Hessian(log_lik) = Hessian(neg_log_lik) = H
        try:
            return np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(H)

    @staticmethod
    def _compute_cons_var(*args, **kwargs):
        raise NotImplementedError("No longer needed; intercept is now a direct parameter")

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for IV probit estimation")
