#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : iv.py

import numpy as np
from scipy import stats as sp_stats

from tabra.models.estimate.base import BaseModel
from tabra.results.iv_result import IVResult


class IVModel(BaseModel):
    """Instrumental-variables regression (2SLS, GMM, LIML, CUE, Fuller, k-class)."""

    def fit(self, df, y, exog, endog, instruments,
            estimator="2sls", vce="unadjusted", is_con=True,
            fuller_alpha=1.0, kclass_k=None, cluster=None):
        valid_estimators = ("2sls", "gmm", "liml", "cue", "fuller", "kclass")
        if estimator not in valid_estimators:
            raise ValueError(
                f"Unknown estimator '{estimator}', use one of {valid_estimators}"
            )
        if vce not in ("unadjusted", "robust", "cluster"):
            raise ValueError(
                f"Unknown vce '{vce}', use 'unadjusted', 'robust', or 'cluster'"
            )
        if vce == "cluster" and cluster is None:
            raise ValueError("cluster vce requires cluster variable names")

        exog = list(exog)
        endog = list(endog)
        instruments = list(instruments)

        # Underidentification check
        L = len(instruments)
        K_endog = len(endog)
        if L < K_endog:
            raise ValueError(
                f"Underidentified: {K_endog} endogenous variables but only "
                f"{L} instruments."
            )

        # Prepare data
        all_cols = [y] + exog + endog + instruments
        if cluster is not None:
            clust_cols = cluster if isinstance(cluster, list) else [cluster]
            all_cols.extend(clust_cols)
        df = df[all_cols].dropna()

        y_vec = df[y].values.astype(float)
        X1 = df[exog].values.astype(float) if exog else np.empty((len(df), 0))
        X2 = df[endog].values.astype(float)
        Z_inst = df[instruments].values.astype(float)

        n = len(y_vec)
        n_exog = X1.shape[1]
        n_endog = X2.shape[1]

        # Z = [const, X1, Z_inst], X = [X2, X1, const]
        if is_con:
            const = np.ones((n, 1))
            if n_exog > 0:
                Z = np.column_stack([const, X1, Z_inst])
                X = np.column_stack([X2, X1, const])
            else:
                Z = np.column_stack([const, Z_inst])
                X = np.column_stack([X2, const])
            var_names = endog + exog + ["_cons"]
        else:
            if n_exog > 0:
                Z = np.column_stack([X1, Z_inst])
                X = np.column_stack([X2, X1])
            else:
                Z = Z_inst
                X = X2.copy()
            var_names = endog + exog

        k = X.shape[1]
        PZ = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        MZ = np.eye(n) - PZ

        # Choose estimator
        kclass_kappa = None  # track kappa for k-class VCE
        resid_init = None
        if estimator == "2sls":
            beta, resid = self._fit_2sls(y_vec, X, PZ)
        elif estimator == "gmm":
            beta, resid, resid_init = self._fit_gmm(y_vec, X, Z, PZ, n)
        elif estimator == "liml":
            beta, resid, kclass_kappa = self._fit_liml(
                y_vec, X, Z, PZ, MZ, n_exog, n_endog, n, is_con
            )
        elif estimator == "cue":
            if vce == "unadjusted":
                # Homoskedastic CUE is equivalent to LIML
                beta, resid, kclass_kappa = self._fit_liml(
                    y_vec, X, Z, PZ, MZ, n_exog, n_endog, n, is_con
                )
            else:
                beta, resid = self._fit_cue(y_vec, X, Z, n)
        elif estimator == "fuller":
            beta, resid, kclass_kappa = self._fit_fuller(
                y_vec, X, Z, PZ, MZ, n_exog, n_endog, n, is_con, fuller_alpha
            )
        elif estimator == "kclass":
            beta, resid, kclass_kappa = self._fit_kclass(y_vec, X, MZ, n, kclass_k)

        # First-stage F (Cragg-Donald)
        first_stage_f = self._first_stage_fstat(X2, Z, Z_inst, is_con, n, n_exog)

        # Underidentification: Anderson LM
        idstat, idpval = self._anderson_lm(X2, Z, Z_inst, n, L, is_con, n_exog)

        # Standard errors
        X_hat = PZ @ X
        XtX_hat_inv = np.linalg.inv(X_hat.T @ X_hat)

        # For k-class estimators, use Q^{-1} instead of (X_hat'X_hat)^{-1}
        # CUE unadjusted is equivalent to LIML in coefficients but uses
        # 2SLS-style VCE (XtX_hat_inv) rather than k-class VCE.
        if kclass_kappa is not None and estimator != "cue":
            Q_kclass = X.T @ (np.eye(n) - kclass_kappa * MZ) @ X
            try:
                Q_kclass_inv = np.linalg.inv(Q_kclass)
            except np.linalg.LinAlgError:
                Q_kclass_inv = XtX_hat_inv
        else:
            Q_kclass_inv = None

        if estimator == "gmm":
            # GMM VCE: sandwich with bread from 2SLS residuals, meat from GMM residuals
            # V = Q^{-1} R Q^{-1}
            # Q = X'Z inv(S_2sls) Z'X,  R = X'Z inv(S_2sls) S_gmm inv(S_2sls) Z'X
            S_bread = (Z.T * resid_init ** 2) @ Z
            S_meat = (Z.T * resid ** 2) @ Z
            try:
                S_bread_inv = np.linalg.inv(S_bread)
            except np.linalg.LinAlgError:
                S_bread_inv = np.linalg.pinv(S_bread)
            XZSiZ = X.T @ Z @ S_bread_inv @ Z.T @ X
            try:
                XZSiZ_inv = np.linalg.inv(XZSiZ)
            except np.linalg.LinAlgError:
                sigma2 = resid @ resid / n
                var_beta = sigma2 * XtX_hat_inv
            else:
                meat = X.T @ Z @ S_bread_inv @ S_meat @ S_bread_inv @ Z.T @ X
                var_beta = XZSiZ_inv @ meat @ XZSiZ_inv
        elif estimator == "cue":
            if vce == "unadjusted":
                # Homoskedastic CUE: same VCE as 2SLS with CUE residuals
                sigma2 = resid @ resid / n
                var_beta = sigma2 * XtX_hat_inv
            else:
                # CUE VCE: use own residuals for S (no /n division)
                S = (Z.T * resid ** 2) @ Z
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)
                XZWSZX = X.T @ Z @ S_inv @ Z.T @ X
                try:
                    var_beta = np.linalg.inv(XZWSZX)
                except np.linalg.LinAlgError:
                    sigma2 = resid @ resid / n
                    var_beta = sigma2 * XtX_hat_inv
        elif vce == "unadjusted":
            sigma2 = resid @ resid / n
            V_inv = Q_kclass_inv if Q_kclass_inv is not None else XtX_hat_inv
            var_beta = sigma2 * V_inv
        elif vce == "robust":
            e2 = resid ** 2
            meat = (X_hat.T * e2) @ X_hat
            bread_inv = Q_kclass_inv if Q_kclass_inv is not None else XtX_hat_inv
            var_beta = bread_inv @ meat @ bread_inv
        elif vce == "cluster":
            clust_cols = cluster if isinstance(cluster, list) else [cluster]
            bread_inv = Q_kclass_inv if Q_kclass_inv is not None else XtX_hat_inv
            var_beta = self._cluster_vce(
                X_hat, resid, n, k, bread_inv, df, clust_cols
            )

        std_err = np.sqrt(np.maximum(np.diag(var_beta), 0))
        z_stat = beta / std_err
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # R-squared
        y_mean = np.mean(y_vec)
        SST = float((y_vec - y_mean) @ (y_vec - y_mean))
        SSR = float(resid @ resid)
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        df_resid = n - k
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0
        root_mse = np.sqrt(SSR / n) if n > 0 else 0.0

        df_m = k - 1 if is_con else k

        # Overidentification J test
        if L > K_endog:
            j_df = L - K_endog
            if estimator == "gmm":
                # Hansen J: n * g_bar' W g_bar = Ze' W Ze / n
                S_gmm = (Z.T * resid ** 2) @ Z / n
                try:
                    W_gmm = np.linalg.inv(S_gmm)
                except np.linalg.LinAlgError:
                    W_gmm = np.linalg.pinv(S_gmm)
                Ze = Z.T @ resid
                j_stat = float(Ze @ W_gmm @ Ze) / n
            elif estimator == "cue":
                S_cue = (Z.T * resid ** 2) @ Z / n
                try:
                    W_cue = np.linalg.inv(S_cue)
                except np.linalg.LinAlgError:
                    W_cue = np.linalg.pinv(S_cue)
                Ze = Z.T @ resid
                j_stat = float(Ze @ W_cue @ Ze) / n
            else:
                # Sargan statistic for 2SLS/LIML/k-class
                sigma2_hat = SSR / n
                j_stat = float(resid @ PZ @ resid / sigma2_hat)
            j_pval = float(1 - sp_stats.chi2.cdf(j_stat, j_df))
        else:
            j_stat = None
            j_pval = None

        # DWH endogeneity test
        endog_test_stat, endog_test_pval, wh_f, wh_pval = self._dwh_test(
            y_vec, X, Z, PZ, n, k, n_endog, n_exog, is_con
        )

        vce_label = vce
        if vce == "cluster" and cluster is not None:
            vce_label = f"cluster({', '.join(clust_cols)})"

        return IVResult(
            coef=beta, std_err=std_err, z_stat=z_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            root_mse=root_mse,
            n_obs=n, k_vars=k, df_m=df_m,
            first_stage_f=first_stage_f,
            j_stat=j_stat, j_pval=j_pval,
            endog_test_stat=endog_test_stat,
            endog_test_pval=endog_test_pval,
            var_names=var_names, y_name=y,
            estimator=estimator, vce_type=vce,
            endog_names=endog, exog_names=exog,
            inst_names=instruments,
            idstat=idstat, idpval=idpval,
            widstat=first_stage_f,
            kappa=kclass_kappa,
            var_beta=var_beta,
        )

    # ── Estimators ──

    def _fit_2sls(self, y, X, PZ):
        X_hat = PZ @ X
        beta = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ y
        resid = y - X @ beta
        return beta, resid

    def _fit_gmm(self, y, X, Z, PZ, n):
        _, resid_init = self._fit_2sls(y, X, PZ)
        S = (Z.T * resid_init ** 2) @ Z / n
        W = np.linalg.inv(S)
        XZW = X.T @ Z @ W
        beta = np.linalg.inv(XZW @ Z.T @ X) @ XZW @ Z.T @ y
        resid = y - X @ beta
        return beta, resid, resid_init

    def _liml_kappa(self, y, X, Z, MZ, n_exog, n_endog, n, is_con):
        """Compute LIML kappa via generalized eigenvalue problem."""
        from scipy.linalg import eig as sp_eig
        X_endog = X[:, :n_endog]
        Y = np.column_stack([y.reshape(-1, 1), X_endog])
        if is_con:
            Z_exog = Z[:, :1 + n_exog] if n_exog > 0 else Z[:, :1]
        else:
            Z_exog = Z[:, :n_exog] if n_exog > 0 else Z[:, :0]
        if Z_exog.shape[1] > 0 and Z_exog.shape[1] < n:
            M_X1 = np.eye(n) - Z_exog @ np.linalg.inv(Z_exog.T @ Z_exog) @ Z_exog.T
        else:
            M_X1 = np.eye(n)
        # LIML: kappa = min eigenvalue of (Y'MZ'Y)^{-1}(Y'M_X1'Y)
        A = Y.T @ MZ @ Y
        B = Y.T @ M_X1 @ Y
        # Use scipy generalized eigenvalue for numerical stability
        try:
            eigvals = sp_eig(B, A, right=False)
            eigvals = np.real(eigvals)
        except Exception:
            try:
                eigvals = np.linalg.eigvals(np.linalg.solve(A, B))
                eigvals = np.real(eigvals)
            except np.linalg.LinAlgError:
                eigvals = np.linalg.eigvals(np.linalg.pinv(A) @ B)
                eigvals = np.real(eigvals)
        # Filter out inf/nan and pick minimum
        valid = eigvals[np.isfinite(eigvals)]
        if len(valid) == 0:
            return 1.0  # fallback to 2SLS
        kappa = float(np.min(valid))
        return kappa

    def _fit_liml(self, y, X, Z, PZ, MZ, n_exog, n_endog, n, is_con):
        kappa = self._liml_kappa(y, X, Z, MZ, n_exog, n_endog, n, is_con)
        beta, resid = self._kclass_estimate(y, X, MZ, n, kappa)
        return beta, resid, kappa

    def _fit_cue(self, y, X, Z, n):
        """Continuously-updated GMM (Hansen-Heaton-Yaron 1996)."""
        from scipy.optimize import minimize

        k = X.shape[1]
        L = Z.shape[1]

        # Init from 2SLS
        PZ = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        beta_init, _ = self._fit_2sls(y, X, PZ)

        def cue_objective(beta):
            resid = y - X @ beta
            # Weight matrix depends on beta
            S = (Z.T * resid ** 2) @ Z / n
            try:
                W = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                W = np.linalg.pinv(S)
            # GMM criterion: (Z'e)' W (Z'e) / n
            Ze = Z.T @ resid
            return float(Ze @ W @ Ze) / n

        # Use Nelder-Mead for robust global convergence
        res = minimize(cue_objective, beta_init, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                'adaptive': True})
        beta = res.x
        resid = y - X @ beta
        return beta, resid

    def _fit_fuller(self, y, X, Z, PZ, MZ, n_exog, n_endog, n, is_con, alpha):
        """Fuller-modified LIML: κ = κ_liml - alpha/(n - L - K1)."""
        kappa_liml = self._liml_kappa(y, X, Z, MZ, n_exog, n_endog, n, is_con)

        # Fuller adjustment
        if is_con:
            L_excl = Z.shape[1] - (1 + n_exog)
        else:
            L_excl = Z.shape[1] - n_exog
        if is_con:
            Z_exog = Z[:, :1 + n_exog] if n_exog > 0 else Z[:, :1]
        else:
            Z_exog = Z[:, :n_exog] if n_exog > 0 else Z[:, :0]
        denom = max(n - L_excl - Z_exog.shape[1], 1)
        kappa = kappa_liml - alpha / denom

        beta, resid = self._kclass_estimate(y, X, MZ, n, kappa)
        return beta, resid, kappa

    def _fit_kclass(self, y, X, MZ, n, kclass_k):
        """General k-class estimator."""
        if kclass_k is None:
            raise ValueError("kclass requires kclass_k parameter")
        beta, resid = self._kclass_estimate(y, X, MZ, n, kclass_k)
        return beta, resid, kclass_k

    def _kclass_estimate(self, y, X, MZ, n, kappa):
        """β = (X'(I - κ·M_Z)X)^{-1} X'(I - κ·M_Z)y"""
        Q = X.T @ (np.eye(n) - kappa * MZ) @ X
        q = X.T @ (np.eye(n) - kappa * MZ) @ y
        try:
            beta = np.linalg.solve(Q, q)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Q, q, rcond=None)[0]
        resid = y - X @ beta
        return beta, resid

    # ── Diagnostics ──

    def _anderson_lm(self, X2, Z, Z_inst, n, L, is_con, n_exog):
        """Anderson (1951) canonical correlations LM test for underidentification.

        Uses partial R2: regress endogenous on full Z, compare to regressing
        on exogenous only (Stata estat firststage convention).
        """
        k_Z = Z.shape[1]
        ZtZ_inv = np.linalg.inv(Z.T @ Z)

        if X2.shape[1] == 1:
            y_j = X2[:, 0]
            # Full model: y_j on Z (exog + excluded instruments)
            beta_j = ZtZ_inv @ Z.T @ y_j
            resid_j = y_j - Z @ beta_j
            SSR_full = resid_j @ resid_j

            # Restricted model: y_j on exogenous only
            if is_con:
                if n_exog > 0:
                    Z_exog = Z[:, :1 + n_exog]
                else:
                    Z_exog = Z[:, :1]
            else:
                if n_exog > 0:
                    Z_exog = Z[:, :n_exog]
                else:
                    SST = (y_j - np.mean(y_j)) @ (y_j - np.mean(y_j))
                    cc2 = 1.0 if SST > 0 else 0.0
                    lm_stat = float(n * cc2)
                    df_test = max(L, 1)
                    pval = float(1 - sp_stats.chi2.cdf(lm_stat, df_test))
                    return lm_stat, pval

            if Z_exog.shape[1] > 0:
                beta_r = np.linalg.inv(Z_exog.T @ Z_exog) @ Z_exog.T @ y_j
                resid_r = y_j - Z_exog @ beta_r
                SSR_r = resid_r @ resid_r
            else:
                SSR_r = (y_j - np.mean(y_j)) @ (y_j - np.mean(y_j))

            partial_r2 = 1 - SSR_full / SSR_r if SSR_r > 0 else 0.0
            lm_stat = float(n * partial_r2)
        else:
            # Multiple endogenous: use smallest canonical correlation
            # CCA between X2 and Z_inst (excluded instruments only)
            Z_inst_c = Z_inst - Z_inst.mean(axis=0)
            X2_c = X2 - X2.mean(axis=0)
            C = X2_c.T @ Z_inst_c / n
            Vx = np.linalg.pinv(X2_c.T @ X2_c / n)
            Vz = np.linalg.pinv(Z_inst_c.T @ Z_inst_c / n)
            M = Vx @ C @ Vz @ C.T
            try:
                eigvals = np.sort(np.real(np.linalg.eigvals(M)))
            except Exception:
                return 0.0, 1.0
            cc2_min = max(float(eigvals[0]), 0)
            lm_stat = float(n * cc2_min)

        df_test = max(L - X2.shape[1] + 1, 1)
        pval = float(1 - sp_stats.chi2.cdf(lm_stat, df_test))
        return lm_stat, pval

    def _first_stage_fstat(self, X2, Z, Z_inst, is_con, n, n_exog):
        """First-stage F for excluded instruments (Stata estat firststage)."""
        k_Z = Z.shape[1]
        L = Z_inst.shape[1]
        ZtZ_inv = np.linalg.inv(Z.T @ Z)

        f_stats = []
        for j in range(X2.shape[1]):
            y_j = X2[:, j]
            beta_j = ZtZ_inv @ Z.T @ y_j
            resid_j = y_j - Z @ beta_j
            SSR_j = float(np.dot(resid_j, resid_j))

            # Restricted model: y_j on exogenous only (no excluded instruments)
            if is_con:
                if n_exog > 0:
                    Z_exog = Z[:, :1 + n_exog]
                else:
                    Z_exog = Z[:, :1]
            else:
                if n_exog > 0:
                    Z_exog = Z[:, :n_exog]
                else:
                    f_stats.append(0.0)
                    continue

            if Z_exog.shape[1] > 0:
                beta_r = np.linalg.inv(Z_exog.T @ Z_exog) @ Z_exog.T @ y_j
                resid_r = y_j - Z_exog @ beta_r
                SSR_r = float(np.dot(resid_r, resid_r))
            else:
                SSR_r = float(np.dot(y_j - np.mean(y_j), y_j - np.mean(y_j)))

            df_excl = L
            df_resid = n - k_Z
            if df_excl > 0 and df_resid > 0:
                f_j = ((SSR_r - SSR_j) / df_excl) / (SSR_j / df_resid)
            else:
                f_j = 0.0
            f_stats.append(f_j)
        return min(f_stats) if f_stats else 0.0

    def _cluster_vce(self, X_hat, resid, n, k, bread_inv, df, cluster_cols):
        """Cluster-robust VCE."""
        clust_arr = df[cluster_cols[0]].values if len(cluster_cols) == 1 \
            else self._intersect_clusters(df, cluster_cols)
        unique_clusts = np.unique(clust_arr)
        meat = np.zeros((k, k))
        for c in unique_clusts:
            idx = clust_arr == c
            Xc = X_hat[idx]
            ec = resid[idx]
            meat += Xc.T @ np.outer(ec, ec) @ Xc
        return bread_inv @ meat @ bread_inv

    @staticmethod
    def _intersect_clusters(df, cluster_cols):
        arrs = [df[c].values for c in cluster_cols]
        combined = arrs[0].astype(str)
        for a in arrs[1:]:
            combined = combined + "_" + a.astype(str)
        _, codes = np.unique(combined, return_inverse=True)
        return codes

    def _dwh_test(self, y, X, Z, PZ, n, k, n_endog, n_exog, is_con):
        """Durbin score and Wu-Hausman endogeneity tests.

        Returns Durbin (score) chi2 statistic and p-value.
        Wu-Hausman F is stored separately if needed.
        """
        # Step 1: OLS of y on X
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_ols = XtX_inv @ X.T @ y
        resid_ols = y - X @ beta_ols
        SSR_ols = resid_ols @ resid_ols

        # Step 2: First-stage residuals for each endogenous variable
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        # Extract endogenous columns from X
        X_endog = X[:, :n_endog]
        V_hat = np.empty((n, n_endog))
        for j in range(n_endog):
            gamma_j = ZtZ_inv @ Z.T @ X_endog[:, j]
            V_hat[:, j] = X_endog[:, j] - Z @ gamma_j

        # Step 3: Augmented regression: y on X + V_hat
        X_aug = np.column_stack([X, V_hat])
        try:
            XtX_aug_inv = np.linalg.inv(X_aug.T @ X_aug)
        except np.linalg.LinAlgError:
            XtX_aug_inv = np.linalg.pinv(X_aug.T @ X_aug)
        beta_aug = XtX_aug_inv @ X_aug.T @ y
        resid_aug = y - X_aug @ beta_aug
        SSR_aug = resid_aug @ resid_aug

        # Durbin score test: n * (SSR_ols - SSR_aug) / SSR_ols
        m = n_endog
        durbin_chi2 = float(n * (SSR_ols - SSR_aug) / SSR_ols)
        durbin_chi2 = max(durbin_chi2, 0.0)
        durbin_pval = float(1 - sp_stats.chi2.cdf(durbin_chi2, m))

        # Wu-Hausman F test
        df_den = n - k - m
        if df_den > 0:
            wh_f = float(((SSR_ols - SSR_aug) / m) / (SSR_aug / df_den))
            wh_f = max(wh_f, 0.0)
            wh_pval = float(1 - sp_stats.f.cdf(wh_f, m, df_den))
        else:
            wh_f = 0.0
            wh_pval = 1.0

        return durbin_chi2, durbin_pval, wh_f, wh_pval

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for IV estimation")
