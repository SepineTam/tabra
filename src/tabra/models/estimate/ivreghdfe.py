#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ivreghdfe.py

import numpy as np
from scipy import stats as sp_stats
from tabra.models.estimate.base import BaseModel
from tabra.models.estimate.reghdfe import (
    _remove_singletons, _map_partial_out, _compute_df_a,
    _compute_df_a_nested, pd_factorize,
)
from tabra.results.iv_result import IVResult


class IVRegHDFEModel(BaseModel):
    """IV regression with high-dimensional fixed effects."""

    def fit(self, df, y, exog, endog, instruments, absorb,
            estimator="2sls", vce="unadjusted", cluster=None,
            is_con=True, tolerance=1e-8, max_iter=10000):
        if len(instruments) < len(endog):
            raise ValueError(
                f"Underidentified: {len(endog)} endogenous but "
                f"only {len(instruments)} instruments."
            )

        exog = list(exog)
        endog = list(endog)
        instruments = list(instruments)
        absorb = list(absorb)

        # Deduplicate columns to avoid duplicate column name issues
        all_cols = [y] + exog + endog + instruments + absorb
        if cluster is not None:
            cluster_list = cluster if isinstance(cluster, list) else [cluster]
            for c in cluster_list:
                if c not in all_cols:
                    all_cols.append(c)
        df = df[all_cols].dropna().reset_index(drop=True)

        y_vec = df[y].values.astype(float)
        X1 = df[exog].values.astype(float) if exog else np.empty((len(df), 0))
        X2 = df[endog].values.astype(float)
        Z_inst = df[instruments].values.astype(float)

        n_orig = len(df)

        # Build FE arrays
        fe_arrays = []
        for col in absorb:
            arr = df[col].values
            if arr.dtype.kind not in ('i', 'u'):
                arr = pd_factorize(arr)
            fe_arrays.append(arr)

        # Singleton removal
        mask = _remove_singletons(fe_arrays, n_orig)
        y_vec = y_vec[mask]
        X1 = X1[mask] if X1.shape[1] > 0 else X1
        X2 = X2[mask]
        Z_inst = Z_inst[mask]
        fe_arrays = [arr[mask] for arr in fe_arrays]

        # Store cluster arrays for later VCE
        cluster_arrays = None
        if cluster is not None:
            cluster_list = cluster if isinstance(cluster, list) else [cluster]
            retained_idx = np.where(mask)[0]
            cluster_arrays = []
            for col in cluster_list:
                arr = df[col].values[retained_idx]
                if arr.dtype.kind not in ('i', 'u'):
                    arr = pd_factorize(arr)
                cluster_arrays.append(arr)

        n = len(y_vec)
        n_exog = X1.shape[1]
        n_endog = X2.shape[1]
        N_hdfe = len(fe_arrays)

        # MAP partial-out: demean y, X1, X2, Z_inst
        if len(fe_arrays) > 0:
            all_vars = np.column_stack([X1, X2, Z_inst]) if n_exog > 0 \
                else np.column_stack([X2, Z_inst])

            y_tilde, all_tilde = _map_partial_out(
                y_vec, all_vars, fe_arrays,
                tolerance=tolerance, max_iter=max_iter
            )

            col_offset = 0
            X1_tilde = all_tilde[:, col_offset:col_offset + n_exog] if n_exog > 0 \
                else np.empty((n, 0))
            col_offset += n_exog
            X2_tilde = all_tilde[:, col_offset:col_offset + n_endog]
            col_offset += n_endog
            Z_tilde = all_tilde[:, col_offset:]
        else:
            y_tilde = y_vec
            X1_tilde = X1
            X2_tilde = X2
            Z_tilde = Z_inst

        # Build Z_full (all exogenous + instruments) and X_full (endog + exog)
        Z_full = np.column_stack([X1_tilde, Z_tilde]) if n_exog > 0 \
            else Z_tilde.copy()
        X_full = np.column_stack([X2_tilde, X1_tilde]) if n_exog > 0 \
            else X2_tilde.copy()

        k_z = Z_full.shape[1]
        k = X_full.shape[1]

        ZtZ_inv = np.linalg.inv(Z_full.T @ Z_full)
        PZ = Z_full @ ZtZ_inv @ Z_full.T
        MZ = np.eye(n) - PZ

        # 2SLS on transformed data
        X_hat = PZ @ X_full
        XtX_hat_inv = np.linalg.inv(X_hat.T @ X_hat)
        beta = XtX_hat_inv @ X_hat.T @ y_tilde
        resid = y_tilde - X_full @ beta

        # Degrees of freedom
        if cluster is not None and vce == "cluster":
            df_a = _compute_df_a_nested(fe_arrays, cluster_list, absorb)
        else:
            df_a = _compute_df_a(fe_arrays, cluster, vce)

        df_r = n - k - df_a
        if df_r <= 0:
            df_r = n - k
        df_m = k

        # For cluster VCE, test df_r = N_g - 1
        df_r_test = df_r
        N_g = None
        if vce == "cluster" and cluster_arrays is not None:
            N_g = len(np.unique(cluster_arrays[0]))
            df_r_test = N_g - 1

        # VCE
        if vce == "unadjusted":
            sigma2 = float(resid @ resid) / df_r
            var_beta = sigma2 * XtX_hat_inv
        elif vce == "robust":
            e2 = resid ** 2
            meat = (X_hat.T * e2) @ X_hat
            hc1_adj = n / df_r  # small-sample correction
            var_beta = hc1_adj * XtX_hat_inv @ meat @ XtX_hat_inv
        elif vce == "cluster":
            if cluster_arrays is None:
                raise ValueError("cluster vce requires cluster variable names")
            clust_arr = cluster_arrays[0]
            unique_clusts = np.unique(clust_arr)
            meat = np.zeros((k, k))
            for c in unique_clusts:
                idx = clust_arr == c
                Xc = X_hat[idx]
                ec = resid[idx]
                meat += Xc.T @ np.outer(ec, ec) @ Xc
            N_g = len(unique_clusts)
            small_adj = (N_g / (N_g - 1)) * ((n - 1) / df_r)
            var_beta = small_adj * XtX_hat_inv @ meat @ XtX_hat_inv

        std_err = np.sqrt(np.maximum(np.diag(var_beta), 0))
        z_stat = beta / std_err
        p_value = 2 * (1 - sp_stats.t.cdf(np.abs(z_stat), df_r_test))

        # R-squared and RMSE using correct df_r
        SST = float(y_tilde @ y_tilde)
        SSR = float(resid @ resid)
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_r \
            if df_r > 0 else 0.0
        root_mse = np.sqrt(SSR / df_r) if df_r > 0 else 0.0

        # Model F-statistic
        F_stat = None
        try:
            F_stat = float(
                (beta.T @ np.linalg.solve(var_beta, beta)) / df_m
            ) if df_m > 0 else 0.0
        except Exception:
            pass

        var_names = endog + exog

        # Weak identification and underidentification tests
        # For unadjusted VCE: Cragg-Donald F + Anderson LM
        # For robust/cluster VCE: return None (Kleibergen-Paap not yet implemented)
        widstat = None
        idstat = None
        idpval = None
        if vce == "unadjusted":
            try:
                # Partial F: regress endogenous on excluded instruments only,
                # after partialling out exogenous variables
                if n_exog > 0:
                    Q1 = X1_tilde @ np.linalg.inv(
                        X1_tilde.T @ X1_tilde
                    ) @ X1_tilde.T
                    X2_resid = X2_tilde - Q1 @ X2_tilde
                    Z_excl_resid = Z_tilde - Q1 @ Z_tilde
                else:
                    X2_resid = X2_tilde
                    Z_excl_resid = Z_tilde

                L_excl = Z_excl_resid.shape[1]
                K_endog = X2_resid.shape[1]

                if n_endog == 1:
                    beta_j = np.linalg.solve(
                        Z_excl_resid.T @ Z_excl_resid,
                        Z_excl_resid.T @ X2_resid[:, 0]
                    )
                    resid_j = X2_resid[:, 0] - Z_excl_resid @ beta_j
                    SSR_j = float(resid_j @ resid_j)
                    SST_j = float(X2_resid[:, 0] @ X2_resid[:, 0])
                    SSE_j = SST_j - SSR_j
                    df_num = L_excl
                    df_denom = df_r
                    if df_num > 0 and df_denom > 0:
                        widstat = (SSE_j / df_num) / (SSR_j / df_denom)
                    else:
                        widstat = 0.0
                else:
                    widstat = self._cragg_donald(
                        X2_resid, Z_excl_resid, df_r, L_excl, K_endog
                    )

                # Anderson LM
                L_excl_id = Z_excl_resid.shape[1]
                K_endog_id = X2_resid.shape[1]
                min_dim = min(L_excl_id, K_endog_id)
                if min_dim > 0:
                    A = X2_resid.T @ Z_excl_resid
                    B_z = Z_excl_resid.T @ Z_excl_resid
                    B_x = X2_resid.T @ X2_resid
                    B_z_inv = np.linalg.inv(B_z)
                    B_x_inv = np.linalg.inv(B_x)
                    M = B_x_inv @ A @ B_z_inv @ A.T
                    eigenvalues = np.linalg.eigvalsh(M)
                    min_eig = float(np.min(np.maximum(eigenvalues, 0)))
                    idstat = n * min_eig
                    df_id = L_excl_id - K_endog_id + 1 if L_excl_id >= K_endog_id \
                        else K_endog_id - L_excl_id + 1
                    if df_id > 0:
                        idpval = float(1 - sp_stats.chi2.cdf(idstat, df_id))
                    else:
                        idpval = None
            except Exception:
                pass

        # Overidentification J test
        L = Z_inst.shape[1]
        K_endog = n_endog
        if L > K_endog:
            sigma2_hat = SSR / n
            j_stat = float(resid @ PZ @ resid / sigma2_hat)
            j_df = L - K_endog
            j_pval = float(1 - sp_stats.chi2.cdf(j_stat, j_df))
        else:
            j_stat = None
            j_pval = None

        vce_label = vce
        if vce == "cluster" and cluster is not None:
            cluster_list = cluster if isinstance(cluster, list) else [cluster]
            vce_label = f"cluster({', '.join(cluster_list)})"

        return IVResult(
            coef=beta, std_err=std_err, z_stat=z_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            root_mse=root_mse,
            n_obs=n, k_vars=k, df_m=df_m, df_r=df_r, df_a=df_a,
            F=F_stat, N_hdfe=N_hdfe,
            first_stage_f=None,
            j_stat=j_stat, j_pval=j_pval,
            endog_test_stat=None, endog_test_pval=None,
            var_names=var_names, y_name=y,
            estimator=estimator, vce_type=vce_label,
            endog_names=endog, exog_names=exog,
            inst_names=instruments,
            idstat=idstat, idpval=idpval,
            widstat=widstat,
        )

    @staticmethod
    def _cragg_donald(X2_resid, Z_excl_resid, df_r, L_excl, K_endog):
        """Compute Cragg-Donald Wald F for multiple endogenous variables."""
        A = X2_resid.T @ Z_excl_resid
        B_z = Z_excl_resid.T @ Z_excl_resid
        B_x = X2_resid.T @ X2_resid
        B_z_inv = np.linalg.inv(B_z)
        B_x_inv = np.linalg.inv(B_x)
        M = B_x_inv @ A @ B_z_inv @ A.T
        eigenvalues = np.linalg.eigvalsh(M)
        min_eig = float(np.min(np.maximum(eigenvalues, 0)))
        df_num = L_excl - K_endog + 1 if L_excl >= K_endog else 1
        if df_num > 0 and df_r > 0:
            return df_r * min_eig / df_num
        return 0.0

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for IV HDFE estimation")
