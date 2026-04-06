#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ivreghdfe.py

import numpy as np
from tabra.models.estimate.base import BaseModel
from tabra.models.estimate.reghdfe import (
    _remove_singletons, _map_partial_out, _compute_df_a,
    pd_factorize,
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

        all_cols = [y] + exog + endog + instruments + absorb
        if cluster is not None:
            cluster_list = cluster if isinstance(cluster, list) else [cluster]
            all_cols.extend(cluster_list)
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

        # MAP partial-out: demean y, X1, X2, Z_inst
        if len(fe_arrays) > 0:
            # Combine all variables to partial out together
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

        # Now use IVModel logic on transformed data
        from tabra.models.estimate.iv import IVModel
        iv_model = IVModel()

        # Build Z_full and X for IV (no constant — already demeaned)
        # After demeaning, constant is absorbed, so is_con=False
        Z_full = np.column_stack([X1_tilde, Z_tilde]) if n_exog > 0 \
            else Z_tilde.copy()
        X_full = np.column_stack([X2_tilde, X1_tilde]) if n_exog > 0 \
            else X2_tilde.copy()

        k_z = Z_full.shape[1]
        k = X_full.shape[1]

        PZ = Z_full @ np.linalg.inv(Z_full.T @ Z_full) @ Z_full.T

        # 2SLS on transformed data
        X_hat = PZ @ X_full
        XtX_hat_inv = np.linalg.inv(X_hat.T @ X_hat)
        beta = XtX_hat_inv @ X_hat.T @ y_tilde
        resid = y_tilde - X_full @ beta

        # VCE
        if vce == "unadjusted":
            df_a = _compute_df_a(fe_arrays, cluster, vce)
            df_resid = n - k - df_a
            if df_resid <= 0:
                df_resid = n - k
            sigma2 = resid @ resid / df_resid
            var_beta = sigma2 * XtX_hat_inv
        elif vce == "robust":
            e2 = resid ** 2
            meat = (X_hat.T * e2) @ X_hat
            var_beta = XtX_hat_inv @ meat @ XtX_hat_inv
            df_resid = n - k
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
            df_a = _compute_df_a(fe_arrays, cluster, vce)
            df_resid = n - k - df_a
            if df_resid <= 0:
                df_resid = n - k
            adj = (N_g / (N_g - 1)) * ((n - 1) / df_resid)
            var_beta = adj * XtX_hat_inv @ meat @ XtX_hat_inv
            df_resid = n - k  # for simplicity

        std_err = np.sqrt(np.maximum(np.diag(var_beta), 0))
        from scipy import stats as sp_stats
        z_stat = beta / std_err
        p_value = 2 * (1 - sp_stats.norm.cdf(np.abs(z_stat)))

        # R-squared
        SST = float(y_tilde @ y_tilde)
        SSR = float(resid @ resid)
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        df_resid_final = n - k
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid_final if df_resid_final > 0 else 0.0
        root_mse = np.sqrt(SSR / df_resid_final) if df_resid_final > 0 else 0.0

        df_m = k
        df_a = _compute_df_a(fe_arrays, cluster, vce)

        var_names = endog + exog

        # First-stage F
        first_stage_f = None
        try:
            f_stats = []
            ZtZ_inv_local = np.linalg.inv(Z_full.T @ Z_full)
            for j in range(n_endog):
                y_j = X2_tilde[:, j]
                beta_j = ZtZ_inv_local @ Z_full.T @ y_j
                resid_j = y_j - Z_full @ beta_j
                SSR_j = resid_j @ resid_j
                SST_j = y_j @ y_j
                SSE_j = SST_j - SSR_j
                df_model_j = k_z
                df_resid_j = n - k_z
                if df_model_j > 0 and df_resid_j > 0:
                    f_j = (SSE_j / df_model_j) / (SSR_j / df_resid_j)
                else:
                    f_j = 0.0
                f_stats.append(f_j)
            first_stage_f = min(f_stats) if f_stats else 0.0
        except Exception:
            pass

        # Overid J
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
            n_obs=n, k_vars=k, df_m=df_m,
            first_stage_f=first_stage_f,
            j_stat=j_stat, j_pval=j_pval,
            endog_test_stat=None, endog_test_pval=None,
            var_names=var_names, y_name=y,
            estimator=estimator, vce_type=vce_label,
            endog_names=endog, exog_names=exog,
            inst_names=instruments,
            idstat=None, idpval=None,
            widstat=first_stage_f,
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for IV HDFE estimation")
