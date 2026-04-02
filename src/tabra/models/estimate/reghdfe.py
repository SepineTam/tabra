#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : reghdfe.py

import numpy as np
from tabra.models.estimate.base import BaseModel
from tabra.ops.linalg import mat_mul, mat_transpose, mat_inv
from tabra.ops.stats import t_pval, f_pval


class RegHDFE(BaseModel):

    def fit(self, df, y, x, absorb, vce="unadjusted", cluster=None,
            tolerance=1e-8, max_iter=10000, is_con=True):
        x_cols = list(x)
        absorb_cols = list(absorb)
        all_cols = [y] + x_cols + absorb_cols

        # Drop rows with missing values
        df_clean = df[all_cols].dropna().reset_index(drop=True)

        y_vec = df_clean[y].values.astype(float)
        X = df_clean[x_cols].values.astype(float)
        n_orig = len(df_clean)

        # Build FE arrays
        fe_arrays = []
        fe_names = []
        for col in absorb_cols:
            arr = df_clean[col].values
            # Convert to integer codes if not already
            if arr.dtype.kind not in ('i', 'u'):
                arr = pd_factorize(arr)
            fe_arrays.append(arr)
            fe_names.append(col)

        # Singleton removal
        mask = _remove_singletons(fe_arrays, n_orig)
        y_vec = y_vec[mask]
        X = X[mask]
        fe_arrays = [arr[mask] for arr in fe_arrays]
        n = len(y_vec)
        k = X.shape[1]

        # MAP: partial out fixed effects
        y_tilde, X_tilde = _map_partial_out(
            y_vec, X, fe_arrays, tolerance=tolerance, max_iter=max_iter
        )

        # OLS on transformed data
        if is_con:
            X_tilde_full = np.column_stack([X_tilde, np.ones(n)])
            var_names = x_cols + ["_cons"]
        else:
            X_tilde_full = X_tilde
            var_names = x_cols

        k_full = X_tilde_full.shape[1]
        XtX = mat_mul(mat_transpose(X_tilde_full), X_tilde_full)
        Xty = mat_mul(mat_transpose(X_tilde_full), y_tilde.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta = mat_mul(XtX_inv, Xty).flatten()

        resid = y_tilde - X_tilde_full @ beta
        SSR = float(resid @ resid)

        # DoF adjustment
        df_a = _compute_df_a(fe_arrays, cluster, vce)
        df_model = k_full - 1 if is_con else k_full
        df_resid = n - k_full - df_a

        # Standard errors
        if vce == "unadjusted":
            sigma2 = SSR / df_resid
            var_beta = sigma2 * XtX_inv
        elif vce == "robust":
            var_beta = _robust_vce(X_tilde_full, resid, n, k_full, XtX_inv)
        elif vce == "cluster":
            if cluster is None:
                raise ValueError("cluster requires cluster variable names")
            var_beta = _cluster_vce(
                X_tilde_full, resid, n, k_full, XtX_inv,
                df_clean, mask, cluster
            )
        else:
            raise ValueError(f"Unknown vce type: {vce}")

        std_err = np.sqrt(np.diag(np.abs(var_beta)))
        t_stat = beta / std_err
        p_value = np.array([t_pval(t, df_resid) for t in t_stat])

        # R-squared
        # Total SS: on demeaned y (within R2)
        SST_within = float(y_tilde @ y_tilde)
        SSE_within = SST_within - SSR
        r2_within = 1 - SSR / SST_within if SST_within > 0 else 0.0

        # Full R-squared
        y_mean = float(np.mean(y_vec))
        SST = float((y_vec - y_mean) @ (y_vec - y_mean))
        SSE = SST - SSR
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0
        r2_a_within = 1 - (1 - r2_within) * (n - 1) / df_resid if df_resid > 0 else 0.0

        # F stat
        f_stat = (SSE_within / df_model) / (SSR / df_resid) if df_model > 0 and df_resid > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        # Log-likelihood
        mse = SSR / df_resid
        root_mse = np.sqrt(mse)
        ll = -0.5 * n * (np.log(2 * np.pi) + np.log(mse) + 1)

        from tabra.results.reghdfe_result import RegHDFEResult
        return RegHDFEResult(
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid, fitted=X_tilde_full @ beta,
            n_obs=n, k_vars=k_full, var_names=var_names,
            SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_resid,
            mse=mse, root_mse=root_mse,
            y_name=y,
            r2_within=r2_within, r2_a_within=r2_a_within,
            df_a=df_a, n_hdfe=len(fe_arrays),
            absorbed_fe=_build_absorbed_fe_info(fe_arrays, fe_names, cluster, vce),
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for RegHDFE estimation")


def pd_factorize(arr):
    """Factorize a non-integer array to 0-based integer codes."""
    _, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64)


def _remove_singletons(fe_arrays, n):
    """Iteratively remove singleton observations."""
    mask = np.ones(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        for fe_arr in fe_arrays:
            sub = fe_arr[mask]
            counts = np.bincount(sub)
            # Find categories with only 1 obs
            singletons = np.where(counts == 1)[0]
            if len(singletons) > 0:
                singleton_mask = np.isin(sub, singletons)
                # Map back to original indices
                orig_indices = np.where(mask)[0]
                for idx in orig_indices[singleton_mask]:
                    mask[idx] = False
                changed = True
    return mask


def _map_partial_out(y_vec, X, fe_arrays, tolerance=1e-8, max_iter=10000):
    """Method of Alternating Projections to partial out fixed effects.

    Uses Symmetric Kaczmarz transform (forward + backward pass per iteration).
    """
    n, k = X.shape
    n_fe = len(fe_arrays)

    # Precompute group indices for each FE dimension
    group_indices = []
    for fe_arr in fe_arrays:
        cats = np.unique(fe_arr)
        idx_map = {}
        for c in cats:
            idx_map[c] = np.where(fe_arr == c)[0]
        group_indices.append(idx_map)

    # Working copies
    y_work = y_vec.copy()
    X_work = X.copy()

    # Pre-compute group means function
    def _demean(arr, group_indices):
        result = arr.copy()
        for idx_map in group_indices:
            new_result = result.copy()
            for c, idx in idx_map.items():
                new_result[idx] = result[idx] - result[idx].mean()
            result = new_result
        return result

    # MAP iteration with Symmetric Kaczmarz
    converged = False
    for iteration in range(max_iter):
        y_old = y_work.copy()

        # Forward pass
        for dim_idx in range(n_fe):
            idx_map = group_indices[dim_idx]
            for c, idx in idx_map.items():
                y_work[idx] -= y_work[idx].mean()
                for j in range(k):
                    X_work[idx, j] -= X_work[idx, j].mean()

        # Backward pass
        for dim_idx in range(n_fe - 1, -1, -1):
            idx_map = group_indices[dim_idx]
            for c, idx in idx_map.items():
                y_work[idx] -= y_work[idx].mean()
                for j in range(k):
                    X_work[idx, j] -= X_work[idx, j].mean()

        # Check convergence on y
        max_diff = np.max(np.abs(y_work - y_old))
        if max_diff < tolerance:
            converged = True
            break

    return y_work, X_work


def _compute_df_a(fe_arrays, cluster, vce):
    """Compute degrees of freedom absorbed by fixed effects.

    For single FE: K - 1 (one category is base / absorbed by constant).
    For two FE: uses connected components of bipartite graph.
    For >2 FE: uses pairwise approximation.

    If cluster is specified, FEs nested in cluster vars are treated as redundant.
    """
    n_fe = len(fe_arrays)
    if n_fe == 0:
        return 0

    # Category counts per FE dimension
    K = [len(np.unique(arr)) for arr in fe_arrays]

    # Check for FE nested in cluster
    if cluster is not None and vce == "cluster":
        cluster_cols = cluster if isinstance(cluster, list) else [cluster]
        # If a FE variable is also a cluster variable, it's nested
        # We'll handle this via the absorbed_fe_info for display
        pass

    if n_fe == 1:
        # Single FE: K - 1 (one absorbed by intercept)
        return K[0] - 1

    if n_fe == 2:
        # Two-way: use connected components
        M = _count_connected_components(fe_arrays[0], fe_arrays[1])
        df_a = (K[0] - 1) + (K[1] - M)
        return df_a

    # Multi-way: pairwise approximation
    df_a = K[0] - 1  # First FE: K - 1
    M_prev = 1

    for j in range(1, n_fe):
        M_max = 1
        for i in range(j):
            M_ij = _count_connected_components(fe_arrays[i], fe_arrays[j])
            M_max = max(M_max, M_ij)
        df_a += K[j] - M_max

    return df_a


def _count_connected_components(fe1, fe2):
    """Count connected components in bipartite graph between two FE dimensions.

    Uses BFS on the adjacency structure.
    """
    n = len(fe1)
    cats1 = np.unique(fe1)
    cats2 = np.unique(fe2)

    # Map to 0-based indices
    map1 = {c: i for i, c in enumerate(cats1)}
    map2 = {c: i for i, c in enumerate(cats2)}
    n1, n2 = len(cats1), len(cats2)

    # Build adjacency: for each cat1, list of connected cat2
    adj_1_to_2 = [set() for _ in range(n1)]
    adj_2_to_1 = [set() for _ in range(n2)]
    for i in range(n):
        c1 = map1[fe1[i]]
        c2 = map2[fe2[i]]
        adj_1_to_2[c1].add(c2)
        adj_2_to_1[c2].add(c1)

    # BFS to find connected components
    visited_1 = np.zeros(n1, dtype=bool)
    visited_2 = np.zeros(n2, dtype=bool)
    n_components = 0

    for start in range(n1):
        if visited_1[start]:
            continue
        n_components += 1
        # BFS
        queue = [('1', start)]
        visited_1[start] = True
        while queue:
            side, node = queue.pop(0)
            if side == '1':
                for nb in adj_1_to_2[node]:
                    if not visited_2[nb]:
                        visited_2[nb] = True
                        queue.append(('2', nb))
            else:
                for nb in adj_2_to_1[node]:
                    if not visited_1[nb]:
                        visited_1[nb] = True
                        queue.append(('1', nb))

    return n_components


def _robust_vce(X, resid, n, k, XtX_inv):
    """HC1 robust variance-covariance estimator."""
    # Sandwich: (X'X)^-1 X' diag(e^2) X (X'X)^-1 * N/(N-k)
    e2 = resid ** 2
    # Meat: X' diag(e^2) X
    meat = (X.T * e2) @ X
    # HC1 adjustment
    adj = n / (n - k)
    return adj * XtX_inv @ meat @ XtX_inv


def _cluster_vce(X, resid, n, k, XtX_inv, df_clean, mask, cluster_vars):
    """Cluster-robust variance-covariance estimator (Cameron-Gelbach-Miller)."""
    cluster_cols = cluster_vars if isinstance(cluster_vars, list) else [cluster_vars]

    # Get cluster arrays for the retained observations
    retained_idx = np.where(mask)[0]
    cluster_arrays = []
    for col in cluster_cols:
        arr = df_clean[col].values[retained_idx]
        if arr.dtype.kind not in ('i', 'u'):
            arr = pd_factorize(arr)
        cluster_arrays.append(arr)

    if len(cluster_cols) == 1:
        # One-way clustering
        clust_arr = cluster_arrays[0]
        unique_clusts = np.unique(clust_arr)
        meat = np.zeros((k, k))
        for c in unique_clusts:
            idx = clust_arr == c
            Xc = X[idx]
            ec = resid[idx]
            meat += Xc.T @ np.outer(ec, ec) @ Xc
        # Small-sample correction: N_g/(N_g-1) * (N-1)/(N-k)
        N_g = len(unique_clusts)
        adj = (N_g / (N_g - 1)) * ((n - 1) / (n - k))
        return adj * XtX_inv @ meat @ XtX_inv

    else:
        # Multi-way clustering (Cameron, Gelbach, Miller 2011)
        # V = V_12 + V_13 + V_23 - V_1 - V_2 - V_3 + V_0 (for 3-way)
        # General formula using inclusion-exclusion
        from itertools import combinations

        n_clust = len(cluster_cols)
        V_total = np.zeros((k, k))

        for r in range(1, n_clust + 1):
            # Intersect clusters for each combination of r cluster vars
            for combo in combinations(range(n_clust), r):
                # Create intersection key
                intersect_arr = _intersect_clusters(
                    [cluster_arrays[i] for i in combo]
                )
                unique_clusts = np.unique(intersect_arr)
                meat = np.zeros((k, k))
                for c in unique_clusts:
                    idx = intersect_arr == c
                    Xc = X[idx]
                    ec = resid[idx]
                    meat += Xc.T @ np.outer(ec, ec) @ Xc
                N_g = len(unique_clusts)
                if N_g > 1:
                    adj = (N_g / (N_g - 1)) * ((n - 1) / (n - k))
                else:
                    adj = 1.0
                V_combo = adj * XtX_inv @ meat @ XtX_inv

                # Inclusion-exclusion sign
                sign = (-1) ** (n_clust - r)
                V_total += sign * V_combo

        return V_total


def _intersect_clusters(cluster_arrays):
    """Create a single cluster variable from intersection of multiple."""
    if len(cluster_arrays) == 1:
        return cluster_arrays[0]

    # Hash-based intersection
    combined = np.zeros(len(cluster_arrays[0]), dtype=np.int64)
    multiplier = 1
    for arr in cluster_arrays:
        combined += arr * multiplier
        multiplier *= (arr.max() + 1)

    # Re-factorize to compact codes
    _, codes = np.unique(combined, return_inverse=True)
    return codes.astype(np.int64)


def _build_absorbed_fe_info(fe_arrays, fe_names, cluster, vce):
    """Build info dict for each absorbed FE dimension."""
    info = []
    cluster_cols = set()
    if cluster is not None and vce == "cluster":
        cluster_cols = set(cluster if isinstance(cluster, list) else [cluster])

    for i, (arr, name) in enumerate(zip(fe_arrays, fe_names)):
        n_cats = len(np.unique(arr))
        is_nested = name in cluster_cols
        redundant = n_cats if is_nested else 0
        num_coefs = n_cats - redundant if i == 0 else n_cats - 1

        # Adjust for first FE: one more redundant (intercept)
        if i == 0 and not is_nested:
            num_coefs = n_cats - 1

        info.append({
            "name": name,
            "categories": n_cats,
            "redundant": redundant if is_nested else (1 if i == 0 else 0),
            "num_coefs": n_cats - (redundant if is_nested else (1 if i == 0 else 0)),
            "nested": is_nested,
        })
    return info
