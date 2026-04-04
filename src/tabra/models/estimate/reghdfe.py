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
        """Fit a linear model with high-dimensional fixed effects via HDFE.

        Args:
            df (pd.DataFrame): Input dataset.
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            absorb (list[str]): Variables to absorb as fixed effects.
            vce (str): Variance-covariance estimator type. One of "unadjusted",
                "robust", "cluster". Default "unadjusted".
            cluster (list[str]): Cluster variable names. Required when vce="cluster".
            tolerance (float): MAP convergence tolerance. Default 1e-8.
            max_iter (int): Maximum MAP iterations. Default 10000.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            RegHDFEResult: Estimation result.

        Example:
            >>> dta = load_data("nlswork")
            >>> result = RegHDFE().fit(dta._df, "ln_wage", ["age", "tenure"], absorb=["idcode", "year"])
        """
        x_cols = list(x)
        absorb_cols = list(absorb)
        df_clean = self._prepare_df(df, y, x, extra_cols=absorb_cols).reset_index(drop=True)

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

        # --- Collinearity detection (Bug 3) ---
        # After MAP, columns with near-zero variance are absorbed by FE
        col_var = np.var(X_tilde, axis=0)
        non_collinear = col_var > tolerance
        omitted_idx = np.where(~non_collinear)[0]
        active_idx = np.where(non_collinear)[0]

        x_cols_active = [x_cols[i] for i in active_idx]
        X_tilde_active = X_tilde[:, active_idx]
        X_active = X[:, active_idx]
        k_active = len(active_idx)

        # --- OLS on transformed data, NO constant column (Bug 1) ---
        XtX = mat_mul(mat_transpose(X_tilde_active), X_tilde_active)
        Xty = mat_mul(mat_transpose(X_tilde_active), y_tilde.reshape(-1, 1))
        XtX_inv = mat_inv(XtX)
        beta_slope = mat_mul(XtX_inv, Xty).flatten()

        # --- Recover constant term (Bug 1) ---
        y_mean = float(np.mean(y_vec))
        x_means = np.mean(X_active, axis=0)
        b_cons = y_mean - x_means @ beta_slope

        # Residuals in the demeaned space (FE already absorbed)
        resid = y_tilde - X_tilde_active @ beta_slope
        SSR = float(resid @ resid)

        # Full coefficient vector: active slopes + omitted (0) + constant
        beta = np.zeros(k + 1)
        for j_pos, j_orig in enumerate(active_idx):
            beta[j_orig] = beta_slope[j_pos]
        beta[k] = b_cons  # constant is last
        var_names = x_cols + ["_cons"]

        # DoF adjustment (Bug 2: first FE uses K not K-1)
        df_a = _compute_df_a(fe_arrays, cluster, vce, fe_names=fe_names)
        k_full = k + 1  # all original x cols + constant
        df_model = k_active  # active slopes only
        df_resid = n - df_a - df_model  # df_a already absorbs constant DoF
        # For cluster VCE, t/F tests use G-1 as denominator df
        df_r_test = df_resid
        # VCE adjustment df: for cluster with nested FE, use nested df_a
        df_resid_vce = df_resid
        if vce == "cluster" and cluster is not None:
            cluster_cols = cluster if isinstance(cluster, list) else [cluster]
            retained_idx = np.where(mask)[0]
            _clust_arr = df_clean[cluster_cols[0]].values[retained_idx]
            if _clust_arr.dtype.kind not in ('i', 'u'):
                _clust_arr = pd_factorize(_clust_arr)
            df_r_test = len(np.unique(_clust_arr)) - 1
            # Compute nested df_a for VCE adjustment
            df_a_nested = _compute_df_a_nested(fe_arrays, cluster_cols, fe_names)
            df_resid_vce = n - df_a_nested - df_model

        # --- Standard errors (Bug 1: VCE on demeaned data without constant) ---
        if vce == "unadjusted":
            sigma2 = SSR / df_resid
            var_slope = sigma2 * XtX_inv
            se_cons = np.sqrt(sigma2 * (1.0 / n + x_means @ XtX_inv @ x_means))
        elif vce == "robust":
            var_slope = _robust_vce(X_tilde_active, resid, n, k_active, XtX_inv, df_resid=df_resid)
            # HC1 constant SE via sandwich
            se_cons = _robust_cons_se(X_tilde_active, resid, n, k_active, XtX_inv, x_means, df_resid=df_resid)
        elif vce == "cluster":
            if cluster is None:
                raise ValueError("cluster requires cluster variable names")
            var_slope = _cluster_vce(
                X_tilde_active, resid, n, k_active, XtX_inv,
                df_clean, mask, cluster, df_resid=df_resid_vce
            )
            se_cons = _cluster_cons_se(
                X_tilde_active, resid, n, k_active, XtX_inv,
                x_means, df_clean, mask, cluster, df_resid=df_resid_vce
            )
        else:
            raise ValueError(f"Unknown vce type: {vce}")

        std_err_slope = np.sqrt(np.diag(np.abs(var_slope)))
        # Build full std_err: active + omitted (0) + constant
        std_err = np.zeros(k + 1)
        for j_pos, j_orig in enumerate(active_idx):
            std_err[j_orig] = std_err_slope[j_pos]
        std_err[k] = se_cons

        t_stat = beta / std_err
        # Avoid 0/0 for omitted vars
        t_stat = np.where(std_err > 0, t_stat, 0.0)
        p_value = np.array([t_pval(t, df_r_test) for t in t_stat])

        # R-squared
        SST_within = float(y_tilde @ y_tilde)
        SSE_within = SST_within - SSR
        r2_within = 1 - SSR / SST_within if SST_within > 0 else 0.0

        SST = float((y_vec - y_mean) @ (y_vec - y_mean))
        SSE = SST - SSR
        r_squared = 1 - SSR / SST if SST > 0 else 0.0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_resid if df_resid > 0 else 0.0
        r2_a_within = 1 - (1 - r2_within) * (n - 1) / df_resid if df_resid > 0 else 0.0

        # F stat
        if vce == "unadjusted":
            f_stat = (SSE_within / df_model) / (SSR / df_resid) if df_model > 0 and df_resid > 0 else 0.0
            f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0
        else:
            # Wald F: (1/q) * beta_slope' V^(-1) beta_slope
            if df_model > 0:
                V_slope_inv = mat_inv(var_slope)
                f_stat = float(beta_slope @ V_slope_inv @ beta_slope) / df_model
                f_pval_val = f_pval(f_stat, df_model, df_r_test)
            else:
                f_stat = 0.0
                f_pval_val = 0.0

        # Log-likelihood (Stata uses SSR/N, not SSR/df_r)
        mse = SSR / df_resid
        root_mse = np.sqrt(mse)
        ll = -0.5 * n * (1 + np.log(2 * np.pi) + np.log(SSR / n))

        from tabra.results.reghdfe_result import RegHDFEResult
        return RegHDFEResult(
            coef=beta, std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj,
            f_stat=f_stat, f_pval=f_pval_val,
            resid=resid, fitted=X_active @ beta_slope + b_cons,
            n_obs=n, k_vars=k_full, var_names=var_names,
            SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_r_test,
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


def _compute_df_a(fe_arrays, cluster=None, vce=None, fe_names=None):
    """Compute degrees of freedom absorbed by fixed effects.

    For single FE: K (reghdfe counts all categories).
    For two FE: uses connected components of bipartite graph.
    For >2 FE: uses pairwise approximation.

    Note: cluster nesting does NOT affect df_a for model statistics
    (R², MSE, etc.). It only affects test df via df_r_test.
    """
    n_fe = len(fe_arrays)
    if n_fe == 0:
        return 0

    K = [len(np.unique(arr)) for arr in fe_arrays]

    if n_fe == 1:
        return K[0]

    if n_fe == 2:
        M = _count_connected_components(fe_arrays[0], fe_arrays[1])
        return K[0] + K[1] - M

    # Multi-way: pairwise approximation
    df_a = K[0]
    for j in range(1, n_fe):
        M_max = 1
        for i in range(j):
            M_ij = _count_connected_components(fe_arrays[i], fe_arrays[j])
            M_max = max(M_max, M_ij)
        df_a += K[j] - M_max

    return df_a


def _compute_df_a_nested(fe_arrays, cluster_cols, fe_names):
    """Compute df_a with cluster nesting: nested FEs contribute 0 to df_a."""
    n_fe = len(fe_arrays)
    if n_fe == 0:
        return 0

    cluster_set = set(cluster_cols if isinstance(cluster_cols, list) else [cluster_cols])
    nested = [fe_names[i] in cluster_set for i in range(n_fe)]
    K = [len(np.unique(arr)) for arr in fe_arrays]

    if n_fe == 1:
        return 0 if nested[0] else K[0]

    if n_fe == 2:
        M = _count_connected_components(fe_arrays[0], fe_arrays[1])
        df_a = 0
        df_a += 0 if nested[0] else K[0]
        df_a += 0 if nested[1] else K[1] - M
        return df_a

    df_a = 0
    df_a += 0 if nested[0] else K[0]
    for j in range(1, n_fe):
        if nested[j]:
            continue
        M_max = 1
        for i in range(j):
            if nested[i]:
                continue
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


def _robust_vce(X, resid, n, k, XtX_inv, df_resid=None):
    """HC1 robust variance-covariance estimator.

    Args:
        X: Demeaned design matrix (no constant column).
        resid: Residuals from demeaned OLS.
        n: Number of observations.
        k: Number of slope parameters.
        XtX_inv: Inverse of X'X.
        df_resid: Residual degrees of freedom. If provided, uses N/df_resid
            as the HC1 correction (reghdfe convention). Otherwise uses N/(N-k).
    """
    e2 = resid ** 2
    meat = (X.T * e2) @ X
    adj = n / df_resid if df_resid is not None else n / (n - k)
    return adj * XtX_inv @ meat @ XtX_inv


def _robust_cons_se(X, resid, n, k, XtX_inv, x_means, df_resid=None):
    """HC1 robust SE for the recovered constant term."""
    e2 = resid ** 2
    meat = (X.T * e2) @ X
    adj = n / df_resid if df_resid is not None else n / (n - k)
    s1 = X.T @ e2
    sum_e2 = float(np.sum(e2))
    var_cons = adj * (sum_e2 / (n * n)
                      - 2.0 * x_means @ XtX_inv @ s1 / n
                      + x_means @ XtX_inv @ meat @ XtX_inv @ x_means)
    return np.sqrt(max(var_cons, 0.0))


def _cluster_cons_se(X, resid, n, k, XtX_inv, x_means, df_clean, mask, cluster_vars, df_resid=None):
    """Cluster-robust SE for the recovered constant term."""
    cluster_cols = cluster_vars if isinstance(cluster_vars, list) else [cluster_vars]
    retained_idx = np.where(mask)[0]
    cluster_arrays = []
    for col in cluster_cols:
        arr = df_clean[col].values[retained_idx]
        if arr.dtype.kind not in ('i', 'u'):
            arr = pd_factorize(arr)
        cluster_arrays.append(arr)

    if len(cluster_cols) == 1:
        clust_arr = cluster_arrays[0]
        unique_clusts = np.unique(clust_arr)
        meat_slope = np.zeros((k, k))
        meat_aug11 = 0.0  # sum_g (sum e_g)^2
        meat_aug1k = np.zeros(k)  # sum_g (sum e_g) * (X_g'e_g)
        for c in unique_clusts:
            idx = clust_arr == c
            Xc = X[idx]
            ec = resid[idx]
            g = Xc.T @ ec  # k-vector
            se = float(np.sum(ec))
            meat_slope += np.outer(g, g)
            meat_aug11 += se * se
            meat_aug1k += se * g
        N_g = len(unique_clusts)
        adj = (N_g / (N_g - 1)) * ((n - 1) / df_resid) if df_resid is not None else (N_g / (N_g - 1)) * ((n - 1) / (n - k))
        var_cons = adj * (meat_aug11 / (n * n)
                          - 2.0 * x_means @ XtX_inv @ meat_aug1k / n
                          + x_means @ XtX_inv @ meat_slope @ XtX_inv @ x_means)
        return np.sqrt(max(var_cons, 0.0))
    else:
        # Multi-way: use the slope VCE and approximate
        var_slope = _cluster_vce(X, resid, n, k, XtX_inv, df_clean, mask, cluster_vars, df_resid=df_resid)
        sigma2 = float(resid @ resid) / df_resid if df_resid is not None else float(resid @ resid) / (n - k - 1)
        var_cons = sigma2 / n + x_means @ var_slope @ x_means
        return np.sqrt(max(var_cons, 0.0))


def _cluster_vce(X, resid, n, k, XtX_inv, df_clean, mask, cluster_vars, df_resid=None):
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
        N_g = len(unique_clusts)
        adj = (N_g / (N_g - 1)) * ((n - 1) / df_resid) if df_resid is not None else (N_g / (N_g - 1)) * ((n - 1) / (n - k))
        return adj * XtX_inv @ meat @ XtX_inv

    else:
        # Multi-way clustering (Cameron, Gelbach, Miller 2011)
        from itertools import combinations

        n_clust = len(cluster_cols)
        V_total = np.zeros((k, k))

        for r in range(1, n_clust + 1):
            for combo in combinations(range(n_clust), r):
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
                    adj = (N_g / (N_g - 1)) * ((n - 1) / df_resid) if df_resid is not None else (N_g / (N_g - 1)) * ((n - 1) / (n - k))
                else:
                    adj = 1.0
                V_combo = adj * XtX_inv @ meat @ XtX_inv
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

    n_fe = len(fe_arrays)
    for i, (arr, name) in enumerate(zip(fe_arrays, fe_names)):
        n_cats = len(np.unique(arr))
        is_nested = name in cluster_cols

        if is_nested:
            redundant = n_cats
            num_coefs = 0
        elif i == 0:
            # First FE: no redundant (reghdfe counts all K categories)
            redundant = 0
            num_coefs = n_cats
        else:
            # Subsequent FE: connected components with previous
            M = _count_connected_components(fe_arrays[0], arr)
            redundant = M
            num_coefs = n_cats - M

        info.append({
            "name": name,
            "categories": n_cats,
            "redundant": redundant,
            "num_coefs": num_coefs,
            "nested": is_nested,
        })
    return info
