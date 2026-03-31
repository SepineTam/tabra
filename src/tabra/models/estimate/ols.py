#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ols.py

import numpy as np
from tabra.models.estimate.base import BaseModel
from tabra.ops.linalg import mat_mul, mat_transpose, mat_inv
from tabra.ops.stats import t_pval, f_pval
from tabra.results.ols_result import OLSResult


class OLS(BaseModel):
    def fit(self, df, y, x, is_con=True):
        y_vec = df[y].values.reshape(-1, 1).astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        if is_con:
            X = np.column_stack([np.ones(X.shape[0]), X])
            var_names = ["_cons"] + var_names
        n, k = X.shape

        XtX = mat_mul(mat_transpose(X), X)
        Xty = mat_mul(mat_transpose(X), y_vec)
        XtX_inv = mat_inv(XtX)

        beta = mat_mul(XtX_inv, Xty)
        fitted = mat_mul(X, beta)
        resid = y_vec - fitted

        sigma2 = float(mat_mul(mat_transpose(resid), resid)[0, 0]) / (n - k)
        var_beta = sigma2 * XtX_inv
        std_err = np.sqrt(np.diag(var_beta))
        t_stat = beta.flatten() / std_err
        p_value = np.array([t_pval(t, n - k) for t in t_stat])

        SSR = float(mat_mul(mat_transpose(resid), resid)[0, 0])
        y_mean = float(np.mean(y_vec))
        SST = float(mat_mul(mat_transpose(y_vec - y_mean), y_vec - y_mean)[0, 0])
        SSE = SST - SSR

        r_squared = 1 - SSR / SST
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k)

        df_model = k - 1 if is_con else k
        df_resid = n - k
        f_stat = (SSE / df_model) / (SSR / df_resid) if df_model > 0 else 0.0
        f_pval_val = f_pval(f_stat, df_model, df_resid) if df_model > 0 else 0.0

        return OLSResult(
            coef=beta.flatten(), std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj, f_stat=f_stat, f_pval=f_pval_val,
            resid=resid.flatten(), fitted=fitted.flatten(), n_obs=n, k_vars=k,
            var_names=var_names, SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_resid, mse=SSR / (n - k), root_mse=np.sqrt(SSR / (n - k)),
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for OLS estimation")
