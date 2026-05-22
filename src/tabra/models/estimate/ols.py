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
from tabra.models.estimate.stats import EstimationStats
from tabra.ops.linalg import mat_mul, mat_transpose, mat_inv
from tabra.ops.stats import t_pval
from tabra.results.ols_result import OLSResult


class OLS(BaseModel):
    def fit(self, df, y, x, is_con=True):
        """Fit an OLS regression model.

        Args:
            df (pd.DataFrame): Input dataset.
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            OLSResult: Estimation result with coefficients, std errors, etc.

        Example:
            >>> dta = load_data("auto")
            >>> result = OLS().fit(dta._df, "price", ["weight", "mpg"])
        """
        df = self._prepare_df(df, y, x)
        y_vec = df[y].values.reshape(-1, 1).astype(float)
        X = df[x].values.astype(float)
        var_names = list(x)
        if is_con:
            X = np.column_stack([X, np.ones(X.shape[0])])
            var_names = var_names + ["_cons"]
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
        if is_con:
            y_mean = float(np.mean(y_vec))
            SST = float(mat_mul(mat_transpose(y_vec - y_mean), y_vec - y_mean)[0, 0])
        else:
            y_mean = 0.0
            SST = float(mat_mul(mat_transpose(y_vec), y_vec)[0, 0])
        SSE = SST - SSR

        r_squared = EstimationStats.r_squared(SSR, SST)
        r_squared_adj = self._adjusted_r_squared(r_squared, n, k, is_con=is_con)
        df_model = self._model_df(k, is_con=is_con)
        df_resid = self._resid_df(n, k)
        f_stat, f_pval_val = self._f_statistics(SSE, SSR, df_model, df_resid)
        mse = EstimationStats.mse(SSR, df_resid)
        root_mse = EstimationStats.root_mse(SSR, df_resid)

        return OLSResult(
            coef=beta.flatten(), std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj, f_stat=f_stat, f_pval=f_pval_val,
            resid=resid.flatten(), fitted=fitted.flatten(), n_obs=n, k_vars=k,
            var_names=var_names, SSR=SSR, SSE=SSE, SST=SST,
            df_model=df_model, df_resid=df_resid, mse=mse, root_mse=root_mse,
            y_name=y,
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for OLS estimation")
