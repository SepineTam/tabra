#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : ols.py

# Scientific support: EST-OLS in SCIENTIFIC_SUPPORT.csv.

import numpy as np
from tabra.models.estimate.base import BaseModel
from tabra.models.estimate.stats import EstimationStats
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
        design = df[x].values.astype(float)
        var_names = list(x)
        if is_con:
            design = np.column_stack([design, np.ones(design.shape[0])])
            var_names = var_names + ["_cons"]
        n, k = design.shape

        if n == 0:
            raise ValueError("OLS requires at least one complete observation")
        if not np.isfinite(y_vec).all() or not np.isfinite(design).all():
            raise ValueError("OLS inputs must contain only finite values")
        if n - k <= 0:
            raise ValueError(
                f"OLS requires positive residual degrees of freedom: n={n}, k={k}"
            )

        column_scale = np.max(np.abs(design), axis=0)
        safe_column_scale = np.where(column_scale > 0, column_scale, 1.0)
        scaled_design = design / safe_column_scale

        try:
            scaled_beta, _, rank, singular_values = np.linalg.lstsq(
                scaled_design, y_vec, rcond=None
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("OLS stable least-squares solve failed") from exc
        if rank < k:
            raise ValueError(f"OLS design is rank deficient: rank={rank}, k={k}")

        beta = scaled_beta / safe_column_scale.reshape(-1, 1)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                fitted = scaled_design @ scaled_beta
                resid = y_vec - fitted
                ss_residual = float((resid.T @ resid).item())
                if is_con:
                    y_mean = float(np.mean(y_vec))
                    centered_y = y_vec - y_mean
                    ss_total = float((centered_y.T @ centered_y).item())
                else:
                    ss_total = float((y_vec.T @ y_vec).item())
                raw_ss_model = ss_total - ss_residual
                ss_roundoff = (
                    np.finfo(float).eps
                    * max(n, k)
                    * max(ss_total, ss_residual, np.finfo(float).tiny)
                )
                if raw_ss_model < -ss_roundoff:
                    raise ValueError("OLS model sum of squares is negative")
                ss_model = max(raw_ss_model, 0.0)
                df_model = self._model_df(k, is_con=is_con)
                df_resid = self._resid_df(n, k)
                sigma2 = ss_residual / df_resid
                _, _, right_singular_vectors = np.linalg.svd(
                    scaled_design, full_matrices=False
                )
                covariance_factor = (
                    right_singular_vectors.T / singular_values
                ) / safe_column_scale.reshape(-1, 1)
                covariance_scale = np.max(np.abs(covariance_factor), axis=1)
                normalized_covariance = (
                    covariance_factor / covariance_scale.reshape(-1, 1)
                )
                std_err = (
                    np.sqrt(sigma2)
                    * covariance_scale
                    * np.sqrt(np.sum(normalized_covariance ** 2, axis=1))
                )
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("OLS estimation produced non-finite intermediate values") from exc

        core_arrays = (beta, fitted, resid, singular_values, std_err)
        core_scalars = (ss_residual, ss_model, ss_total, sigma2)
        if not all(np.isfinite(values).all() for values in core_arrays):
            raise ValueError("OLS estimation produced non-finite results")
        if not np.isfinite(core_scalars).all():
            raise ValueError("OLS estimation produced non-finite sums of squares")

        fit_scale = max(
            float(np.max(np.abs(y_vec))),
            float(np.max(np.abs(fitted))),
            np.finfo(float).tiny,
        )
        residual_tolerance = (
            np.finfo(float).eps
            * max(n, k, 1)
            * fit_scale
        )
        inference_is_degenerate = bool(
            sigma2 <= 0
            or np.max(np.abs(resid)) <= residual_tolerance
            or np.any(std_err <= 0)
        )
        if inference_is_degenerate:
            t_stat = np.full(k, np.nan)
            p_value = np.full(k, np.nan)
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                t_stat = beta.flatten() / std_err
            if not np.isfinite(t_stat).all():
                raise ValueError("OLS test statistics are non-finite")
            p_value = np.array([t_pval(t, df_resid) for t in t_stat])
            if not np.isfinite(p_value).all():
                raise ValueError("OLS p-values are non-finite")

        r_squared = EstimationStats.r_squared(ss_residual, ss_total)
        r_squared_adj = self._adjusted_r_squared(r_squared, n, k, is_con=is_con)
        if inference_is_degenerate:
            f_stat, f_pval_val = np.nan, np.nan
        else:
            f_stat, f_pval_val = self._f_statistics(
                ss_model, ss_residual, df_model, df_resid
            )
        mse = EstimationStats.mse(ss_residual, df_resid)
        root_mse = EstimationStats.root_mse(ss_residual, df_resid)

        return OLSResult(
            coef=beta.flatten(), std_err=std_err, t_stat=t_stat, p_value=p_value,
            r_squared=r_squared, r_squared_adj=r_squared_adj, f_stat=f_stat, f_pval=f_pval_val,
            resid=resid.flatten(), fitted=fitted.flatten(), n_obs=n, k_vars=k,
            var_names=var_names, SSR=ss_residual, SSE=ss_model, SST=ss_total,
            df_model=df_model, df_resid=df_resid, mse=mse, root_mse=root_mse,
            y_name=y, is_con=is_con,
        )

    def estimate(self, df, x, **kwargs):
        raise NotImplementedError("Use fit() for OLS estimation")
