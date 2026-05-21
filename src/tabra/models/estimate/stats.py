#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats as sp_stats

from tabra.ops.stats import f_pval


class EstimationStats:
    """Utility class for reusable estimator statistics."""

    @staticmethod
    def model_df(k_vars, is_con=True):
        return (k_vars - 1) if is_con else k_vars

    @staticmethod
    def resid_df(n_obs, k_vars):
        return n_obs - k_vars

    @classmethod
    def adjusted_r_squared(cls, r_squared, n_obs, k_vars, is_con=True):
        df_resid = cls.resid_df(n_obs, k_vars)
        if df_resid <= 0:
            return 0.0
        adj_n = (n_obs - 1) if is_con else n_obs
        return 1 - (1 - r_squared) * adj_n / df_resid

    @staticmethod
    def pseudo_r_squared(ll, ll_0):
        return 1 - ll / ll_0 if ll_0 != 0 else 0.0

    @staticmethod
    def r_squared(ssr, sst):
        if sst <= 0:
            return 0.0
        return 1 - ssr / sst

    @staticmethod
    def f_statistics(sse, ssr, df_model, df_resid):
        if df_model <= 0 or df_resid <= 0 or ssr <= 0:
            return 0.0, 0.0
        f_stat = (sse / df_model) / (ssr / df_resid)
        return f_stat, f_pval(f_stat, df_model, df_resid)

    @staticmethod
    def chi2_p_value(chi2, df_model):
        if df_model <= 0:
            return 1.0
        return 1 - sp_stats.chi2.cdf(chi2, df_model)

    @staticmethod
    def mse(ssr, df_resid):
        if df_resid <= 0:
            return 0.0
        return ssr / df_resid

    @classmethod
    def root_mse(cls, ssr, df_resid):
        return np.sqrt(cls.mse(ssr, df_resid))
