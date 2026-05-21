#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod
from tabra.ops.stats import f_pval


class BaseModel(ABC):
    def __init__(self):
        pass

    @staticmethod
    def _prepare_df(df, y, x, extra_cols=None):
        use_cols = [y] + list(x)
        if extra_cols:
            use_cols.extend(extra_cols)
        return df[use_cols].dropna()

    @staticmethod
    def _model_df(k_vars, is_con=True):
        return (k_vars - 1) if is_con else k_vars

    @staticmethod
    def _resid_df(n_obs, k_vars):
        return n_obs - k_vars

    @classmethod
    def _adjusted_r_squared(cls, r_squared, n_obs, k_vars, is_con=True):
        df_resid = cls._resid_df(n_obs, k_vars)
        if df_resid <= 0:
            return 0.0
        adj_n = (n_obs - 1) if is_con else n_obs
        return 1 - (1 - r_squared) * adj_n / df_resid

    @staticmethod
    def _pseudo_r_squared(ll, ll_0):
        return 1 - ll / ll_0 if ll_0 != 0 else 0.0

    @staticmethod
    def _f_statistics(sse, ssr, df_model, df_resid):
        if df_model <= 0 or df_resid <= 0 or ssr <= 0:
            return 0.0, 0.0
        f_stat = (sse / df_model) / (ssr / df_resid)
        return f_stat, f_pval(f_stat, df_model, df_resid)

    @abstractmethod
    def fit(self, df, y, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def estimate(self, df, x, **kwargs):
        raise NotImplementedError
