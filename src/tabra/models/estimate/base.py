#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod
from tabra.models.estimate.stats import EstimationStats


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
        return EstimationStats.model_df(k_vars, is_con=is_con)

    @staticmethod
    def _resid_df(n_obs, k_vars):
        return EstimationStats.resid_df(n_obs, k_vars)

    @classmethod
    def _adjusted_r_squared(cls, r_squared, n_obs, k_vars, is_con=True):
        return EstimationStats.adjusted_r_squared(
            r_squared, n_obs, k_vars, is_con=is_con
        )

    @staticmethod
    def _pseudo_r_squared(ll, ll_0):
        return EstimationStats.pseudo_r_squared(ll, ll_0)

    @staticmethod
    def _f_statistics(sse, ssr, df_model, df_resid):
        return EstimationStats.f_statistics(sse, ssr, df_model, df_resid)

    @abstractmethod
    def fit(self, df, y, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def estimate(self, df, x, **kwargs):
        raise NotImplementedError
