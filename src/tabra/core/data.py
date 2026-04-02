#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : data.py

import warnings

import numpy as np
import pandas as pd

from tabra.core.data_ops import DataOps
from tabra.plot import PlotOps
from tabra.models.estimate.ols import OLS
from tabra.models.estimate.panel import PanelModel
from tabra.models.estimate.reghdfe import RegHDFE
from tabra.models.estimate.binary_choice import ProbitModel, LogitModel
from tabra.models.estimate.heckman import HeckmanModel
from tabra.models.estimate.tobit import TobitModel
from tabra.models.estimate.qreg import QuantileRegression


class TabraData:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        style: str = "stata",
        is_display_result: bool = True,
    ):
        """
        is_display_result 是在使用完reg之类的东西之后是不是直接就打印出来结果。
        """
        self._df = df
        self._style = style
        self._is_display_result = is_display_result

        self._result = None
        self._model = None
        self._panel_var = None
        self._time_var = None

    @property
    def result(self):
        return self._result

    @property
    def model(self):
        return self._model

    @property
    def data(self) -> DataOps:
        return DataOps(self)

    @property
    def plot(self) -> PlotOps:
        return PlotOps(self)

    def set_style(self, style: str):
        self._style = style

    def display_result(self, is_display: bool = None):
        if is_display is not None and isinstance(is_display, bool):
            self._is_display_result = is_display

    def reg(self, y: str, x: list[str], is_con: bool = True):
        model = OLS()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def reghdfe(self, y: str, x: list[str], absorb: list[str],
                vce: str = "unadjusted", cluster: list[str] = None,
                is_con: bool = True):
        model = RegHDFE()
        result = model.fit(self._df, y, x, absorb=absorb,
                           vce=vce, cluster=cluster, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def xeset(self, panel_var: str = None, time_var: str = None, clear: bool = False):
        if clear:
            self._panel_var = None
            self._time_var = None
            return
        if panel_var is None:
            raise ValueError("请提供 panel_var，或使用 xeset(clear=True) 清除面板设置")
        self._panel_var = panel_var
        self._time_var = time_var

    def xereg(self, y: str, x: list[str], model: str = "fe", is_con: bool = True):
        if self._panel_var is None:
            raise ValueError("请先调用 xeset() 设置面板变量")
        panel_model = PanelModel()
        result = panel_model.fit(self._df, y, x, self._panel_var, model=model, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def probit(self, y: str, x: list[str], is_con: bool = True):
        model = ProbitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def logit(self, y: str, x: list[str], is_con: bool = True):
        model = LogitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def heckman(self, y: str, x: list[str], select_x: list[str],
                select_var: str = None, is_con: bool = True,
                method: str = "mle"):
        model = HeckmanModel()
        result = model.fit(self._df, y, x, select_x=select_x,
                           select_var=select_var, is_con=is_con,
                           method=method)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def tobit(self, y: str, x: list[str], ll=None, ul=None,
              is_con: bool = True):
        model = TobitModel()
        result = model.fit(self._df, y, x, ll=ll, ul=ul, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def qreg(self, y: str, x: list[str], quantile: float = 0.5,
             is_con: bool = True):
        model = QuantileRegression()
        result = model.fit(self._df, y, x, quantile=quantile, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def summarize(self, var_list: list[str] = None, detail: bool = False):
        from tabra.results.summarize_result import SummarizeResult
        from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

        if var_list is None:
            var_list = self._df.select_dtypes(include="number").columns.tolist()

        obs, mean, std, min_val, max_val = {}, {}, {}, {}, {}
        percentiles, skewness, kurtosis = {}, {}, {}

        for col in var_list:
            series = self._df[col].dropna()
            obs[col] = len(series)
            mean[col] = float(series.mean())
            std[col] = float(series.std(ddof=1))
            min_val[col] = float(series.min())
            max_val[col] = float(series.max())

            if detail:
                percentiles[col] = {}
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    percentiles[col][f"{p}%"] = float(np.percentile(series, p))
                skewness[col] = float(scipy_skew(series, bias=False))
                kurtosis[col] = float(scipy_kurtosis(series, bias=False))

        result = SummarizeResult(
            var_names=var_list, obs=obs, mean=mean, std=std,
            min_val=min_val, max_val=max_val,
            percentiles=percentiles if detail else None,
            skewness=skewness if detail else None,
            kurtosis=kurtosis if detail else None,
            detail=detail,
        )
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def gen(self, var: str, expr: str):
        warnings.warn("建议使用 tab.data.gen() 代替 tab.gen()", DeprecationWarning, stacklevel=2)
        return self.data.gen(var, expr)

    def replace(self, var: str, expr: str, cond: str = None):
        warnings.warn("建议使用 tab.data.replace() 代替 tab.replace()", DeprecationWarning, stacklevel=2)
        return self.data.replace(var, expr, cond=cond)

    def drop(self, vars):
        warnings.warn("建议使用 tab.data.drop() 代替 tab.drop()", DeprecationWarning, stacklevel=2)
        return self.data.drop(vars)

    def keep(self, vars):
        warnings.warn("建议使用 tab.data.keep() 代替 tab.keep()", DeprecationWarning, stacklevel=2)
        return self.data.keep(vars)

    def rename(self, old: str, new: str):
        warnings.warn("建议使用 tab.data.rename() 代替 tab.rename()", DeprecationWarning, stacklevel=2)
        return self.data.rename(old, new)
