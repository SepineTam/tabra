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

from tabra.core.config import Config
from tabra.core.data_ops import DataOps
from tabra.plot import PlotOps
from tabra.models.estimate.ols import OLS
from tabra.models.estimate.panel import PanelModel
from tabra.models.estimate.reghdfe import RegHDFE
from tabra.models.estimate.binary_choice import ProbitModel, LogitModel
from tabra.models.estimate.heckman import HeckmanModel
from tabra.models.estimate.tobit import TobitModel
from tabra.models.estimate.qreg import QuantileRegression
from tabra.models.estimate.ordered_choice import OrderedProbitModel, OrderedLogitModel
from tabra.models.estimate.glm import GLMModel
from tabra.models.estimate.mlogit import MultinomialLogitModel


class TabraData:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        style: str = "stata",
        is_display_result: bool = True,
    ):
        """Initialize a TabraData instance.

        Args:
            df (pd.DataFrame): Underlying dataset.
            style (str): Output display style. Default "stata".
            is_display_result (bool): Whether to print estimation results
                immediately after calling reg, probit, etc. Default True.
        """
        self._df = df
        self._style = style
        self._is_display_result = is_display_result

        self._result = None
        self._model = None
        self._panel_var = None
        self._time_var = None
        self._config = Config(self)

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
    def config(self) -> Config:
        return self._config

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
        """Set or clear panel variables for panel data estimation.

        Args:
            panel_var (str): Column name for the entity (panel) identifier.
            time_var (str): Column name for the time identifier.
            clear (bool): If True, clear the current panel settings. Default False.

        Example:
            >>> dta = load_data("nlswork")
            >>> dta.xeset("idcode", "year")
        """
        if clear:
            self._panel_var = None
            self._time_var = None
            return
        if panel_var is None:
            raise ValueError("panel_var is required, or use xeset(clear=True) to clear")
        self._panel_var = panel_var
        self._time_var = time_var

    def xereg(self, y: str, x: list[str], model: str = "fe", is_con: bool = True):
        if self._panel_var is None:
            raise ValueError("Call xeset() first to set panel variables")
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

    def oprobit(self, y: str, x: list[str], is_con: bool = True):
        model = OrderedProbitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ologit(self, y: str, x: list[str], is_con: bool = True):
        model = OrderedLogitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def glm(self, y: str, x: list[str], family: str = "gaussian",
            link: str = None, is_con: bool = True):
        model = GLMModel()
        result = model.fit(self._df, y, x, family=family, link=link, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def mlogit(self, y: str, x: list[str], base_outcome=None,
               is_con: bool = True):
        model = MultinomialLogitModel()
        result = model.fit(self._df, y, x, base_outcome=base_outcome,
                           is_con=is_con)
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
        warnings.warn("Use tab.data.gen() instead of tab.gen()", DeprecationWarning, stacklevel=2)
        return self.data.gen(var, expr)

    def replace(self, var: str, expr: str, cond: str = None):
        warnings.warn("Use tab.data.replace() instead of tab.replace()", DeprecationWarning, stacklevel=2)
        return self.data.replace(var, expr, cond=cond)

    def drop(self, vars):
        warnings.warn("Use tab.data.drop() instead of tab.drop()", DeprecationWarning, stacklevel=2)
        return self.data.drop(vars)

    def keep(self, vars):
        warnings.warn("Use tab.data.keep() instead of tab.keep()", DeprecationWarning, stacklevel=2)
        return self.data.keep(vars)

    def rename(self, old: str, new: str):
        warnings.warn("Use tab.data.rename() instead of tab.rename()", DeprecationWarning, stacklevel=2)
        return self.data.rename(old, new)

    def __add__(self, other):
        """Support tab + df / tab + tab2, returns a new TabraData."""
        new_tab = TabraData(
            self._df.copy(),
            style=self._style,
            is_display_result=self._is_display_result,
        )
        new_tab.data.append(other)
        return new_tab
