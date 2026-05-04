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

from tabra.core.about import About
from tabra.core.config import Config
from tabra.core.data_ops import DataOps
from tabra.core.est_accessor import EstAccessor
from tabra.plot import PlotOps


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
    def setting(self) -> Config:
        return self._config

    @property
    def plot(self) -> PlotOps:
        return PlotOps(self)

    @property
    def est(self) -> EstAccessor:
        return EstAccessor(self)

    @property
    def about(self) -> About:
        return About(self)

    def set_style(self, style: str):
        """Set the output display style.

        Args:
            style (str): Style name, e.g. "stata".

        Returns:
            None
        """
        self._style = style

    def display_result(self, is_display: bool = None):
        """Toggle whether estimation results are printed immediately.

        Args:
            is_display (bool): If True, results are printed after estimation.
                If None, no change is made.
        """
        if is_display is not None and isinstance(is_display, bool):
            self._is_display_result = is_display

    def reg(self, y: str, x: list[str], is_con: bool = True):
        warnings.warn(
            "dta.reg() is deprecated, use dta.est.reg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.reg(y, x, is_con=is_con)

    def reghdfe(self, y: str, x: list[str], absorb: list[str],
                vce: str = "unadjusted", cluster: list[str] = None,
                is_con: bool = True):
        warnings.warn(
            "dta.reghdfe() is deprecated, use dta.est.reghdfe() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.reghdfe(y, x, absorb=absorb, vce=vce, cluster=cluster, is_con=is_con)

    def xtset(self, panel_var: str = None, time_var: str = None, clear: bool = False):
        """Set or clear panel variables for panel data estimation.

        Args:
            panel_var (str): Column name for the entity (panel) identifier.
            time_var (str): Column name for the time identifier.
            clear (bool): If True, clear the current panel settings. Default False.

        Example:
            >>> dta = load_data("nlswork")
            >>> dta.xtset("idcode", "year")
        """
        if clear:
            self._panel_var = None
            self._time_var = None
            return
        if panel_var is None:
            raise ValueError("panel_var is required, or use xtset(clear=True) to clear")
        self._panel_var = panel_var
        self._time_var = time_var

    def xtreg(self, y: str, x: list[str], model: str = "fe", is_con: bool = True):
        warnings.warn(
            "dta.xtreg() is deprecated, use dta.est.xtreg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.xtreg(y, x, model=model, is_con=is_con)

    def probit(self, y: str, x: list[str], is_con: bool = True):
        warnings.warn(
            "dta.probit() is deprecated, use dta.est.probit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.probit(y, x, is_con=is_con)

    def logit(self, y: str, x: list[str], is_con: bool = True):
        warnings.warn(
            "dta.logit() is deprecated, use dta.est.logit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.logit(y, x, is_con=is_con)

    def heckman(self, y: str, x: list[str], select_x: list[str],
                select_var: str = None, is_con: bool = True,
                method: str = "mle"):
        warnings.warn(
            "dta.heckman() is deprecated, use dta.est.heckman() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.heckman(y, x, select_x=select_x, select_var=select_var, is_con=is_con, method=method)

    def tobit(self, y: str, x: list[str], ll=None, ul=None,
              vce: str = "unadjusted", is_con: bool = True):
        warnings.warn(
            "dta.tobit() is deprecated, use dta.est.tobit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.tobit(y, x, ll=ll, ul=ul, vce=vce, is_con=is_con)

    def qreg(self, y: str, x: list[str], quantile: float = 0.5,
             is_con: bool = True):
        warnings.warn(
            "dta.qreg() is deprecated, use dta.est.qreg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.qreg(y, x, quantile=quantile, is_con=is_con)

    def oprobit(self, y: str, x: list[str], is_con: bool = True):
        warnings.warn(
            "dta.oprobit() is deprecated, use dta.est.oprobit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.oprobit(y, x, is_con=is_con)

    def ologit(self, y: str, x: list[str], is_con: bool = True):
        warnings.warn(
            "dta.ologit() is deprecated, use dta.est.ologit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.ologit(y, x, is_con=is_con)

    def glm(self, y: str, x: list[str], family: str = "gaussian",
            link: str = None, is_con: bool = True):
        warnings.warn(
            "dta.glm() is deprecated, use dta.est.glm() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.glm(y, x, family=family, link=link, is_con=is_con)

    def mlogit(self, y: str, x: list[str], base_outcome=None,
               is_con: bool = True):
        warnings.warn(
            "dta.mlogit() is deprecated, use dta.est.mlogit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.mlogit(y, x, base_outcome=base_outcome, is_con=is_con)

    def ivreg(self, y: str, exog: list[str], endog: list[str],
              iv: list[str], estimator: str = "2sls",
              vce: str = "unadjusted", is_con: bool = True):
        warnings.warn(
            "dta.ivreg() is deprecated, use dta.est.ivreg() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.ivreg(y, exog=exog, endog=endog, iv=iv,
                              estimator=estimator, vce=vce, is_con=is_con)

    def ivreg2(self, y: str, exog: list[str], endog: list[str],
               iv: list[str], estimator: str = "2sls",
               vce: str = "unadjusted", cluster: list[str] = None,
               is_con: bool = True, fuller_alpha: float = 1.0,
               kclass_k: float = None):
        warnings.warn(
            "dta.ivreg2() is deprecated, use dta.est.ivreg2() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.ivreg2(y, exog=exog, endog=endog, iv=iv,
                               estimator=estimator, vce=vce,
                               cluster=cluster, is_con=is_con,
                               fuller_alpha=fuller_alpha,
                               kclass_k=kclass_k)

    def ivprobit(self, y: str, exog: list[str], endog: list[str],
                 iv: list[str], method: str = "mle",
                 vce: str = "unadjusted", is_con: bool = True):
        warnings.warn(
            "dta.ivprobit() is deprecated, use dta.est.ivprobit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.ivprobit(y, exog=exog, endog=endog, iv=iv,
                                 method=method, vce=vce, is_con=is_con)

    def ivtobit(self, y: str, exog: list[str], endog: list[str],
                iv: list[str], ll=None, ul=None,
                method: str = "mle", vce: str = "unadjusted",
                is_con: bool = True):
        warnings.warn(
            "dta.ivtobit() is deprecated, use dta.est.ivtobit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.ivtobit(y, exog=exog, endog=endog, iv=iv,
                                ll=ll, ul=ul, method=method,
                                vce=vce, is_con=is_con)

    def ivreghdfe(self, y: str, exog: list[str], endog: list[str],
                  iv: list[str], absorb: list[str],
                  estimator: str = "2sls", vce: str = "unadjusted",
                  cluster: list[str] = None, is_con: bool = True):
        warnings.warn(
            "dta.ivreghdfe() is deprecated, use dta.est.ivreghdfe() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.est.ivreghdfe(y, exog=exog, endog=endog, iv=iv,
                                  absorb=absorb, estimator=estimator,
                                  vce=vce, cluster=cluster, is_con=is_con)

    def summarize(self, var_list: list[str] = None, detail: bool = False):
        """Compute summary statistics for numeric variables.

        .. deprecated::
            Use ``dta.data.sum()`` instead.

        Args:
            var_list (list[str]): Variable names to summarize. If None, all numeric columns are used.
            detail (bool): If True, include percentiles, skewness, and kurtosis. Default False.

        Returns:
            SummarizeResult: Summary statistics result.

        Example:
            >>> dta = load_data("auto")
            >>> dta.summarize(["price", "weight", "mpg"])
            >>> dta.summarize(detail=True)
        """
        import warnings
        warnings.warn(
            "dta.summarize() is deprecated, use dta.data.sum() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.data.sum(var_list, detail=detail)

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
        """Support tab + df / tab + tab2, returns a new TabraData.

        Args:
            other (pd.DataFrame or TabraData): Data to vertically stack.

        Returns:
            TabraData: A new TabraData instance with stacked data.
        """
        new_tab = TabraData(
            self._df.copy(),
            style=self._style,
            is_display_result=self._is_display_result,
        )
        new_tab.data.append(other)
        return new_tab
