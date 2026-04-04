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
from tabra.models.estimate.iv import IVModel
from tabra.models.estimate.ivprobit import IVProbitModel
from tabra.models.estimate.ivtobit import IVTobitModel
from tabra.models.estimate.ivreghdfe import IVRegHDFEModel


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
        """Fit an OLS linear regression.

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            OLSResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.reg("price", ["weight", "mpg"])
        """
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
        """Fit a linear model with high-dimensional fixed effects (HDFE).

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            absorb (list[str]): Variables to absorb as fixed effects.
            vce (str): Variance-covariance estimator type. One of "unadjusted",
                "robust", "cluster". Default "unadjusted".
            cluster (list[str]): Cluster variable names. Required when vce="cluster".
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            RegHDFEResult: Estimation result.

        Example:
            >>> dta = load_data("nlswork")
            >>> dta.xtset("idcode", "year")
            >>> result = dta.reghdfe("ln_wage", ["age", "tenure"], absorb=["idcode", "year"])
        """
        model = RegHDFE()
        result = model.fit(self._df, y, x, absorb=absorb,
                           vce=vce, cluster=cluster, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

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
        """Fit a panel data regression model.

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            model (str): Estimation method. One of "fe", "re", "be", "mle", "pa". Default "fe".
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            PanelResult: Panel estimation result.

        Example:
            >>> dta = load_data("nlswork")
            >>> dta.xtset("idcode", "year")
            >>> result = dta.xtreg("ln_wage", ["age", "tenure"], model="fe")
        """
        if self._panel_var is None:
            raise ValueError("Call xtset() first to set panel variables")
        panel_model = PanelModel()
        result = panel_model.fit(self._df, y, x, self._panel_var, model=model, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def probit(self, y: str, x: list[str], is_con: bool = True):
        """Fit a probit regression model.

        Args:
            y (str): Binary dependent variable name (0/1).
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            BinaryChoiceResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.probit("foreign", ["price", "weight"])
        """
        model = ProbitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def logit(self, y: str, x: list[str], is_con: bool = True):
        """Fit a logistic regression model.

        Args:
            y (str): Binary dependent variable name (0/1).
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            BinaryChoiceResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.logit("foreign", ["price", "weight"])
        """
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
        """Fit a Heckman selection model.

        Args:
            y (str): Outcome variable name.
            x (list[str]): Outcome equation independent variable names.
            select_x (list[str]): Selection equation independent variable names.
            select_var (str): Binary selection indicator variable name.
                If None, missing y values indicate non-selection.
            is_con (bool): Whether to include constant terms. Default True.
            method (str): Estimation method. One of "mle", "twostep". Default "mle".

        Returns:
            HeckmanResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.heckman("price", ["weight"], select_x=["mpg", "weight"])
        """
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
        """Fit a Tobit censored regression model.

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            ll (float): Left-censoring limit. Default None (no censoring).
            ul (float): Right-censoring limit. Default None (no censoring).
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            TobitResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.tobit("price", ["weight", "mpg"], ll=0)
        """
        model = TobitModel()
        result = model.fit(self._df, y, x, ll=ll, ul=ul, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def qreg(self, y: str, x: list[str], quantile: float = 0.5,
             is_con: bool = True):
        """Fit a quantile regression model.

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            quantile (float): Target quantile (0, 1). Default 0.5 (median).
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            QRegResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.qreg("price", ["weight", "mpg"], quantile=0.25)
        """
        model = QuantileRegression()
        result = model.fit(self._df, y, x, quantile=quantile, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def oprobit(self, y: str, x: list[str], is_con: bool = True):
        """Fit an ordered probit regression model.

        Args:
            y (str): Ordinal dependent variable name.
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            OrderedChoiceResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.oprobit("rep78", ["price", "weight"])
        """
        model = OrderedProbitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ologit(self, y: str, x: list[str], is_con: bool = True):
        """Fit an ordered logistic regression model.

        Args:
            y (str): Ordinal dependent variable name.
            x (list[str]): Independent variable names.
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            OrderedChoiceResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.ologit("rep78", ["price", "weight"])
        """
        model = OrderedLogitModel()
        result = model.fit(self._df, y, x, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def glm(self, y: str, x: list[str], family: str = "gaussian",
            link: str = None, is_con: bool = True):
        """Fit a Generalized Linear Model.

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            family (str): Distribution family. One of "gaussian", "binomial",
                "poisson", "gamma". Default "gaussian".
            link (str): Link function. Default None (use canonical link).
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            GLMResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.glm("price", ["weight", "mpg"], family="poisson")
        """
        model = GLMModel()
        result = model.fit(self._df, y, x, family=family, link=link, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def mlogit(self, y: str, x: list[str], base_outcome=None,
               is_con: bool = True):
        """Fit a multinomial logistic regression model.

        Args:
            y (str): Categorical dependent variable name.
            x (list[str]): Independent variable names.
            base_outcome: Base category value. Default None (uses first category).
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            MLogitResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.mlogit("rep78", ["price", "weight"])
        """
        model = MultinomialLogitModel()
        result = model.fit(self._df, y, x, base_outcome=base_outcome,
                           is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ivreg(self, y: str, exog: list[str], endog: list[str],
              iv: list[str], estimator: str = "2sls",
              vce: str = "unadjusted", is_con: bool = True):
        """Fit an instrumental-variables regression model.

        Args:
            y (str): Dependent variable name.
            exog (list[str]): Exogenous explanatory variable names.
            endog (list[str]): Endogenous variable names.
            iv (list[str]): Instrument variable names.
            estimator (str): One of "2sls", "gmm", "liml". Default "2sls".
            vce (str): "unadjusted" or "robust". Default "unadjusted".
            is_con (bool): Whether to add a constant. Default True.

        Returns:
            IVResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.ivreg("price", exog=["weight"],
            ...     endog=["mpg"], iv=["foreign", "headroom"], estimator="2sls")
        """
        model = IVModel()
        result = model.fit(self._df, y, exog=exog, endog=endog,
                           instruments=iv, estimator=estimator,
                           vce=vce, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ivreg2(self, y: str, exog: list[str], endog: list[str],
               iv: list[str], estimator: str = "2sls",
               vce: str = "unadjusted", cluster: list[str] = None,
               is_con: bool = True, fuller_alpha: float = 1.0,
               kclass_k: float = None):
        """Fit an enhanced IV regression (ivreg2) with more estimators and diagnostics.

        Args:
            y (str): Dependent variable name.
            exog (list[str]): Exogenous explanatory variable names.
            endog (list[str]): Endogenous variable names.
            iv (list[str]): Instrument variable names.
            estimator (str): One of "2sls", "gmm", "liml", "cue", "fuller", "kclass".
            vce (str): "unadjusted", "robust", or "cluster".
            cluster (list[str]): Cluster variable names. Required when vce="cluster".
            is_con (bool): Whether to add a constant. Default True.
            fuller_alpha (float): Fuller alpha parameter. Default 1.0.
            kclass_k (float): k-class k value. Required when estimator="kclass".

        Returns:
            IVResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.ivreg2("price", exog=["weight"],
            ...     endog=["mpg"], iv=["foreign", "headroom"],
            ...     estimator="cue")
        """
        model = IVModel()
        result = model.fit(self._df, y, exog=exog, endog=endog,
                           instruments=iv, estimator=estimator,
                           vce=vce, is_con=is_con,
                           fuller_alpha=fuller_alpha,
                           kclass_k=kclass_k,
                           cluster=cluster)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ivprobit(self, y: str, exog: list[str], endog: list[str],
                 iv: list[str], method: str = "mle",
                 vce: str = "unadjusted", is_con: bool = True):
        """Fit an IV probit regression model.

        Args:
            y (str): Binary dependent variable name (0/1).
            exog (list[str]): Exogenous explanatory variable names.
            endog (list[str]): Endogenous variable names.
            iv (list[str]): Instrument variable names.
            method (str): "mle" or "twostep". Default "mle".
            vce (str): "unadjusted" or "robust". Default "unadjusted".
            is_con (bool): Whether to add a constant. Default True.

        Returns:
            IVProbitResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.ivprobit("foreign", exog=["weight"],
            ...     endog=["mpg"], iv=["headroom", "trunk"], method="mle")
        """
        model = IVProbitModel()
        result = model.fit(self._df, y, exog=exog, endog=endog,
                           instruments=iv, method=method, vce=vce,
                           is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ivtobit(self, y: str, exog: list[str], endog: list[str],
                iv: list[str], ll=None, ul=None,
                method: str = "mle", vce: str = "unadjusted",
                is_con: bool = True):
        """Fit an IV tobit censored regression model.

        Args:
            y (str): Censored dependent variable name.
            exog (list[str]): Exogenous explanatory variable names.
            endog (list[str]): Endogenous variable names.
            iv (list[str]): Instrument variable names.
            ll (float): Left-censoring limit. Default None.
            ul (float): Right-censoring limit. Default None.
            method (str): "mle" or "twostep". Default "mle".
            vce (str): "unadjusted" or "robust". Default "unadjusted".
            is_con (bool): Whether to add a constant. Default True.

        Returns:
            IVTobitResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.ivtobit("price", exog=["weight"],
            ...     endog=["mpg"], iv=["headroom", "trunk"], ll=0)
        """
        model = IVTobitModel()
        result = model.fit(self._df, y, exog=exog, endog=endog,
                           instruments=iv, ll=ll, ul=ul,
                           method=method, vce=vce, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

    def ivreghdfe(self, y: str, exog: list[str], endog: list[str],
                  iv: list[str], absorb: list[str],
                  estimator: str = "2sls", vce: str = "unadjusted",
                  cluster: list[str] = None, is_con: bool = True):
        """Fit an IV regression with high-dimensional fixed effects.

        Args:
            y (str): Dependent variable name.
            exog (list[str]): Exogenous explanatory variable names.
            endog (list[str]): Endogenous variable names.
            iv (list[str]): Instrument variable names.
            absorb (list[str]): FE variable names to absorb.
            estimator (str): "2sls", "gmm", "liml". Default "2sls".
            vce (str): "unadjusted", "robust", "cluster". Default "unadjusted".
            cluster (list[str]): Cluster variable names.
            is_con (bool): Whether to add a constant. Default True.

        Returns:
            IVResult: Estimation result.

        Example:
            >>> dta = load_data("nlswork")
            >>> dta.xtset("idcode", "year")
            >>> result = dta.ivreghdfe("ln_wage", exog=["age"],
            ...     endog=["tenure"], iv=["union"], absorb=["idcode"])
        """
        model = IVRegHDFEModel()
        result = model.fit(self._df, y, exog=exog, endog=endog,
                           instruments=iv, absorb=absorb,
                           estimator=estimator, vce=vce,
                           cluster=cluster, is_con=is_con)
        result.set_style(self._style)
        self._result = result
        if self._is_display_result:
            result.set_display(True)
        return result

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
