#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : est_accessor.py

"""Estimation accessor for TabraData. All regression/estimation methods live here."""

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


class EstAccessor:
    """Estimation methods accessor for TabraData.

    Access via ``tab.est.reg(...)``, ``tab.est.probit(...)``, etc.
    """

    def __init__(self, tabra):
        self._tabra = tabra

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
            >>> result = dta.est.reg("price", ["weight", "mpg"])
        """
        model = OLS()
        result = model.fit(self._tabra._df, y, x, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.reghdfe("ln_wage", ["age", "tenure"], absorb=["idcode", "year"])
        """
        model = RegHDFE()
        result = model.fit(self._tabra._df, y, x, absorb=absorb,
                           vce=vce, cluster=cluster, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
            result.set_display(True)
        return result

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
            >>> result = dta.est.xtreg("ln_wage", ["age", "tenure"], model="fe")
        """
        if self._tabra._panel_var is None:
            raise ValueError("Call xtset() first to set panel variables")
        panel_model = PanelModel()
        result = panel_model.fit(self._tabra._df, y, x, self._tabra._panel_var, model=model, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.probit("foreign", ["price", "weight"])
        """
        model = ProbitModel()
        result = model.fit(self._tabra._df, y, x, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.logit("foreign", ["price", "weight"])
        """
        model = LogitModel()
        result = model.fit(self._tabra._df, y, x, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.heckman("price", ["weight"], select_x=["mpg", "weight"])
        """
        model = HeckmanModel()
        result = model.fit(self._tabra._df, y, x, select_x=select_x,
                           select_var=select_var, is_con=is_con,
                           method=method)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
            result.set_display(True)
        return result

    def tobit(self, y: str, x: list[str], ll=None, ul=None,
              vce: str = "unadjusted", is_con: bool = True):
        """Fit a Tobit censored regression model.

        Args:
            y (str): Dependent variable name.
            x (list[str]): Independent variable names.
            ll (float): Left-censoring limit. Default None (no censoring).
            ul (float): Right-censoring limit. Default None (no censoring).
            vce (str): Variance-covariance estimator type. "unadjusted" or "robust".
            is_con (bool): Whether to include a constant term. Default True.

        Returns:
            TobitResult: Estimation result.

        Example:
            >>> dta = load_data("auto")
            >>> result = dta.est.tobit("price", ["weight", "mpg"], ll=0)
        """
        model = TobitModel()
        result = model.fit(self._tabra._df, y, x, ll=ll, ul=ul, vce=vce, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.qreg("price", ["weight", "mpg"], quantile=0.25)
        """
        model = QuantileRegression()
        result = model.fit(self._tabra._df, y, x, quantile=quantile, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.oprobit("rep78", ["price", "weight"])
        """
        model = OrderedProbitModel()
        result = model.fit(self._tabra._df, y, x, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.ologit("rep78", ["price", "weight"])
        """
        model = OrderedLogitModel()
        result = model.fit(self._tabra._df, y, x, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.glm("price", ["weight", "mpg"], family="poisson")
        """
        model = GLMModel()
        result = model.fit(self._tabra._df, y, x, family=family, link=link, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.mlogit("rep78", ["price", "weight"])
        """
        model = MultinomialLogitModel()
        result = model.fit(self._tabra._df, y, x, base_outcome=base_outcome,
                           is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.ivreg("price", exog=["weight"],
            ...     endog=["mpg"], iv=["foreign", "headroom"], estimator="2sls")
        """
        model = IVModel()
        result = model.fit(self._tabra._df, y, exog=exog, endog=endog,
                           instruments=iv, estimator=estimator,
                           vce=vce, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.ivreg2("price", exog=["weight"],
            ...     endog=["mpg"], iv=["foreign", "headroom"],
            ...     estimator="cue")
        """
        model = IVModel()
        result = model.fit(self._tabra._df, y, exog=exog, endog=endog,
                           instruments=iv, estimator=estimator,
                           vce=vce, is_con=is_con,
                           fuller_alpha=fuller_alpha,
                           kclass_k=kclass_k,
                           cluster=cluster)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.ivprobit("foreign", exog=["weight"],
            ...     endog=["mpg"], iv=["headroom", "trunk"], method="mle")
        """
        model = IVProbitModel()
        result = model.fit(self._tabra._df, y, exog=exog, endog=endog,
                           instruments=iv, method=method, vce=vce,
                           is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.ivtobit("price", exog=["weight"],
            ...     endog=["mpg"], iv=["headroom", "trunk"], ll=0)
        """
        model = IVTobitModel()
        result = model.fit(self._tabra._df, y, exog=exog, endog=endog,
                           instruments=iv, ll=ll, ul=ul,
                           method=method, vce=vce, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
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
            >>> result = dta.est.ivreghdfe("ln_wage", exog=["age"],
            ...     endog=["tenure"], iv=["union"], absorb=["idcode"])
        """
        model = IVRegHDFEModel()
        result = model.fit(self._tabra._df, y, exog=exog, endog=endog,
                           instruments=iv, absorb=absorb,
                           estimator=estimator, vce=vce,
                           cluster=cluster, is_con=is_con)
        result.set_style(self._tabra._style)
        self._tabra._result = result
        if self._tabra._is_display_result:
            result.set_display(True)
        return result
