#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : data.py

import numpy as np
import pandas as pd

from tabra.models.estimate.ols import OLS
from tabra.models.estimate.panel import PanelModel


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

    @property
    def result(self):
        return self._result

    @property
    def model(self):
        return self._model

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

    def xeset(self, panel_var: str):
        self._panel_var = panel_var

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
