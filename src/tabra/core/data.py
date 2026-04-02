#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : data.py

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
