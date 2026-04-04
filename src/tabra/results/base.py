#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod


class BaseResult(ABC):
    def __init__(self):
        self._style = "stata"

    def set_style(self, style: str):
        self._style = style

    def set_display(self, is_display: bool = True):
        if is_display:
            print(self.summary())

    @abstractmethod
    def summary(self): ...

    def __repr__(self):
        return self.summary()

    @abstractmethod
    def save(self, path): ...

    def coefplot(self, **kwargs):
        """Create a coefficient plot from this result.

        Args:
            **kwargs: Passed to tabra.plot.coefplot.plot_coefplot().

        Returns:
            TabraFigure: Wrapped matplotlib figure.
        """
        from tabra.plot.coefplot import plot_coefplot
        return plot_coefplot(self, **kwargs)
