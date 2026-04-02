#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

import matplotlib.pyplot as plt


class TabraFigure:
    """Wrapper around matplotlib Figure with user-friendly API."""

    def __init__(self, fig):
        self._fig = fig

    def save(self, filename: str, dpi: int = 300, **kwargs):
        self._fig.savefig(filename, dpi=dpi, bbox_inches="tight", **kwargs)
        return self

    def show(self):
        plt.show()
        return self

    def close(self):
        plt.close(self._fig)
        return self

    @property
    def figure(self):
        return self._fig


class PlotOps:
    def __init__(self, tabra):
        self._tabra = tabra

    @property
    def _df(self):
        return self._tabra._df

    def scatter(self, y: str, x: str, title: str = None,
                xtitle: str = None, ytitle: str = None,
                fig_setting=None):
        fig, ax = plt.subplots()
        ax.scatter(self._df[x], self._df[y])
        ax.set_xlabel(xtitle if xtitle is not None else x)
        ax.set_ylabel(ytitle if ytitle is not None else y)
        if title is not None:
            ax.set_title(title)
        return TabraFigure(fig)
