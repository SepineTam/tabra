#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

import os

import matplotlib.pyplot as plt


class TabraFigure:
    """Wrapper around matplotlib Figure with user-friendly API."""

    def __init__(self, fig, tabra=None):
        self._fig = fig
        self._tabra = tabra

    def save(self, filename: str, dpi: int = 300, **kwargs):
        if self._tabra is not None and self._tabra._config.auto_create_missing_dir:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
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
                template=None,
                fig_setting=None):
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = plt.subplots(
            figsize=(template.fig_width, template.fig_height),
            dpi=template.dpi,
        )
        ax.scatter(self._df[x], self._df[y],
                   s=template.marker_size, c=template.primary_color)
        ax.set_xlabel(xtitle if xtitle is not None else x)
        ax.set_ylabel(ytitle if ytitle is not None else y)
        if title is not None:
            ax.set_title(title)
        if not template.spine_top:
            ax.spines["top"].set_visible(False)
        if not template.spine_right:
            ax.spines["right"].set_visible(False)
        return TabraFigure(fig, tabra=self._tabra)
