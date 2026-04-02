#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : config.py

from tabra.plot.fig_setting import PlotTemplate, TEMPLATES


class Config:
    """Global configuration for a TabraData instance."""

    def __init__(self, tabra):
        self._tabra = tabra
        self._plot_template = TEMPLATES["DEFAULT"]
        self._auto_create_missing_dir = True

    def set_plot_template(self, template: PlotTemplate):
        """
        Set the plot template.

        Parameters
        ----------
        template : PlotTemplate
            A PlotTemplate instance. Use presets from tabra.plot.fig_setting:
            e.g. from tabra.plot.fig_setting import AER, QJE
        """
        if not isinstance(template, PlotTemplate):
            raise TypeError("template must be a PlotTemplate instance")
        self._plot_template = template
        self._plot_template.apply()
        return self

    @property
    def plot_template(self) -> PlotTemplate:
        return self._plot_template

    @property
    def auto_create_missing_dir(self) -> bool:
        """Whether to auto-create missing directories when saving figures."""
        return self._auto_create_missing_dir

    def set_auto_create_missing_dir(self, value: bool = True):
        """Toggle auto-creation of missing directories on save."""
        self._auto_create_missing_dir = value
        return self
