#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : config.py

from pathlib import Path

from tabra.plot.templates import PlotTemplateBase, TEMPLATES


class Config:
    """Global configuration for a TabraData instance."""

    def __init__(self, tabra):
        self._tabra = tabra
        self._plot_template = TEMPLATES["DEFAULT"]
        self._auto_create_missing_dir = True
        self._figure_save_base = Path.cwd()

    def set_plot_template(self, template: PlotTemplateBase):
        """Set the plot template.

        Args:
            template (PlotTemplateBase): A PlotTemplateBase instance. Use presets from
                tabra.plot.fig_setting, e.g. ``from tabra.plot.fig_setting import AER, QJE``.

        Returns:
            Config: Returns self for method chaining.

        Example:
            >>> from tabra.plot.fig_setting import AER, QJE
            >>> dta = load_data("auto")
            >>> dta.config.set_plot_template(AER)
        """
        if not isinstance(template, PlotTemplateBase):
            raise TypeError("template must be a PlotTemplateBase instance")
        self._plot_template = template
        self._plot_template.apply()
        return self

    @property
    def plot_template(self) -> PlotTemplateBase:
        return self._plot_template

    @property
    def auto_create_missing_dir(self) -> bool:
        """Whether to auto-create missing directories when saving figures."""
        return self._auto_create_missing_dir

    def set_auto_create_missing_dir(self, value: bool = True):
        """Toggle auto-creation of missing directories on save."""
        self._auto_create_missing_dir = value
        return self

    @property
    def figure_save_base(self) -> Path:
        """Base directory for relative figure save paths."""
        return self._figure_save_base

    def set_figure_save_base(self, path):
        """Set the base directory for relative figure save paths.

        Args:
            path (str | Path | os.PathLike): Absolute or relative directory path.

        Returns:
            Config: Returns self for method chaining.
        """
        self._figure_save_base = Path(path)
        return self


# ---- Module-level global settings ----

_global_plot_template = None
_global_figure_save_base = None


def set_plot_template(template):
    """Set the global default plot template.

    Args:
        template (PlotTemplateBase): A PlotTemplateBase instance, e.g. ``AER``, ``QJE``.

    Example:
        >>> from tabra.plot.template import AER
        >>> from tabra.core.config import set_plot_template
        >>> set_plot_template(AER)
    """
    global _global_plot_template
    if not isinstance(template, PlotTemplateBase):
        raise TypeError("template must be a PlotTemplateBase instance")
    _global_plot_template = template
    template.apply()


def set_figure_save_base(base_dir):
    """Set the global base directory for relative figure save paths.

    Args:
        base_dir (str | Path | os.PathLike): Base directory path.

    Example:
        >>> from tabra.core.config import set_figure_save_base
        >>> set_figure_save_base("tmp/figs")
    """
    global _global_figure_save_base
    _global_figure_save_base = Path(base_dir)
