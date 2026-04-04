#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from dataclasses import dataclass, field


@dataclass
class PlotTemplateBase:
    """Base class for plot styling templates."""

    # Font
    font_family: str = "sans-serif"
    font_name: str = "Arial"
    title_size: float = 12.0
    label_size: float = 10.0
    tick_size: float = 8.0
    legend_size: float = 9.0

    # Figure
    fig_width: float = 3.5
    fig_height: float = 2.5
    dpi: int = 300

    # Axes
    line_width: float = 1.0
    marker_size: float = 20.0
    grid: bool = False

    # Spines
    spine_top: bool = True
    spine_right: bool = True
    tick_direction: str = "out"

    # Color
    primary_color: str = "#1f77b4"
    color_cycle: tuple = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    )

    # Background
    facecolor: str = "white"

    # Export
    default_format: str = "pdf"

    def apply(self):
        """Apply this template to matplotlib rcParams."""
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": self.font_family,
            "font.sans-serif": [self.font_name, "Helvetica", "DejaVu Sans"],
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "lines.linewidth": self.line_width,
            "axes.grid": self.grid,
            "figure.dpi": self.dpi,
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            "axes.facecolor": self.facecolor,
            "figure.facecolor": self.facecolor,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.prop_cycle": __import__("matplotlib").cycler(
                color=list(self.color_cycle)
            ),
        })
