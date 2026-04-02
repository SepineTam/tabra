#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : fig_setting.py

from dataclasses import dataclass


@dataclass
class PlotTemplate:
    """Plot styling template for journal submissions."""

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
    tick_direction: str = "out"  # "in", "out", "inout"

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


# ---- Preset templates ----

# AER: tight single column, no top/right spine, classic economics style
AER = PlotTemplate(
    font_name="Arial",
    title_size=11.0,
    label_size=9.0,
    tick_size=8.0,
    legend_size=8.0,
    fig_width=3.5,
    fig_height=2.5,
    dpi=300,
    line_width=0.8,
    marker_size=15.0,
    grid=False,
    spine_top=False,
    spine_right=False,
    tick_direction="out",
    primary_color="#2171b5",
    color_cycle=("#2171b5", "#cb181d", "#238b45", "#f16913",
                 "#6a51a3", "#d94801", "#08519c", "#006d2c"),
    facecolor="white",
    default_format="pdf",
)

# QJE: OUP standard, larger fonts, boxed frame, warmer palette
QJE = PlotTemplate(
    font_name="Arial",
    title_size=13.0,
    label_size=12.0,
    tick_size=10.0,
    legend_size=10.0,
    fig_width=4.0,
    fig_height=3.0,
    dpi=300,
    line_width=0.9,
    marker_size=20.0,
    grid=False,
    spine_top=True,
    spine_right=True,
    tick_direction="in",
    primary_color="#e41a1c",
    color_cycle=("#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                 "#ff7f00", "#a65628", "#f781bf", "#999999"),
    facecolor="white",
    default_format="pdf",
)

# JPE: Chicago Press, minimal, smallest, muted palette
JPE = PlotTemplate(
    font_name="Arial",
    title_size=10.0,
    label_size=9.0,
    tick_size=7.0,
    legend_size=8.0,
    fig_width=3.5,
    fig_height=2.5,
    dpi=300,
    line_width=0.75,
    marker_size=12.0,
    grid=False,
    spine_top=False,
    spine_right=False,
    tick_direction="in",
    primary_color="#252525",
    color_cycle=("#252525", "#636363", "#969696", "#bdbdbd",
                 "#737373", "#525252", "#cccccc", "#424242"),
    facecolor="white",
    default_format="pdf",
)

# PRESENTATION: big, bold, colorful, grid
PRESENTATION = PlotTemplate(
    font_name="Arial",
    title_size=18.0,
    label_size=15.0,
    tick_size=13.0,
    legend_size=14.0,
    fig_width=8.0,
    fig_height=5.5,
    dpi=150,
    line_width=2.5,
    marker_size=60.0,
    grid=True,
    spine_top=False,
    spine_right=False,
    tick_direction="out",
    primary_color="#4c72b0",
    color_cycle=("#4c72b0", "#dd8452", "#55a868", "#c44e52",
                 "#8172b3", "#937860", "#da8bc3", "#8c8c8c"),
    facecolor="white",
    default_format="png",
)

DEFAULT = PlotTemplate()

TEMPLATES = {
    "AER": AER,
    "QJE": QJE,
    "JPE": JPE,
    "PRESENTATION": PRESENTATION,
    "DEFAULT": DEFAULT,
}
