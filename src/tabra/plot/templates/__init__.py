#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

from .base import PlotTemplateBase
from .econ import AER, QJE, JPE, ECONOMETRICA, RES

# Presentation: big, bold, colorful, grid
PRESENTATION = PlotTemplateBase(
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

DEFAULT = PlotTemplateBase()

TEMPLATES = {
    "AER": AER,
    "QJE": QJE,
    "JPE": JPE,
    "ECONOMETRICA": ECONOMETRICA,
    "RES": RES,
    "PRESENTATION": PRESENTATION,
    "DEFAULT": DEFAULT,
}

__all__ = [
    "PlotTemplateBase",
    "AER", "QJE", "JPE", "ECONOMETRICA", "RES",
    "PRESENTATION", "DEFAULT",
    "TEMPLATES",
]
