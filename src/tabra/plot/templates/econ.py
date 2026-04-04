#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : econ.py

from .base import PlotTemplateBase

# ---- Economics Journal Templates ----

# AER: American Economic Review
# Tight single column, no top/right spine, classic economics style
AER = PlotTemplateBase(
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

# QJE: Quarterly Journal of Economics
# OUP standard, larger fonts, boxed frame, warmer palette
QJE = PlotTemplateBase(
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

# JPE: Journal of Political Economy
# Chicago Press, minimal, smallest, muted palette
JPE = PlotTemplateBase(
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

# Econometrica: Wiley, clean, blue tones
ECONOMETRICA = PlotTemplateBase(
    font_name="Arial",
    title_size=11.0,
    label_size=9.5,
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
    primary_color="#08519c",
    color_cycle=("#08519c", "#3182bd", "#6baed6", "#9ecae1",
                 "#9e9ac8", "#756bb1", "#54278f", "#4292c6"),
    facecolor="white",
    default_format="pdf",
)

# RES: Review of Economics and Statistics
RES = PlotTemplateBase(
    font_name="Arial",
    title_size=11.0,
    label_size=9.5,
    tick_size=8.5,
    legend_size=8.5,
    fig_width=3.5,
    fig_height=2.5,
    dpi=300,
    line_width=0.8,
    marker_size=16.0,
    grid=False,
    spine_top=False,
    spine_right=True,
    tick_direction="out",
    primary_color="#2c7fb8",
    color_cycle=("#2c7fb8", "#7fcdbb", "#41ab5d", "#d95f02",
                 "#e7298a", "#66a61e", "#a6d854", "#fdae61"),
    facecolor="white",
    default_format="pdf",
)
