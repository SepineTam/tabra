#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : fig_setting.py

"""Backward-compatible re-exports.

Templates now live in ``tabra.plot.templates``.
PlotKind remains here.
"""

from enum import Enum


class PlotKind(Enum):
    """Supported plot types for tab.plot.mix()."""
    scatter = "scatter"
    line = "line"
    hist = "hist"
    bar = "bar"
    box = "box"
    violin = "violin"
    pie = "pie"
    lfit = "lfit"
    lfitci = "lfitci"
    coefplot = "coefplot"
    kdensity = "kdensity"
