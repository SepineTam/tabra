#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : template.py

"""Convenience re-exports for plot templates."""

from tabra.plot.templates import (
    PlotTemplateBase,
    AER, QJE, JPE, ECONOMETRICA, RES,
    PRESENTATION, DEFAULT,
    TEMPLATES,
)
from tabra.plot.fig_setting import PlotKind

__all__ = [
    "PlotTemplateBase",
    "AER", "QJE", "JPE", "ECONOMETRICA", "RES",
    "PRESENTATION", "DEFAULT",
    "TEMPLATES", "PlotKind",
]
