#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : errors.py


class PlotError(Exception):
    """Base exception for plot-related errors."""


class NoResultError(PlotError):
    """No estimation result available to plot."""


class ResultTypeError(PlotError):
    """Invalid result type passed to plot function."""


class NoCommonVarsError(PlotError):
    """No common variables found across multiple models."""


class InvalidLevelError(PlotError):
    """Confidence level out of valid range (0, 1)."""
