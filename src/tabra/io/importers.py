#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : importers.py

from tabra.core.data import TabraData


def load_data(df, *, style="stata", is_display_result=True):
    """Load a DataFrame into a TabraData instance.

    Args:
        df (pd.DataFrame): Input DataFrame.
        style (str): Output display style. Default "stata".
        is_display_result (bool): Whether to print estimation results immediately. Default True.

    Returns:
        TabraData: A wrapped data object.

    Example:
        >>> import pandas as pd
        >>> dta = load_data(pd.DataFrame({"x": [1, 2, 3]}))
    """
    return TabraData(df, style=style, is_display_result=is_display_result)
