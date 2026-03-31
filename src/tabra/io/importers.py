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
    return TabraData(df, style=style, is_display_result=is_display_result)
