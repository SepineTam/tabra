#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Song Tan, Inc. All Rights Reserved
#
# @Author : Song Tan
# @Email  : sepinetam@gmail.com
# @File   : tabra/data/types.py

from enum import Enum


class DataType(Enum):
    ...


class VariableType(Enum):
    string = 'string'
    integer = 'integer'
    float = 'float'
    boolean = 'boolean'
    datetime = 'datetime'
    null = 'null'
