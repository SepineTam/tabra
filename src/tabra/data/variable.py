#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Song Tan, Inc. All Rights Reserved
#
# @Author : Song Tan
# @Email  : sepinetam@gmail.com
# @File   : tabra/data/variable.py

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .types import VariableType


@dataclass
class VariableMetadata:
    name: str
    label: str | None
    dtype: VariableType
    value_label_mask: dict[str, Any] | None = None

@dataclass
class Variable:
    name: str
    values: pd.Series
    dtype: VariableType  # 这列数据的类型，string > float > int > bool > null
    label: str | None = None
    value_label_mask: dict[str, Any] | None = None

    @property
    def metadata(self) -> VariableMetadata:
        return VariableMetadata(
            name=self.name,
            label=self.label,
            dtype=self.dtype,
            value_label_mask=self.value_label_mask,
        )

    def to_numpy(self) -> np.ndarray:
        return self.values.to_numpy()

    def display(self, n: int = 5) -> str:
        return self.mask().head(n).__repr__()

    def mask(self) -> pd.Series:
        values = self.values.copy()
        if self.value_label_mask is None:
            return values

        for value in values:
            if value in self.value_label_mask.keys():
                values[value] = self.value_label_mask[value]
        return values
