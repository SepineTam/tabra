#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : cov_result.py

import numpy as np

from tabra.results.base import BaseResult


class CovResult(BaseResult):
    """Result object for covariance matrix computation."""

    def __init__(
        self,
        matrix: np.ndarray,
        var_names: list[str],
        n_obs: int,
    ):
        super().__init__()
        self._matrix = matrix
        self._var_names = var_names
        self._n_obs = n_obs

    @property
    def matrix(self) -> np.ndarray:
        """Covariance matrix (n x n)."""
        return self._matrix

    @property
    def var_names(self) -> list[str]:
        """Variable names in matrix order."""
        return self._var_names

    @property
    def n_obs(self) -> int:
        """Number of observations used (listwise deletion)."""
        return self._n_obs

    def _summary_style_stata(self) -> str:
        lines = []
        n = len(self._var_names)

        col_width = max(max(len(v) for v in self._var_names), 12) + 2
        header = " " * col_width + "|"
        for v in self._var_names:
            header += f"{v:>{col_width}}"
        lines.append(header)
        sep = "-" * (col_width + 1 + col_width * n)
        lines.append(sep)

        for i, name in enumerate(self._var_names):
            row = f"{name:>{col_width}} |"
            for j in range(i + 1):
                row += f"{self._matrix[i, j]:>{col_width}.4f}"
            lines.append(row)

        lines.append(sep)
        return "\n".join(lines)

    def summary(self) -> str:
        dispatch = {
            "stata": self._summary_style_stata,
        }
        fn = dispatch.get(self._style, self._summary_style_stata)
        return fn()

    def save(self, path: str = "cov_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())
