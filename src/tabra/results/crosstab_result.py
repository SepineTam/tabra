#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : crosstab_result.py

import numpy as np
import pandas as pd

from tabra.results.base import BaseResult


class CrosstabResult(BaseResult):
    """Result object for two-way cross-tabulation."""

    def __init__(
        self,
        matrix: np.ndarray,
        row_labels: list,
        col_labels: list,
        row_var: str,
        col_var: str,
        row_totals: np.ndarray,
        col_totals: np.ndarray,
        grand_total: int,
    ):
        super().__init__()
        self._matrix = matrix
        self._row_labels = row_labels
        self._col_labels = col_labels
        self._row_var = row_var
        self._col_var = col_var
        self._row_totals = row_totals
        self._col_totals = col_totals
        self._grand_total = grand_total

    @property
    def matrix(self) -> np.ndarray:
        """Cross-tabulation count matrix (rows x cols)."""
        return self._matrix

    @property
    def row_labels(self) -> list:
        """Row labels (values of the row variable)."""
        return self._row_labels

    @property
    def col_labels(self) -> list:
        """Column labels (values of the column variable)."""
        return self._col_labels

    @property
    def row_var(self) -> str:
        """Name of the row variable."""
        return self._row_var

    @property
    def col_var(self) -> str:
        """Name of the column variable."""
        return self._col_var

    @property
    def row_totals(self) -> np.ndarray:
        """Row marginal totals."""
        return self._row_totals

    @property
    def col_totals(self) -> np.ndarray:
        """Column marginal totals."""
        return self._col_totals

    @property
    def grand_total(self) -> int:
        """Grand total (sum of all cells)."""
        return self._grand_total

    @property
    def row_percent(self) -> np.ndarray:
        """Row percentages: each cell / row total."""
        return self._matrix / self._row_totals[:, None] * 100

    @property
    def col_percent(self) -> np.ndarray:
        """Column percentages: each cell / column total."""
        return self._matrix / self._col_totals[None, :] * 100

    @property
    def cell_percent(self) -> np.ndarray:
        """Cell percentages: each cell / grand total."""
        return self._matrix / self._grand_total * 100

    def _summary_style_stata(self) -> str:
        lines = []

        col_width = max(max(len(str(l)) for l in self._col_labels), 8) + 2
        row_label_width = max(max(len(str(l)) for l in self._row_labels), len(self._row_var)) + 2

        # Header
        header = " " * row_label_width + "|"
        for label in self._col_labels:
            header += f"{str(label):>{col_width}}"
        header += f"{'Total':>{col_width}}"
        lines.append(header)
        sep = "-" * (row_label_width + 1 + col_width * (len(self._col_labels) + 1))
        lines.append(sep)

        # Rows
        for i, label in enumerate(self._row_labels):
            row = f"{str(label):>{row_label_width}} |"
            for j in range(len(self._col_labels)):
                row += f"{int(self._matrix[i, j]):>{col_width}}"
            row += f"{int(self._row_totals[i]):>{col_width}}"
            lines.append(row)

        lines.append(sep)

        # Total row
        total_row = f"{'Total':>{row_label_width}} |"
        for j in range(len(self._col_labels)):
            total_row += f"{int(self._col_totals[j]):>{col_width}}"
        total_row += f"{self._grand_total:>{col_width}}"
        lines.append(total_row)

        return "\n".join(lines)

    def summary(self) -> str:
        dispatch = {
            "stata": self._summary_style_stata,
        }
        fn = dispatch.get(self._style, self._summary_style_stata)
        return fn()

    def save(self, path: str = "crosstab_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())
