#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : xttrans_result.py

import numpy as np

from tabra.results.base import BaseResult


class XttransResult(BaseResult):
    """Result object for transition probability matrix (Stata xttrans)."""

    def __init__(
        self,
        count_matrix: np.ndarray,
        prob_matrix: np.ndarray,
        state_labels: list,
        var: str,
        n_obs: int,
        n_transitions: int,
    ):
        super().__init__()
        self._count_matrix = count_matrix
        self._prob_matrix = prob_matrix
        self._state_labels = state_labels
        self._var = var
        self._n_obs = n_obs
        self._n_transitions = n_transitions

    @property
    def matrix(self) -> np.ndarray:
        """Transition probability matrix (rows sum to 1)."""
        return self._prob_matrix

    @property
    def count_matrix(self) -> np.ndarray:
        """Transition count matrix."""
        return self._count_matrix

    @property
    def prob_matrix(self) -> np.ndarray:
        """Transition probability matrix (rows sum to 1)."""
        return self._prob_matrix

    @property
    def state_labels(self) -> list:
        """State labels in matrix order."""
        return self._state_labels

    @property
    def var_names(self) -> list:
        """Alias for state_labels (duck-type compatible with heatmap)."""
        return self._state_labels

    @property
    def var(self) -> str:
        """Name of the state variable."""
        return self._var

    @property
    def n_obs(self) -> int:
        """Number of observations used."""
        return self._n_obs

    @property
    def n_transitions(self) -> int:
        """Total number of valid transitions."""
        return self._n_transitions

    def _summary_style_stata(self) -> str:
        lines = []
        n = len(self._state_labels)
        col_width = max(max(len(str(l)) for l in self._state_labels), 10) + 2
        label_width = max(max(len(str(l)) for l in self._state_labels), len(self._var)) + 2

        # Header
        header = " " * label_width + "|"
        for label in self._state_labels:
            header += f"{str(label):>{col_width}}"
        header += f"{'Total':>{col_width}}"
        lines.append(header)
        sep = "-" * (label_width + 1 + col_width * (n + 1))
        lines.append(sep)

        # Rows with probabilities
        for i, label in enumerate(self._state_labels):
            row = f"{str(label):>{label_width}} |"
            for j in range(n):
                row += f"{self._prob_matrix[i, j]:>{col_width}.4f}"
            row_total = self._count_matrix[i].sum()
            row += f"{int(row_total):>{col_width}}"
            lines.append(row)

        lines.append(sep)

        # Total row
        total_row = f"{'Total':>{label_width}} |"
        for j in range(n):
            col_total = self._count_matrix[:, j].sum()
            total_row += f"{int(col_total):>{col_width}}"
        total_row += f"{self._n_transitions:>{col_width}}"
        lines.append(total_row)

        lines.append(f"\nTotal transitions: {self._n_transitions}")

        return "\n".join(lines)

    def summary(self) -> str:
        dispatch = {
            "stata": self._summary_style_stata,
        }
        fn = dispatch.get(self._style, self._summary_style_stata)
        return fn()

    def save(self, path: str = "xttrans_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())
