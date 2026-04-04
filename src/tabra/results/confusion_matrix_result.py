#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : confusion_matrix_result.py

import numpy as np

from tabra.results.base import BaseResult


class ConfusionMatrixResult(BaseResult):
    """Result object for a confusion matrix."""

    def __init__(self, matrix, labels, accuracy, n_obs):
        super().__init__()
        self._matrix = matrix    # np.ndarray, shape (k, k)
        self._labels = labels    # list[str], class labels
        self._accuracy = accuracy
        self._n_obs = n_obs

    @property
    def matrix(self):
        return self._matrix

    @property
    def var_names(self):
        """Alias for labels — heatmap duck-typing relies on this."""
        return self._labels

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def n_obs(self):
        return self._n_obs

    def summary(self):
        k = len(self._labels)
        col_w = max(max(len(str(l)) for l in self._labels), 8) + 2

        lines = []
        header = " " * col_w + "|"
        for l in self._labels:
            header += f"{str(l):>{col_w}}"
        lines.append(header)
        sep = "-" * len(header)
        lines.append(sep)

        for i, row_label in enumerate(self._labels):
            row = f"{str(row_label):>{col_w}} |"
            for j in range(k):
                row += f"{self._matrix[i, j]:>{col_w}d}"
            lines.append(row)

        lines.append(sep)
        lines.append(f"Accuracy: {self._accuracy:.4f}")
        lines.append(f"N = {self._n_obs}")
        return "\n".join(lines)

    def save(self, path="confusion_matrix.txt"):
        with open(path, "w") as f:
            f.write(self.summary())
