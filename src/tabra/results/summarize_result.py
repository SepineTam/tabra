#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : summarize_result.py

import numpy as np
from tabra.results.base import BaseResult


class SummarizeResult(BaseResult):
    def __init__(self, var_names, obs, mean, std, min_val, max_val,
                 percentiles=None, skewness=None, kurtosis=None,
                 detail=False, **kwargs):
        super().__init__()
        self._var_names = var_names
        self._obs = obs
        self._mean = mean
        self._std = std
        self._min_val = min_val
        self._max_val = max_val
        self._percentiles = percentiles or {}
        self._skewness = skewness or {}
        self._kurtosis = kurtosis or {}
        self._detail = detail

    @property
    def var_names(self):
        return self._var_names

    @property
    def obs(self):
        return self._obs

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def min_val(self):
        return self._min_val

    @property
    def max_val(self):
        return self._max_val

    @property
    def percentiles(self):
        return self._percentiles

    @property
    def skewness(self):
        return self._skewness

    @property
    def kurtosis(self):
        return self._kurtosis

    def _summary_style_stata(self):
        lines = []
        if self._detail:
            return self._summary_detail_stata()

        lines.append(f"{'Variable':>12s} | {'Obs':>8s} {'Mean':>12s} {'Std. dev':>12s} {'Min':>12s} {'Max':>12s}")
        lines.append("-" * 76)

        for i, name in enumerate(self._var_names):
            obs = self._obs[name]
            m = self._mean[name]
            s = self._std[name]
            mn = self._min_val[name]
            mx = self._max_val[name]
            lines.append(
                f"{name:>12s} | {obs:>8d} {m:>12.4g} {s:>12.4g} {mn:>12.4g} {mx:>12.4g}"
            )
            if (i + 1) % 5 == 0 and i < len(self._var_names) - 1:
                lines.append("-" * 76)

        return "\n".join(lines)

    def _summary_detail_stata(self):
        pct_labels = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
        lines = []

        for name in self._var_names:
            p = self._percentiles.get(name, {})
            lines.append(f"{'':>12s} {name}")
            lines.append("-" * 40)
            lines.append(f"{'Percentiles':>12s} {'Smallest':>12s}")
            lines.append(f"{p.get('1%', 0):>12.4g} {p.get('1%', 0):>12.4g}")
            lines.append(f"{p.get('5%', 0):>12.4g} {p.get('5%', 0):>12.4g}")
            lines.append(f"{p.get('10%', 0):>12.4g} {p.get('10%', 0):>12.4g} {'Obs':>20s} {self._obs[name]:>10d}")
            lines.append(f"{p.get('25%', 0):>12.4g} {p.get('25%', 0):>12.4g} {'Sum of Wgt.':>20s} {self._obs[name]:>10d}")
            lines.append("")
            lines.append(f"{p.get('50%', 0):>12.4g} {'Mean':>20s} {self._mean[name]:>12.4g}")
            lines.append(f"{'Largest':>12s} {'Std. dev':>20s} {self._std[name]:>12.4g}")
            lines.append(f"{p.get('75%', 0):>12.4g} {p.get('75%', 0):>12.4g}")
            lines.append(f"{p.get('90%', 0):>12.4g} {p.get('90%', 0):>12.4g} {'Variance':>20s} {self._std[name]**2:>12.4g}")
            lines.append(f"{p.get('95%', 0):>12.4g} {p.get('95%', 0):>12.4g} {'Skewness':>20s} {self._skewness.get(name, 0):>12.4g}")
            lines.append(f"{p.get('99%', 0):>12.4g} {p.get('99%', 0):>12.4g} {'Kurtosis':>20s} {self._kurtosis.get(name, 0):>12.4g}")
            lines.append("")

        return "\n".join(lines)

    def summary(self):
        dispatch = {
            "stata": self._summary_style_stata,
        }
        method = dispatch.get(self._style, self._summary_style_stata)
        return method()

    def save(self, path="summarize_result.txt"):
        with open(path, "w") as f:
            f.write(self.summary())

    def __repr__(self):
        return self.summary()
