#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : data_ops.py

import re
from typing import Literal

import numpy as np
import pandas as pd


class DataOps:
    """Data manipulation operations for TabraData."""

    def __init__(self, tabra):
        self._tabra = tabra

    @staticmethod
    def _preprocess_expr(expr: str) -> str:
        """Convert Stata-style expressions to pandas-compatible ones."""
        # ^ → ** (exponentiation)
        expr = re.sub(r'\^', '**', expr)
        return expr

    @staticmethod
    def _log(x, base=None):
        """Log function supporting optional base (Stata-style)."""
        if base is None:
            return np.log(x)
        return np.log(x) / np.log(base)

    def _eval(self, expr: str):
        """Evaluate expression with DataFrame columns + numpy math functions."""
        expr = self._preprocess_expr(expr)
        ns = {col: self._tabra._df[col] for col in self._tabra._df.columns}
        ns.update({
            "exp": np.exp, "log": self._log, "log10": np.log10,
            "sqrt": np.sqrt, "abs": np.abs,
        })
        return eval(expr, {"__builtins__": {}}, ns)

    def gen(self, var: str, expr: str, replace: bool = False):
        """Generate a new variable based on an expression.

        Args:
            var (str): Name of the variable to create.
            expr (str): Expression to evaluate (supports ^, log, exp, sqrt, etc.).
            replace (bool): If True, overwrite existing variable. Default False.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> dta = load_data("auto")
            >>> dta.data.gen("mpg_sq", "mpg^2")
        """
        if var in self._tabra._df.columns:
            if not replace:
                raise ValueError(
                    f"Variable '{var}' already exists. "
                    f"Use replace=True to overwrite, "
                    f"or use tab.data.replace() / tab.data.drop() first."
                )

        df = self._tabra._df.copy()
        df[var] = self._eval(expr)
        self._tabra._df = df
        return self

    def replace(self, var: str, expr: str, cond: str = None):
        """
        Replace values in an existing variable based on an expression.

        Parameters
        ----------
        var : str
            Name of the existing variable to modify
        expr : str
            Expression to evaluate
        cond : str, optional
            Condition string for row selection (uses df.query())

        Returns
        -------
        self : DataOps
            Returns self for method chaining
        """
        if var not in self._tabra._df.columns:
            raise KeyError(f"Variable '{var}' not found in DataFrame")

        if cond is None:
            self._tabra._df[var] = self._eval(expr)
        else:
            mask = self._tabra._df.eval(cond)
            ns = {col: self._tabra._df.loc[mask, col] for col in self._tabra._df.columns}
            ns.update({
                "exp": np.exp, "log": self._log, "log10": np.log10,
                "sqrt": np.sqrt, "abs": np.abs,
            })
            self._tabra._df.loc[mask, var] = eval(
                self._preprocess_expr(expr),
                {"__builtins__": {}}, ns,
            )

        return self

    def _parse_vars(self, vars_input):
        """Parse vars input into a list of variable names."""
        if isinstance(vars_input, str):
            return vars_input.split()
        return list(vars_input)

    _EGEN_AGG = {
        "mean": "mean",
        "sum": "sum",
        "total": "sum",
        "max": "max",
        "min": "min",
        "sd": lambda x: x.std(ddof=1),
        "count": "count",
        "median": "median",
    }

    def egen(
        self,
        new_var: str,
        func: Literal[
            "mean", "sum", "total", "max", "min", "sd",
            "count", "median", "rank", "group", "seq",
        ],
        source: str | list[str],
        *,
        by: str | list[str] | None = None,
    ):
        """Extended generate: compute aggregated / transformed variables.

        Args:
            new_var (str): Name of the new variable. Must not already exist.
            func (str): Egen function name.
            source (str or list): Source variable(s).
            by (str or list, optional): Group variable(s).

        Returns:
            DataOps: Returns self for method chaining.
        """
        df = self._tabra._df

        if new_var in df.columns:
            raise ValueError(f"Variable '{new_var}' already exists in DataFrame")

        func = func.lower()
        by_vars = self._parse_vars(by) if by is not None else None

        # --- group: encode combination into integer IDs ---
        if func == "group":
            source_vars = self._parse_vars(source)
            for v in source_vars:
                if v not in df.columns:
                    raise KeyError(f"Variable '{v}' not found in DataFrame")
            result = df.groupby(source_vars).ngroup() + 1
            self._tabra._df[new_var] = result
            return self

        # --- source validation for remaining functions ---
        if isinstance(source, str):
            if source not in df.columns:
                raise KeyError(f"Variable '{source}' not found in DataFrame")
            src_col = df[source]
        else:
            raise TypeError(
                f"source must be a string for func '{func}', "
                f"got {type(source).__name__}"
            )

        if by_vars is not None:
            for v in by_vars:
                if v not in df.columns:
                    raise KeyError(f"Variable '{v}' not found in DataFrame")

        # --- rank ---
        if func == "rank":
            if by_vars is not None:
                result = df.groupby(by_vars)[source].rank(
                    method="average", ascending=True,
                )
            else:
                result = src_col.rank(method="average", ascending=True)
            self._tabra._df[new_var] = result
            return self

        # --- seq ---
        if func == "seq":
            if by_vars is not None:
                result = df.groupby(by_vars).cumcount() + 1
            else:
                result = pd.Series(range(1, len(df) + 1), index=df.index)
            self._tabra._df[new_var] = result
            return self

        # --- statistical aggregations ---
        agg = self._EGEN_AGG.get(func)
        if agg is None:
            raise ValueError(
                f"Unknown egen function '{func}'. "
                f"Supported: {', '.join(sorted(self._EGEN_AGG.keys()))}, "
                f"rank, group, seq"
            )

        if by_vars is not None:
            grouped = df.groupby(by_vars)[source]
            result = grouped.transform(agg)
        else:
            val = agg(src_col) if callable(agg) else src_col.agg(agg)
            result = pd.Series(val, index=df.index)
            # broadcast scalar to full length
            if result.shape[0] == 1 and len(df) > 1:
                result = pd.Series(val, index=df.index)

        self._tabra._df[new_var] = result
        return self

    def drop(self, vars):
        """Drop specified columns from the DataFrame.

        Args:
            vars (str or list): Variable name(s) to drop.

        Returns:
            DataOps: Returns self for method chaining.
        """
        vars_list = self._parse_vars(vars)
        for var in vars_list:
            if var not in self._tabra._df.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

        self._tabra._df = self._tabra._df.drop(columns=vars_list)
        return self

    def keep(self, vars):
        """Keep only specified columns, dropping all others.

        Args:
            vars (str or list): Variable name(s) to keep.

        Returns:
            DataOps: Returns self for method chaining.
        """
        vars_list = self._parse_vars(vars)
        for var in vars_list:
            if var not in self._tabra._df.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

        cols_to_drop = [col for col in self._tabra._df.columns if col not in vars_list]
        self._tabra._df = self._tabra._df.drop(columns=cols_to_drop)
        return self

    def rename(self, old: str, new: str):
        """Rename an existing variable.

        Args:
            old (str): Existing variable name.
            new (str): New variable name.

        Returns:
            DataOps: Returns self for method chaining.
        """
        if old not in self._tabra._df.columns:
            raise KeyError(f"Variable '{old}' not found in DataFrame")
        if new in self._tabra._df.columns:
            raise ValueError(f"Variable '{new}' already exists in DataFrame")

        self._tabra._df = self._tabra._df.rename(columns={old: new})
        return self

    def winsor2(
        self,
        vars,
        *,
        cuts=(1, 99),
        replace: bool = False,
        trim: bool = False,
        by: str = None,
        suffix: str = "_w",
        prefix: str = None,
    ):
        """Winsorize or trim variables at specified percentiles.

        Mimics Stata's winsor2 command.

        Args:
            vars (str or list): Variable name(s) to winsorize.
            cuts (tuple): Lower and upper percentiles. Default (1, 99).
            replace (bool): If True, overwrite existing variables. Default False.
            trim (bool): If True, trim (set to NaN) instead of winsorize. Default False.
            by (str): Group variable for group-wise winsorization.
            suffix (str): Suffix for new variable names. Default "_w".
            prefix (str): Prefix for new variable names. Overrides suffix.

        Returns:
            DataOps: Returns self for method chaining.
        """
        vars_list = self._parse_vars(vars)
        df = self._tabra._df.copy()

        if isinstance(cuts, (int, float)):
            lo_pct, hi_pct = cuts, 100 - cuts
        else:
            lo_pct, hi_pct = cuts
            if len((lo_pct, hi_pct)) != 2:
                raise ValueError(
                    f"cuts must be a single number or a (low, high) pair, "
                    f"got {cuts}"
                )

        if not (0 <= lo_pct < hi_pct <= 100):
            raise ValueError(
                f"Invalid cuts ({lo_pct}, {hi_pct}): "
                f"must satisfy 0 <= low < high <= 100"
            )

        for var in vars_list:
            if var not in df.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

            if replace:
                target = var
            else:
                if prefix is not None:
                    target = f"{prefix}{var}"
                else:
                    target = f"{var}{suffix}"

            if by is not None:
                if by not in df.columns:
                    raise KeyError(f"Group variable '{by}' not found in DataFrame")
                grouped = df.groupby(by)[var]
                lo = grouped.transform(lambda s: np.nanpercentile(s, lo_pct))
                hi = grouped.transform(lambda s: np.nanpercentile(s, hi_pct))
            else:
                lo = np.nanpercentile(df[var], lo_pct)
                hi = np.nanpercentile(df[var], hi_pct)

            col = df[var].copy()
            if trim:
                col = col.where((col >= lo) & (col <= hi))
            else:
                col = col.clip(lower=lo, upper=hi)

            df[target] = col

        self._tabra._df = df
        return self

    def _append(self, other: pd.DataFrame):
        """Concatenate another DataFrame vertically (Stata append)."""
        self._tabra._df = pd.concat(
            [self._tabra._df, other], ignore_index=True,
        )

    def append(self, other):
        """Append rows from another DataFrame or TabraData.

        Args:
            other (pd.DataFrame or TabraData): Data to append (stacked vertically).

        Returns:
            DataOps: Returns self for method chaining.
        """
        from tabra.core.data import TabraData

        if isinstance(other, TabraData):
            other_df = other._df
        elif isinstance(other, pd.DataFrame):
            other_df = other
        else:
            raise TypeError(
                f"Expected pd.DataFrame or TabraData, got {type(other).__name__}"
            )

        self._append(other_df)
        return self

    def sort(self, vars):
        """Sort DataFrame by variable(s) in ascending order.

        Args:
            vars (str or list): Variable name(s) to sort by.

        Returns:
            DataOps: Returns self for method chaining.
        """
        vars_list = self._parse_vars(vars)
        for var in vars_list:
            if var not in self._tabra._df.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

        self._tabra._df = self._tabra._df.sort_values(
            vars_list, ignore_index=True, na_position="last",
        )
        return self

    def gsort(self, spec):
        """Sort with flexible ascending/descending per variable.

        Mimics Stata's gsort: prefix ``+`` (or no prefix) for ascending,
        ``-`` for descending.

        Args:
            spec (str): Sort spec, e.g. ``"-wage +age"`` or ``"wage -age"``.

        Returns:
            DataOps: Returns self for method chaining.
        """
        parts = spec.split()
        cols, ascending = [], []
        for part in parts:
            if part.startswith("-"):
                cols.append(part[1:])
                ascending.append(False)
            elif part.startswith("+"):
                cols.append(part[1:])
                ascending.append(True)
            else:
                cols.append(part)
                ascending.append(True)

        for col in cols:
            if col not in self._tabra._df.columns:
                raise KeyError(f"Variable '{col}' not found in DataFrame")

        self._tabra._df = self._tabra._df.sort_values(
            cols, ascending=ascending, ignore_index=True, na_position="last",
        )
        return self

    def recode(self, var, mapping, *, gen: str = None):
        """Recode values of a variable using a mapping.

        Args:
            var (str): Variable name to recode.
            mapping (dict): Mapping of {old_value: new_value}. Keys can be single
                values or tuples for ranges: {(1, 5): "low"}.
            gen (str): Name for the new variable. If None, overwrite in place.

        Returns:
            DataOps: Returns self for method chaining.
        """
        if var not in self._tabra._df.columns:
            raise KeyError(f"Variable '{var}' not found in DataFrame")

        target = gen if gen is not None else var
        if gen is not None and gen in self._tabra._df.columns and gen != var:
            raise ValueError(f"Variable '{gen}' already exists in DataFrame")

        orig = self._tabra._df[var].copy()
        result = pd.Series(np.nan, index=orig.index, dtype=object)
        mapped = pd.Series(False, index=orig.index)

        for key, val in mapping.items():
            if isinstance(key, tuple):
                lo, hi = key
                mask = (~mapped) & (orig >= lo) & (orig <= hi)
            else:
                mask = (~mapped) & (orig == key)
            result = result.where(~mask, val)
            mapped |= mask

        result = result.where(mapped, orig)

        self._tabra._df[target] = result
        return self
