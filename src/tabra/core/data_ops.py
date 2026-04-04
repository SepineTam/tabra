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

    def _resolve_vars(self, vars_input) -> list[str]:
        """Resolve vars: exact name first, fallback to regex match."""
        from tabra.utils import resolve_var
        return resolve_var(vars_input, self._tabra._df.columns.tolist())

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
        by_vars = self._resolve_vars(by) if by is not None else None

        # --- group: encode combination into integer IDs ---
        if func == "group":
            source_vars = self._resolve_vars(source)
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
        vars_list = self._resolve_vars(vars)
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
        vars_list = self._resolve_vars(vars)
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
        vars_list = self._resolve_vars(vars)
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

    def reshape_long(
        self,
        stub: str | list[str],
        i: str,
        j: str = "_j",
    ):
        """Reshape from wide to long format (Stata reshape long).

        Finds columns matching ``stub_suffix`` (e.g., ``"wage"`` matches
        ``wage_2020``, ``wage_2021``) and collapses them into long format.

        Args:
            stub (str or list): Variable name prefix(es) to reshape.
                Accepts a string, space-separated string, or list.
            i (str): ID variable that identifies each group.
            j (str): Name for the new variable holding suffixes. Default "_j".

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> tab.data.reshape_long("wage", i="firm", j="year")
            >>> tab.data.reshape_long(["wage", "hours"], i="firm")
        """
        stubs = self._parse_vars(stub)
        df = self._tabra._df

        if i not in df.columns:
            raise KeyError(f"Variable '{i}' not found in DataFrame")

        for s in stubs:
            matching = [c for c in df.columns if c.startswith(s + "_")]
            if not matching:
                raise KeyError(
                    f"No columns found matching stub '{s}_' in DataFrame"
                )

        result = pd.wide_to_long(
            df, stubnames=stubs, i=i, j=j, sep="_", suffix=".*",
        )
        result = result.reset_index()
        result = result.dropna(subset=stubs, how="all")
        result = result.reset_index(drop=True)

        self._tabra._df = result
        return self

    def reshape_wide(
        self,
        stub: str | list[str],
        i: str,
        j: str,
    ):
        """Reshape from long to wide format (Stata reshape wide).

        Pivots long-format data so that each value of ``j`` becomes a new
        column, named ``stub_jvalue``.

        Args:
            stub (str or list): Variable name(s) to pivot into wide columns.
                Accepts a string, space-separated string, or list.
            i (str): ID variable that identifies each group.
            j (str): Variable whose values become column suffixes.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> tab.data.reshape_wide("wage", i="firm", j="year")
            >>> tab.data.reshape_wide(["wage", "hours"], i="firm", j="year")
        """
        stubs = self._parse_vars(stub)
        df = self._tabra._df

        if i not in df.columns:
            raise KeyError(f"Variable '{i}' not found in DataFrame")
        if j not in df.columns:
            raise KeyError(f"Variable '{j}' not found in DataFrame")
        for s in stubs:
            if s not in df.columns:
                raise KeyError(f"Variable '{s}' not found in DataFrame")

        pivoted = df.pivot(index=i, columns=j, values=stubs)
        pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]

        self._tabra._df = pivoted.reset_index()
        return self

    _COLLAPSE_STATS = {
        "mean": "mean",
        "median": "median",
        "sum": "sum",
        "total": "sum",
        "sd": lambda x: x.std(ddof=1),
        "min": "min",
        "max": "max",
        "count": "count",
        "first": "first",
        "last": "last",
        "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
    }

    def _resolve_collapse_stat(self, stat: str):
        """Resolve a stat name (including pXX percentiles) to a callable or string."""
        stat = stat.lower()
        if stat in self._COLLAPSE_STATS:
            return self._COLLAPSE_STATS[stat], stat
        # percentile: p1 .. p99
        if stat.startswith("p") and stat[1:].isdigit():
            pct = int(stat[1:])
            if 1 <= pct <= 99:
                return lambda x, p=pct: x.quantile(p / 100), stat
        raise ValueError(
            f"Unknown stat '{stat}'. Supported: "
            f"{', '.join(sorted(self._COLLAPSE_STATS.keys()))}, p1-p99"
        )

    def collapse(
        self,
        stat: str = "mean",
        vars: str | list[str] = None,
        by: str | list[str] = None,
    ):
        """Aggregate data by groups, replacing the dataset with summary stats.

        Mimics Stata's ``collapse`` command. Each group becomes one row.

        Args:
            stat (str): Statistic to compute. One of "mean", "median", "sum",
                "total", "sd", "min", "max", "count", "first", "last", "iqr",
                or "p1"–"p99". Default "mean".
            vars (str or list): Variables to aggregate. None aggregates all
                numeric columns.
            by (str or list): Grouping variable(s). None collapses the entire
                dataset into one row.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> tab.data.collapse("mean", vars="wage age", by="industry")
            >>> tab.data.collapse("median", by=["state", "year"])
            >>> tab.data.collapse("p25", vars="wage", by="industry")
        """
        df = self._tabra._df

        agg_func, _ = self._resolve_collapse_stat(stat)

        # Parse vars
        if vars is not None:
            agg_vars = self._resolve_vars(vars)
        else:
            agg_vars = df.select_dtypes(include="number").columns.tolist()

        # Validate vars exist
        for v in agg_vars:
            if v not in df.columns:
                raise KeyError(f"Variable '{v}' not found in DataFrame")

        # Parse by
        by_vars = self._resolve_vars(by) if by is not None else None
        if by_vars is not None:
            for v in by_vars:
                if v not in df.columns:
                    raise KeyError(f"Variable '{v}' not found in DataFrame")

        # Build aggregation dict
        agg_dict = {v: agg_func for v in agg_vars}

        if by_vars is not None:
            grouped = df.groupby(by_vars)
            result = grouped.agg(agg_dict)
        else:
            result = df.agg(agg_dict).to_frame().T

        result = result.reset_index()
        # Remove any extra index columns from groupby
        if by_vars is not None:
            # Keep only by_vars + agg_vars
            keep = [c for c in result.columns if c in by_vars or c in agg_vars]
            result = result[keep]

        self._tabra._df = result
        return self

    def duplicates(
        self,
        cmd: str,
        vars: str | list[str] = None,
        gen: str = None,
    ):
        """Report, tag, or drop duplicate observations (Stata duplicates).

        Args:
            cmd (str): Sub-command. One of "report", "examples", "list",
                "tag", "drop".
            vars (str or list, optional): Variables to check for duplicates.
                None checks all columns. Accepts string, space-separated
                string, or list.
            gen (str, optional): Name for the new duplicate-tag variable.
                Required when cmd="tag".

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> tab.data.duplicates("report")
            >>> tab.data.duplicates("tag", vars="id", gen="dup")
            >>> tab.data.duplicates("drop", vars="id")
        """
        cmd = cmd.lower()
        valid_cmds = {"report", "examples", "list", "tag", "drop"}
        if cmd not in valid_cmds:
            raise ValueError(
                f"cmd must be one of {sorted(valid_cmds)}, got '{cmd}'"
            )

        df = self._tabra._df
        check_cols = self._resolve_vars(vars) if vars is not None else list(df.columns)

        for v in check_cols:
            if v not in df.columns:
                raise KeyError(f"Variable '{v}' not found in DataFrame")

        # Find duplicate mask
        dup_mask = df.duplicated(subset=check_cols, keep=False)
        has_dup = df.duplicated(subset=check_cols, keep="first")

        if cmd == "report":
            n_total = len(df)
            n_unique = (~dup_mask).sum()
            n_dup_groups = dup_mask.sum() - has_dup.sum()  # unique groups that have dups
            grouped = df.groupby(check_cols).size()
            groups_with_dups = (grouped > 1).sum()
            n_excess = has_dup.sum()
            print(f"Duplicates report")
            print(f"  Observations:     {n_total}")
            print(f"  Unique obs:       {n_unique}")
            print(f"  Duplicate groups: {groups_with_dups}")
            print(f"  Excess obs:       {n_excess}")

        elif cmd == "examples":
            dup_df = df[dup_mask]
            if dup_df.empty:
                print("No duplicates found.")
            else:
                # One representative per duplicate group
                examples = dup_df.drop_duplicates(subset=check_cols, keep="first")
                print(examples.to_string(index=True))

        elif cmd == "list":
            dup_df = df[dup_mask]
            if dup_df.empty:
                print("No duplicates found.")
            else:
                print(dup_df.to_string(index=True))

        elif cmd == "tag":
            if gen is None:
                raise ValueError(
                    "gen parameter is required for cmd='tag'. "
                    "Usage: tab.data.duplicates('tag', gen='dup')"
                )
            if gen in df.columns:
                raise ValueError(f"Variable '{gen}' already exists in DataFrame")
            # tag = group_size - 1; 0 means unique
            group_sizes = df.groupby(check_cols)[check_cols[0]].transform("count")
            self._tabra._df[gen] = group_sizes - 1

        elif cmd == "drop":
            self._tabra._df = df.drop_duplicates(
                subset=check_cols, keep="first"
            ).reset_index(drop=True)

        return self

    def merge(
        self,
        right,
        key: str | list[str],
        *,
        local_key: str | list[str] = None,
        merge_type: str = "1:1",
        varlist: str | list[str] = None,
        gen: bool = True,
        replace: bool = False,
        assert_uniqueness: bool = False,
    ):
        """Merge another dataset horizontally (Stata merge).

        Args:
            right (pd.DataFrame or TabraData): Right (using) dataset.
            key (str or list): Key column name(s) in the right table.
            local_key (str or list, optional): Key column name(s) in the left
                (master) table. Defaults to ``key``.
            merge_type (str): One of "1:1", "m:1", "1:m". Only affects
                uniqueness validation. Default "1:1".
            varlist (str or list, optional): Right-table variables to merge.
                Default None merges all columns.
            gen (bool): If True, add a ``_merge`` indicator column. Default True.
            replace (bool): If True, overwrite conflicting columns with right
                values. Default False raises on conflict.
            assert_uniqueness (bool): If True, validate key uniqueness per
                merge_type. Default False.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> tab.data.merge(right_tab, key="id")
            >>> tab.data.merge(industry_tab, key="ind", local_key="industry",
            ...                merge_type="m:1", assert_uniqueness=True)
        """
        from tabra.core.data import TabraData

        if isinstance(right, TabraData):
            right_df = right._df.copy()
        elif isinstance(right, pd.DataFrame):
            right_df = right.copy()
        else:
            raise TypeError(
                f"Expected pd.DataFrame or TabraData, got {type(right).__name__}"
            )

        left_df = self._tabra._df

        # Parse keys
        right_keys = self._parse_vars(key)
        left_keys = self._parse_vars(local_key) if local_key is not None else right_keys

        if len(left_keys) != len(right_keys):
            raise ValueError(
                f"local_key ({len(left_keys)} keys) and key "
                f"({len(right_keys)} keys) must have the same length"
            )

        # Validate keys exist
        for lk, rk in zip(left_keys, right_keys):
            if lk not in left_df.columns:
                raise KeyError(f"Left key '{lk}' not found in DataFrame")
            if rk not in right_df.columns:
                raise KeyError(f"Right key '{rk}' not found in right DataFrame")

        # Uniqueness validation
        if assert_uniqueness:
            merge_type = merge_type.lower()
            if merge_type == "1:1":
                if left_df.duplicated(subset=left_keys).any():
                    raise ValueError(
                        f"Left key {left_keys} has duplicates; "
                        f"merge_type='1:1' requires unique keys on both sides"
                    )
                if right_df.duplicated(subset=right_keys).any():
                    raise ValueError(
                        f"Right key {right_keys} has duplicates; "
                        f"merge_type='1:1' requires unique keys on both sides"
                    )
            elif merge_type == "m:1":
                if right_df.duplicated(subset=right_keys).any():
                    raise ValueError(
                        f"Right key {right_keys} has duplicates; "
                        f"merge_type='m:1' requires unique keys on right side"
                    )
            elif merge_type == "1:m":
                if left_df.duplicated(subset=left_keys).any():
                    raise ValueError(
                        f"Left key {left_keys} has duplicates; "
                        f"merge_type='1:m' requires unique keys on left side"
                    )
            else:
                raise ValueError(
                    f"merge_type must be '1:1', 'm:1', or '1:m', "
                    f"got '{merge_type}'"
                )

        # Filter right columns via varlist
        if varlist is not None:
            keep_cols = self._parse_vars(varlist)
            for v in keep_cols:
                if v not in right_df.columns:
                    raise KeyError(f"Variable '{v}' not found in right DataFrame")
            # Always keep key columns
            right_df = right_df[right_keys + [c for c in keep_cols if c not in right_keys]]

        # Detect conflicting columns (non-key columns that exist in both)
        conflict_cols = [
            c for c in right_df.columns
            if c not in right_keys and c in left_df.columns
        ]
        if conflict_cols and not replace:
            raise ValueError(
                f"Conflicting columns {conflict_cols} exist in both tables. "
                f"Use replace=True to overwrite with right values, "
                f"or use varlist to exclude them."
            )

        # Rename right keys to match left keys for merge
        rename_map = {rk: lk for rk, lk in zip(right_keys, left_keys) if rk != lk}
        if rename_map:
            right_df = right_df.rename(columns=rename_map)

        merge_keys = left_keys  # after rename, both sides use left key names

        # Perform outer merge
        result = pd.merge(
            left_df, right_df,
            on=merge_keys,
            how="outer",
            suffixes=("", "_right"),
            indicator="_merge_raw" if gen else False,
        )

        # Handle conflicts: if replace, overwrite with right values
        if conflict_cols and replace:
            for col in conflict_cols:
                right_col = f"{col}_right"
                if right_col in result.columns:
                    result[col] = result[right_col]
                    result = result.drop(columns=[right_col])

        # Clean up any remaining _right columns
        right_suffix_cols = [c for c in result.columns if c.endswith("_right")]
        if right_suffix_cols:
            result = result.drop(columns=right_suffix_cols)

        # Convert _merge indicator to Stata-style labels
        if gen:
            label_map = {
                "left_only": "left_only",
                "right_only": "right_only",
                "both": "both",
            }
            result["_merge"] = result["_merge_raw"].map(label_map)
            result = result.drop(columns=["_merge_raw"])

        self._tabra._df = result
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
        vars_list = self._resolve_vars(vars)
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

    def list_var(self) -> list[str]:
        """Return all variable (column) names in the DataFrame.

        Returns:
            list[str]: List of column names.

        Example:
            >>> dta = load_data("auto")
            >>> dta.data.list_var()
            ['make', 'price', 'mpg', 'rep78', 'headroom', 'trunk', 'weight', 'length', 'turn', 'displacement', 'gear_ratio', 'foreign']
        """
        return self._tabra._df.columns.tolist()

    def head(self, vars=None, lines: int = 5):
        """Print the first N rows of selected variables.

        Args:
            vars (str or list, optional): Variable name(s) to display.
                None displays all columns. Default None.
            lines (int): Number of rows to print. Default 5.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> dta = load_data("auto")
            >>> dta.data.head("price mpg", lines=3)
            >>> dta.data.head(lines=10)
        """
        df = self._tabra._df
        if vars is not None:
            vars_list = self._resolve_vars(vars)
            df = df[vars_list]
        print(df.head(lines).to_string())
        return self

    def search(self, pattern: str) -> list[str]:
        """Search variable names by regex pattern.

        Args:
            pattern (str): Regular expression pattern to match against
                column names.

        Returns:
            list[str]: List of matching column names.

        Example:
            >>> dta.data.search("^year")
            ['year', 'year_birth']
            >>> dta.data.search("price|wage")
            ['price', 'wage']
        """
        import re
        return [c for c in self._tabra._df.columns if re.search(pattern, c)]

    def corr(
        self,
        var_list: list[str] | str = None,
        *,
        method: str = "pearson",
    ):
        """Compute correlation matrix (Stata correlate / pwcorr).

        Args:
            var_list (str or list, optional): Variable name(s) or regex.
                None uses all numeric columns. Default None.
            method (str): Correlation method. One of "pearson",
                "spearman", "kendall". Default "pearson".

        Returns:
            CorrResult: Result object with matrix, var_names, method, n_obs.

        Example:
            >>> dta.data.corr(["price", "mpg", "weight"])
            >>> dta.data.corr("^price|^mpg", method="spearman")
        """
        from tabra.results.corr_result import CorrResult

        df = self._tabra._df

        if var_list is not None:
            cols = self._resolve_vars(var_list)
        else:
            cols = (df.select_dtypes(include=["number", "category"])
                     .columns.tolist())

        sub = df[cols].copy()
        for c in sub.select_dtypes(include="category").columns:
            sub[c] = sub[c].cat.codes.astype(float)
            sub.loc[sub[c] == -1, c] = np.nan
        sub = sub.dropna()
        n_obs = len(sub)

        corr_matrix = sub.corr(method=method).values

        result = CorrResult(
            matrix=corr_matrix,
            var_names=cols,
            method=method,
            n_obs=n_obs,
        )
        result.set_style(self._tabra._style if hasattr(self._tabra, '_style') else "stata")
        self._tabra._result = result

        if getattr(self._tabra, '_is_display_result', True):
            print(result.summary())

        return result

    def cov(
        self,
        var_list: list[str] | str = None,
    ):
        """Compute covariance matrix (Stata correlate, covariance).

        Args:
            var_list (str or list, optional): Variable name(s) or regex.
                None uses all numeric columns. Default None.

        Returns:
            CovResult: Result object with matrix, var_names, n_obs.

        Example:
            >>> dta.data.cov(["price", "mpg", "weight"])
            >>> dta.data.cov()
        """
        from tabra.results.cov_result import CovResult

        df = self._tabra._df

        if var_list is not None:
            cols = self._resolve_vars(var_list)
        else:
            cols = (df.select_dtypes(include=["number", "category"])
                     .columns.tolist())

        sub = df[cols].copy()
        for c in sub.select_dtypes(include="category").columns:
            sub[c] = sub[c].cat.codes.astype(float)
            sub.loc[sub[c] == -1, c] = np.nan
        sub = sub.dropna()
        n_obs = len(sub)

        cov_matrix = sub.cov().values

        result = CovResult(
            matrix=cov_matrix,
            var_names=cols,
            n_obs=n_obs,
        )
        result.set_style(self._tabra._style if hasattr(self._tabra, '_style') else "stata")
        self._tabra._result = result

        if getattr(self._tabra, '_is_display_result', True):
            print(result.summary())

        return result

    def describe(self, vars: list[str] | str = None):
        """Print variable overview (Stata describe).

        Shows variable name, dtype, non-missing count, and missing count.

        Args:
            vars (str or list, optional): Variable name(s) to describe.
                Supports space-separated names or regex pattern.
                None describes all columns. Default None.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> dta.data.describe()
            >>> dta.data.describe("price mpg rep78")
            >>> dta.data.describe("^r")
        """
        import re
        df = self._tabra._df
        if vars is not None:
            vars_list = self._resolve_vars(vars)
            df = df[vars_list]

        rows = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_missing = int(df[col].notna().sum())
            missing = int(df[col].isna().sum())
            rows.append({
                "variable": col,
                "dtype": dtype,
                "non_missing": non_missing,
                "missing": missing,
            })
        result = pd.DataFrame(rows)
        n_obs = len(self._tabra._df)
        print(f"Observations: {n_obs}")
        print(f"Variables:    {len(df.columns)}")
        print(result.to_string(index=False))
        return self

    def tabulate(
        self,
        var: str,
        by: str = None,
        *,
        sort: bool = False,
    ):
        """Frequency table for one variable, or crosstab for two (Stata tabulate).

        Args:
            var (str): Variable to tabulate.
            by (str, optional): Second variable for crosstab. Default None.
            sort (bool): Sort by frequency descending. Default False.

        Returns:
            DataOps: Returns self for method chaining.

        Example:
            >>> dta.data.tabulate("foreign")
            >>> dta.data.tabulate("rep78", sort=True)
            >>> dta.data.tabulate("foreign", by="rep78")
        """
        df = self._tabra._df
        if var not in df.columns:
            raise KeyError(f"Variable '{var}' not found in DataFrame")

        if by is None:
            # --- one-way tabulation ---
            counts = df[var].value_counts(dropna=False, sort=sort)
            total = int(counts.sum())
            rows = []
            cum = 0
            for val, freq in counts.items():
                pct = freq / total * 100
                cum += pct
                label = str(val) if pd.notna(val) else "."
                rows.append({
                    var: label,
                    "Freq.": freq,
                    "Percent": round(pct, 2),
                    "Cum.": round(cum, 2),
                })
            result = pd.DataFrame(rows)
            print(result.to_string(index=False))
            print(f"{'─' * 40}")
            print(f"{'Total':>{len(var)}} | {total:>5}   100.00")
        else:
            # --- two-way crosstab ---
            from tabra.results.crosstab_result import CrosstabResult

            if by not in df.columns:
                raise KeyError(f"Variable '{by}' not found in DataFrame")
            ct = pd.crosstab(df[var], df[by], dropna=False)
            row_labels = [str(v) for v in ct.index]
            col_labels = [str(v) for v in ct.columns]
            matrix = ct.values.astype(float)
            row_totals = matrix.sum(axis=1)
            col_totals = matrix.sum(axis=0)
            grand_total = int(matrix.sum())

            result = CrosstabResult(
                matrix=matrix,
                row_labels=row_labels,
                col_labels=col_labels,
                row_var=var,
                col_var=by,
                row_totals=row_totals,
                col_totals=col_totals,
                grand_total=grand_total,
            )
            result.set_style(self._tabra._style if hasattr(self._tabra, '_style') else "stata")
            self._tabra._result = result

            if getattr(self._tabra, '_is_display_result', True):
                print(result.summary())

            return result

    def xttrans(
        self,
        var: str,
        *,
        id: str = None,
        time: str = None,
    ):
        """Compute transition probability matrix (Stata xttrans).

        Requires panel settings from ``xeset()`` unless ``id`` and ``time``
        are passed explicitly.

        Args:
            var (str): State variable to compute transitions for.
            id (str, optional): Panel identifier. Defaults to xeset value.
            time (str, optional): Time variable. Defaults to xeset value.

        Returns:
            XttransResult: Result with count_matrix, prob_matrix, state_labels.

        Example:
            >>> dta.xeset("idcode", "year")
            >>> dta.data.xttrans("status")
        """
        from tabra.results.xttrans_result import XttransResult

        df = self._tabra._df
        if var not in df.columns:
            raise KeyError(f"Variable '{var}' not found in DataFrame")

        # Resolve id and time from xeset if not provided
        id_var = id or getattr(self._tabra, '_panel_var', None)
        time_var = time or getattr(self._tabra, '_time_var', None)

        if id_var is None:
            raise ValueError(
                "Panel id not set. Call dta.xeset() first, "
                "or pass id= and time= explicitly."
            )
        if time_var is None:
            raise ValueError(
                "Time variable not set. Call dta.xeset() first, "
                "or pass id= and time= explicitly."
            )

        for v in [id_var, time_var]:
            if v not in df.columns:
                raise KeyError(f"Variable '{v}' not found in DataFrame")

        # Sort by (id, time) and generate lag within each id group
        df_sorted = df[[id_var, time_var, var]].copy()
        df_sorted = df_sorted.sort_values([id_var, time_var])
        df_sorted['_lag'] = df_sorted.groupby(id_var)[var].shift(1)

        # Detect minimum time step and drop rows where gap != step
        df_sorted['_gap'] = df_sorted.groupby(id_var)[time_var].diff()
        min_step = df_sorted['_gap'].dropna().min()
        if pd.notna(min_step):
            valid = df_sorted['_gap'].eq(min_step) & df_sorted['_lag'].notna()
        else:
            valid = df_sorted['_lag'].notna()

        transitions = df_sorted.loc[valid, ['_lag', var]].dropna()

        if transitions.empty:
            raise ValueError("No valid transitions found in the data")

        # Build count matrix via crosstab
        ct = pd.crosstab(transitions['_lag'], transitions[var])
        all_states = sorted(set(ct.index.tolist() + ct.columns.tolist()))

        # Ensure square matrix with consistent order
        ct = ct.reindex(index=all_states, columns=all_states, fill_value=0)
        count_matrix = ct.values.astype(float)

        # Row-normalize to get probabilities
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        prob_matrix = count_matrix / row_sums

        state_labels = [str(s) for s in all_states]
        n_transitions = int(count_matrix.sum())

        result = XttransResult(
            count_matrix=count_matrix,
            prob_matrix=prob_matrix,
            state_labels=state_labels,
            var=var,
            n_obs=len(df_sorted),
            n_transitions=n_transitions,
        )
        result.set_style(self._tabra._style if hasattr(self._tabra, '_style') else "stata")
        self._tabra._result = result

        if getattr(self._tabra, '_is_display_result', True):
            print(result.summary())

        return result
