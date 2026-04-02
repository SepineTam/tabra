#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : data_ops.py

import re

import numpy as np


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
        """
        Generate a new variable based on an expression.

        Parameters
        ----------
        var : str
            Name of the variable to create
        expr : str
            Expression to evaluate (supports ^, log, exp, sqrt, etc.)
        replace : bool, optional
            If True, overwrite existing variable. Default False.

        Returns
        -------
        self : DataOps
            Returns self for method chaining
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
        return vars_input

    def drop(self, vars):
        """
        Drop specified columns from the DataFrame.

        Parameters
        ----------
        vars : str or list
            Variable name(s) to drop.

        Returns
        -------
        self : DataOps
            Returns self for method chaining
        """
        vars_list = self._parse_vars(vars)
        for var in vars_list:
            if var not in self._tabra._df.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

        self._tabra._df = self._tabra._df.drop(columns=vars_list)
        return self

    def keep(self, vars):
        """
        Keep only specified columns, dropping all others.

        Parameters
        ----------
        vars : str or list
            Variable name(s) to keep.

        Returns
        -------
        self : DataOps
            Returns self for method chaining
        """
        vars_list = self._parse_vars(vars)
        for var in vars_list:
            if var not in self._tabra._df.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

        cols_to_drop = [col for col in self._tabra._df.columns if col not in vars_list]
        self._tabra._df = self._tabra._df.drop(columns=cols_to_drop)
        return self

    def rename(self, old: str, new: str):
        """
        Rename an existing variable.

        Parameters
        ----------
        old : str
            Existing variable name
        new : str
            New variable name

        Returns
        -------
        self : DataOps
            Returns self for method chaining
        """
        if old not in self._tabra._df.columns:
            raise KeyError(f"Variable '{old}' not found in DataFrame")
        if new in self._tabra._df.columns:
            raise ValueError(f"Variable '{new}' already exists in DataFrame")

        self._tabra._df = self._tabra._df.rename(columns={old: new})
        return self
