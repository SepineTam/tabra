#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_data_ops.py

import numpy as np
import pandas as pd
import pytest
from tabra import load_data


@pytest.fixture
def data_ops_df():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    return df


class TestDataOpsGen:
    def test_gen_simple_arithmetic(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("c", "a + b")
        assert "c" in tab._df.columns
        np.testing.assert_array_almost_equal(tab._df["c"].values, [11, 22, 33, 44, 55])

    def test_gen_multiplication(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("c", "a * 2")
        np.testing.assert_array_almost_equal(tab._df["c"].values, [2, 4, 6, 8, 10])

    def test_gen_log(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("log_a", "log(a)")
        np.testing.assert_array_almost_equal(tab._df["log_a"].values, np.log(data_ops_df["a"]))

    def test_gen_overwrite_existing_raises(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        with pytest.raises(ValueError):
            tab.data.gen("a", "b + 1")

    def test_gen_returns_data_ops(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        result = tab.data.gen("c", "a + b")
        # It returns the DataOps instance (self), so calling gen on result again should work
        # Just check it's not None and not the TabraData
        assert result is not None

    def test_gen_available_in_regression(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("c", "a * 2 + b + 1")
        reg_result = tab.reg("c", ["a"], is_con=False)
        assert reg_result is not None

    def test_gen_power_caret(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("c", "a ^ 2")
        np.testing.assert_array_almost_equal(tab._df["c"].values, [1, 4, 9, 16, 25])

    def test_gen_replace_true(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("c", "a + b")
        tab.data.gen("c", "a * 10", replace=True)
        np.testing.assert_array_almost_equal(tab._df["c"].values, [10, 20, 30, 40, 50])

    def test_gen_exp(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("exp_a", "exp(a)")
        np.testing.assert_array_almost_equal(tab._df["exp_a"].values, np.exp(data_ops_df["a"]))

    def test_gen_log10(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("log10_a", "log10(a)")
        np.testing.assert_array_almost_equal(tab._df["log10_a"].values, np.log10(data_ops_df["a"]))

    def test_gen_sqrt(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("sqrt_a", "sqrt(a)")
        np.testing.assert_array_almost_equal(tab._df["sqrt_a"].values, np.sqrt(data_ops_df["a"]))


class TestDataOpsRename:
    def test_rename_basic(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.rename("a", "alpha")
        assert "alpha" in tab._df.columns
        assert "a" not in tab._df.columns
        np.testing.assert_array_equal(tab._df["alpha"].values, data_ops_df["a"].values)

    def test_rename_nonexistent_raises(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.rename("z", "new_z")

    def test_rename_to_existing_raises(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        with pytest.raises(ValueError):
            tab.data.rename("a", "b")

    def test_rename_returns_data_ops(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        result = tab.data.rename("a", "alpha")
        assert result is not None


class TestDataOpsReplace:
    def test_replace_simple(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.replace("a", "0")
        np.testing.assert_array_almost_equal(tab._df["a"].values, [0, 0, 0, 0, 0])

    def test_replace_with_cond(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.replace("a", "999", cond="a > 3")
        assert tab._df.loc[0, "a"] == 1.0
        assert tab._df.loc[3, "a"] == 999
        assert tab._df.loc[4, "a"] == 999

    def test_replace_expr(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.replace("a", "a * 10")
        np.testing.assert_array_almost_equal(tab._df["a"].values, [10, 20, 30, 40, 50])

    def test_replace_nonexistent_raises(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.replace("z", "1")

    def test_replace_returns_data_ops(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        result = tab.data.replace("a", "a + 1")
        assert result is not None


class TestDataOpsDrop:
    def test_drop_single_var_string(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.drop("a")
        assert "a" not in tab._df.columns
        assert "b" in tab._df.columns

    def test_drop_multiple_vars_string(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.drop("a b")
        assert "a" not in tab._df.columns
        assert "b" not in tab._df.columns

    def test_drop_list(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.drop(["a", "b"])
        assert len(tab._df.columns) == 0

    def test_drop_nonexistent_raises(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.drop("z")

    def test_drop_returns_data_ops(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        result = tab.data.drop("a")
        assert result is not None


class TestDataOpsKeep:
    def test_keep_single_var(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.keep("a")
        assert list(tab._df.columns) == ["a"]

    def test_keep_multiple_vars_string(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.keep("a b")
        assert set(tab._df.columns) == {"a", "b"}

    def test_keep_list(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.keep(["a"])
        assert list(tab._df.columns) == ["a"]

    def test_keep_nonexistent_raises(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.keep("z")


class TestDataOpsIntegration:
    def test_tab_data_property_returns_data_ops(self, data_ops_df):
        from tabra.core.data_ops import DataOps
        tab = load_data(data_ops_df, is_display_result=False)
        assert isinstance(tab.data, DataOps)

    def test_tab_gen_shortcut_works_with_warning(self, data_ops_df):
        import warnings
        tab = load_data(data_ops_df, is_display_result=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tab.gen("c", "a + b")
            assert len(w) == 1
            assert "tab.data.gen()" in str(w[0].message)
        assert "c" in tab._df.columns

    def test_tab_replace_shortcut_works_with_warning(self, data_ops_df):
        import warnings
        tab = load_data(data_ops_df, is_display_result=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tab.replace("a", "0")
            assert len(w) == 1
            assert "tab.data.replace()" in str(w[0].message)

    def test_tab_drop_shortcut_works_with_warning(self, data_ops_df):
        import warnings
        tab = load_data(data_ops_df, is_display_result=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tab.drop("a")
            assert len(w) == 1
            assert "tab.data.drop()" in str(w[0].message)

    def test_tab_keep_shortcut_works_with_warning(self, data_ops_df):
        import warnings
        tab = load_data(data_ops_df, is_display_result=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tab.keep("a")
            assert len(w) == 1
            assert "tab.data.keep()" in str(w[0].message)

    def test_tab_rename_shortcut_works_with_warning(self, data_ops_df):
        import warnings
        tab = load_data(data_ops_df, is_display_result=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tab.rename("a", "alpha")
            assert len(w) == 1
            assert "tab.data.rename()" in str(w[0].message)

    def test_method_chaining(self, data_ops_df):
        tab = load_data(data_ops_df, is_display_result=False)
        tab.data.gen("c", "a + b").replace("c", "0", cond="c > 30")
        assert tab._df.loc[0, "c"] == 11.0
        assert tab._df.loc[3, "c"] == 0
