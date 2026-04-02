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


class TestDataOpsWinsor2:
    @pytest.fixture
    def outlier_df(self):
        np.random.seed(42)
        x = np.random.normal(50, 10, 100).tolist()
        x[0] = 999.0  # extreme upper
        x[1] = -999.0  # extreme lower
        return pd.DataFrame({"x": x, "group": ["A"] * 50 + ["B"] * 50})

    def test_winsor_basic(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x")
        assert "x_w" in tab._df.columns
        assert tab._df["x_w"].max() < 999.0
        assert tab._df["x_w"].min() > -999.0

    def test_winsor_replace(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x", replace=True)
        assert "x_w" not in tab._df.columns
        assert tab._df["x"].max() < 999.0

    def test_winsor_trim(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x", trim=True)
        assert "x_w" in tab._df.columns
        assert tab._df["x_w"].isna().sum() == 2

    def test_winsor_custom_cuts(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x", cuts=(5, 95))
        assert "x_w" in tab._df.columns
        assert tab._df["x_w"].max() < 999.0

    def test_winsor_by_group(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x", by="group")
        assert "x_w" in tab._df.columns
        assert tab._df["x_w"].max() < 999.0

    def test_winsor_suffix(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x", suffix="_win")
        assert "x_win" in tab._df.columns

    def test_winsor_prefix(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x", prefix="w_")
        assert "w_x" in tab._df.columns

    def test_winsor_multiple_vars(self, outlier_df):
        outlier_df["y"] = outlier_df["x"] * 2
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2(["x", "y"])
        assert "x_w" in tab._df.columns
        assert "y_w" in tab._df.columns

    def test_winsor_invalid_cuts_raises(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        with pytest.raises(ValueError):
            tab.data.winsor2("x", cuts=(99, 1))
        with pytest.raises(ValueError):
            tab.data.winsor2("x", cuts=(-1, 99))

    def test_winsor_nonexistent_var_raises(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.winsor2("z")

    def test_winsor_returns_data_ops(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        result = tab.data.winsor2("x")
        assert result is not None

    def test_winsor_chaining(self, outlier_df):
        tab = load_data(outlier_df, is_display_result=False)
        tab.data.winsor2("x").drop("x").rename("x_w", "x")
        assert "x" in tab._df.columns
        assert "x_w" not in tab._df.columns


class TestDataOpsAppend:
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_append_dataframe(self, base_df):
        tab = load_data(base_df, is_display_result=False)
        df2 = pd.DataFrame({"x": [7, 8], "y": [9, 10]})
        tab.data.append(df2)
        assert len(tab._df) == 5

    def test_append_tabradata(self, base_df):
        tab1 = load_data(base_df, is_display_result=False)
        tab2 = load_data(
            pd.DataFrame({"x": [7], "y": [8]}), is_display_result=False,
        )
        tab1.data.append(tab2)
        assert len(tab1._df) == 4

    def test_append_returns_data_ops(self, base_df):
        tab = load_data(base_df, is_display_result=False)
        result = tab.data.append(pd.DataFrame({"x": [0], "y": [0]}))
        assert result is not None

    def test_append_invalid_type_raises(self, base_df):
        tab = load_data(base_df, is_display_result=False)
        with pytest.raises(TypeError):
            tab.data.append("not a dataframe")

    def test_append_different_columns(self, base_df):
        tab = load_data(base_df, is_display_result=False)
        df2 = pd.DataFrame({"x": [7], "z": [99]})
        tab.data.append(df2)
        assert len(tab._df) == 4
        assert "z" in tab._df.columns

    def test_add_operator_dataframes(self, base_df):
        tab1 = load_data(base_df, is_display_result=False)
        tab2 = load_data(
            pd.DataFrame({"x": [7, 8], "y": [9, 10]}), is_display_result=False,
        )
        combined = tab1 + tab2
        assert len(combined._df) == 5
        # original unchanged
        assert len(tab1._df) == 3

    def test_add_operator_tabradata(self, base_df):
        tab1 = load_data(base_df, is_display_result=False)
        tab2 = load_data(
            pd.DataFrame({"x": [7], "y": [8]}), is_display_result=False,
        )
        combined = tab1 + tab2
        assert len(combined._df) == 4

    def test_add_operator_preserves_style(self, base_df):
        tab1 = load_data(base_df, style="stata", is_display_result=False)
        tab2 = load_data(
            pd.DataFrame({"x": [7], "y": [8]}), is_display_result=False,
        )
        combined = tab1 + tab2
        assert combined._style == "stata"

    def test_add_operator_with_df(self, base_df):
        tab1 = load_data(base_df, is_display_result=False)
        df2 = pd.DataFrame({"x": [7], "y": [8]})
        combined = tab1 + df2
        assert len(combined._df) == 4

    def test_append_missing_columns_both_sides(self):
        """df1 有 z 没 w，df2 有 w 没 z → 合并后两列都有 NaN"""
        df1 = pd.DataFrame({"x": [1], "z": [10]})
        df2 = pd.DataFrame({"x": [2], "w": [20]})
        tab = load_data(df1, is_display_result=False)
        tab.data.append(df2)
        assert len(tab._df) == 2
        assert "z" in tab._df.columns
        assert "w" in tab._df.columns
        assert tab._df.loc[0, "z"] == 10
        assert pd.isna(tab._df.loc[1, "z"])
        assert pd.isna(tab._df.loc[0, "w"])
        assert tab._df.loc[1, "w"] == 20

    def test_append_dtype_int_float(self):
        """df1 int, df2 float → 合并后自动提升为 float"""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3.5, 4.5]})
        tab = load_data(df1, is_display_result=False)
        tab.data.append(df2)
        assert len(tab._df) == 4
        assert tab._df["x"].dtype == np.float64

    def test_append_dtype_string_in_one(self):
        """df1 numeric, df2 有 string 列 → 合并后 object 类型"""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": ["a", "b"]})
        tab = load_data(df1, is_display_result=False)
        tab.data.append(df2)
        assert len(tab._df) == 4
        assert tab._df["x"].dtype == object

    def test_append_same_dtypes_preserved(self):
        """同类型合并后类型不变"""
        df1 = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        df2 = pd.DataFrame({"x": [3, 4], "y": ["c", "d"]})
        tab = load_data(df1, is_display_result=False)
        tab.data.append(df2)
        assert tab._df["x"].dtype == np.int64
        assert pd.api.types.is_string_dtype(tab._df["y"])


class TestDataOpsSort:
    @pytest.fixture
    def sort_df(self):
        return pd.DataFrame({
            "x": [3, 1, 2, 1],
            "y": [40, 20, 30, 10],
        })

    def test_sort_single_var(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.sort("x")
        assert list(tab._df["x"]) == [1, 1, 2, 3]

    def test_sort_multiple_vars_string(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.sort("x y")
        assert list(tab._df["x"]) == [1, 1, 2, 3]
        assert list(tab._df["y"]) == [10, 20, 30, 40]

    def test_sort_multiple_vars_list(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.sort(["x", "y"])
        assert list(tab._df["y"]) == [10, 20, 30, 40]

    def test_sort_nonexistent_raises(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.sort("z")

    def test_sort_returns_data_ops(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        result = tab.data.sort("x")
        assert result is not None

    def test_sort_with_nan(self):
        df = pd.DataFrame({"x": [3, np.nan, 1, 2]})
        tab = load_data(df, is_display_result=False)
        tab.data.sort("x")
        assert list(tab._df["x"][:3]) == [1.0, 2.0, 3.0]
        assert pd.isna(tab._df["x"].iloc[3])

    def test_gsort_descending(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.gsort("-x")
        assert list(tab._df["x"]) == [3, 2, 1, 1]

    def test_gsort_ascending_explicit(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.gsort("+x")
        assert list(tab._df["x"]) == [1, 1, 2, 3]

    def test_gsort_mixed(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [10, 30, 20, 40]})
        tab = load_data(df, is_display_result=False)
        tab.data.gsort("+x -y")
        assert list(tab._df["x"]) == [1, 1, 2, 2]
        assert list(tab._df["y"]) == [30, 10, 40, 20]

    def test_gsort_no_prefix_means_asc(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.gsort("x")
        assert list(tab._df["x"]) == [1, 1, 2, 3]

    def test_gsort_nonexistent_raises(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.gsort("-z")

    def test_gsort_returns_data_ops(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        result = tab.data.gsort("-x")
        assert result is not None

    def test_sort_chaining(self, sort_df):
        tab = load_data(sort_df, is_display_result=False)
        tab.data.sort("x").drop("y")
        assert list(tab._df["x"]) == [1, 1, 2, 3]
        assert "y" not in tab._df.columns


class TestDataOpsRecode:
    @pytest.fixture
    def recode_df(self):
        return pd.DataFrame({"edu": [1, 2, 3, 4, 5], "income": [100, 200, 300, 400, 500]})

    def test_recode_simple(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        tab.data.recode("edu", {1: 0, 2: 0, 3: 1, 4: 1, 5: 2})
        assert list(tab._df["edu"]) == [0, 0, 1, 1, 2]

    def test_recode_with_gen(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        tab.data.recode("edu", {1: "low", 2: "low", 3: "mid", 4: "mid", 5: "high"}, gen="edu_group")
        assert "edu_group" in tab._df.columns
        assert list(tab._df["edu_group"]) == ["low", "low", "mid", "mid", "high"]
        assert list(tab._df["edu"]) == [1, 2, 3, 4, 5]

    def test_recode_range_key(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        tab.data.recode("edu", {(1, 2): "low", (3, 4): "mid", (5, 5): "high"})
        assert list(tab._df["edu"]) == ["low", "low", "mid", "mid", "high"]

    def test_recode_overwrite_inplace(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        tab.data.recode("edu", {1: 10, 2: 20})
        assert tab._df.loc[0, "edu"] == 10
        assert tab._df.loc[1, "edu"] == 20
        assert tab._df.loc[2, "edu"] == 3  # unmapped stays

    def test_recode_gen_existing_raises(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        with pytest.raises(ValueError):
            tab.data.recode("edu", {1: 0}, gen="income")

    def test_recode_nonexistent_var_raises(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        with pytest.raises(KeyError):
            tab.data.recode("z", {1: 0})

    def test_recode_returns_data_ops(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        result = tab.data.recode("edu", {1: 0})
        assert result is not None

    def test_recode_chaining(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        tab.data.recode("edu", {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}).sort("income")
        assert list(tab._df["edu"]) == [0, 0, 1, 1, 2]

    def test_recode_mixed_range_and_single(self, recode_df):
        tab = load_data(recode_df, is_display_result=False)
        tab.data.recode("edu", {(1, 2): "low", 3: "mid", (4, 5): "high"})
        assert list(tab._df["edu"]) == ["low", "low", "mid", "high", "high"]
