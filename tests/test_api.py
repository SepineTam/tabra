import numpy as np
import pandas as pd
import pytest

from tabra import load_data


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2 + 3 * x1 + 1.5 * x2 + np.random.randn(n) * 0.5
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestLoadData:
    def test_returns_tabra_data(self, sample_df):
        data = load_data(sample_df)
        assert hasattr(data, "reg")


class TestRegAPI:
    def test_reg_with_constant(self, sample_df):
        data = load_data(sample_df)
        result = data.est.reg("y", ["x1", "x2"], is_con=True)
        assert result.n_obs == 100
        assert len(result.coef) == 3

    def test_reg_no_constant(self, sample_df):
        data = load_data(sample_df)
        result = data.est.reg("y", ["x1", "x2"], is_con=False)
        assert len(result.coef) == 2

    def test_result_summary(self, sample_df):
        data = load_data(sample_df)
        result = data.est.reg("y", ["x1", "x2"])
        summary = result.summary()
        assert "x1" in summary
        assert "x2" in summary
        assert "_cons" in summary
        assert "R-squared" in summary

    def test_reg_default_has_constant(self, sample_df):
        data = load_data(sample_df)
        result = data.est.reg("y", ["x1", "x2"])
        assert "_cons" in result.var_names

    def test_reg_invalid_column(self, sample_df):
        data = load_data(sample_df)
        with pytest.raises(KeyError):
            data.est.reg("y", ["nonexistent"])


class TestDisplayResult:
    def test_display_true_prints(self, sample_df, capsys):
        data = load_data(sample_df, is_display_result=True)
        data.est.reg("y", ["x1", "x2"])
        captured = capsys.readouterr()
        assert "R-squared" in captured.out

    def test_display_false_no_print(self, sample_df, capsys):
        data = load_data(sample_df, is_display_result=False)
        data.est.reg("y", ["x1", "x2"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_style_passed_to_result(self, sample_df):
        data = load_data(sample_df, is_display_result=False)
        result = data.est.reg("y", ["x1", "x2"])
        assert result._style == "stata"

    def test_set_style_then_reg(self, sample_df):
        data = load_data(sample_df, is_display_result=False)
        data.set_style("custom")
        result = data.est.reg("y", ["x1", "x2"])
        assert result._style == "custom"
