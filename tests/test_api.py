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

    def test_loads_csv_path(self, sample_df, tmp_path):
        csv_path = tmp_path / "sample.csv"
        sample_df.to_csv(csv_path, index=False)

        data = load_data(csv_path, is_display_result=False)

        pd.testing.assert_frame_equal(data._df, sample_df)

    def test_missing_path_raises_immediately(self, tmp_path):
        missing_path = tmp_path / "does-not-exist.csv"

        with pytest.raises(FileNotFoundError, match=str(missing_path)):
            load_data(missing_path, is_display_result=False)

    def test_unsupported_file_format_raises_immediately(self, tmp_path):
        unsupported_path = tmp_path / "sample.txt"
        unsupported_path.write_text("x\n1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported data file format '.txt'"):
            load_data(unsupported_path, is_display_result=False)

    @pytest.mark.parametrize("invalid_input", [None, 123, [1, 2, 3]])
    def test_invalid_input_type_raises_immediately(self, invalid_input):
        with pytest.raises(TypeError, match="DataFrame or a supported file path"):
            load_data(invalid_input, is_display_result=False)


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
