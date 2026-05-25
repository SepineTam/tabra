import numpy as np

from tabra.results.confusion_matrix_result import ConfusionMatrixResult
from tabra.results.cov_result import CovResult
from tabra.results.crosstab_result import CrosstabResult
from tabra.results.xttrans_result import XttransResult


def test_confusion_matrix_result_summary_and_properties(tmp_path):
    matrix = np.array([[8, 2], [1, 9]])
    labels = ["neg", "pos"]
    result = ConfusionMatrixResult(matrix, labels, accuracy=0.85, n_obs=20)

    assert result.var_names == labels
    assert result.accuracy == 0.85
    assert result.n_obs == 20

    text = result.summary()
    assert "Accuracy: 0.8500" in text
    assert "N = 20" in text
    assert "neg" in text and "pos" in text

    out = tmp_path / "cm.txt"
    result.save(out)
    assert "Accuracy: 0.8500" in out.read_text()


def test_cov_result_summary_lower_triangle_and_save(tmp_path):
    matrix = np.array([[4.0, 1.2], [1.2, 9.0]])
    result = CovResult(matrix, ["x", "long_name"], n_obs=50)

    text = result.summary()
    assert "x" in text and "long_name" in text
    assert "4.0000" in text
    assert "1.2000" in text

    out = tmp_path / "cov.txt"
    result.save(out)
    assert "1.2000" in out.read_text()


def test_crosstab_result_percents_and_summary(tmp_path):
    matrix = np.array([[3, 1], [2, 4]])
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    grand_total = int(matrix.sum())
    result = CrosstabResult(
        matrix=matrix,
        row_labels=["A", "B"],
        col_labels=["X", "Y"],
        row_var="row_var",
        col_var="col_var",
        row_totals=row_totals,
        col_totals=col_totals,
        grand_total=grand_total,
    )

    np.testing.assert_allclose(result.row_percent.sum(axis=1), [100.0, 100.0])
    np.testing.assert_allclose(result.col_percent.sum(axis=0), [100.0, 100.0])
    assert np.isclose(result.cell_percent.sum(), 100.0)

    text = result.summary()
    assert "Total" in text
    assert "A" in text and "B" in text

    out = tmp_path / "crosstab.txt"
    result.save(out)
    assert "Total" in out.read_text()


def test_xttrans_result_summary_and_aliases(tmp_path):
    count = np.array([[4, 1], [2, 3]])
    prob = np.array([[0.8, 0.2], [0.4, 0.6]])
    result = XttransResult(
        count_matrix=count,
        prob_matrix=prob,
        state_labels=["s0", "s1"],
        var="state",
        n_obs=12,
        n_transitions=10,
    )

    assert result.var_names == ["s0", "s1"]
    np.testing.assert_allclose(result.matrix, prob)
    assert result.n_transitions == 10

    text = result.summary()
    assert "Total transitions: 10" in text
    assert "s0" in text and "s1" in text

    out = tmp_path / "xttrans.txt"
    result.save(out)
    assert "Total transitions: 10" in out.read_text()
