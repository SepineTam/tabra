"""Tests for the restricted data expression boundary."""

import numpy as np
import pandas as pd
import pytest

from tabra.data.expressions import ExpressionEngine


@pytest.fixture
def frame():
    return pd.DataFrame({"x": [1.0, 2.0, 4.0], "y": [2.0, 5.0, 8.0]})


def test_arithmetic_and_stata_power_operator(frame):
    result = ExpressionEngine.evaluate("x ^ 2 + y", frame)
    assert np.allclose(result, frame["x"] ** 2 + frame["y"])


def test_approved_math_functions(frame):
    result = ExpressionEngine.evaluate("log(x) + sqrt(y)", frame)
    assert np.allclose(result, np.log(frame["x"]) + np.sqrt(frame["y"]))


def test_comparisons_and_boolean_operators(frame):
    result = ExpressionEngine.evaluate("(x >= 2) & (y < 8)", frame)
    assert result.tolist() == [False, True, False]


def test_string_comparison_is_supported():
    frame = pd.DataFrame({"group": ["control", "treated", "treated"]})
    result = ExpressionEngine.evaluate("group == 'treated'", frame)
    assert result.tolist() == [False, True, True]


@pytest.mark.parametrize(
    "expression",
    [
        "x.__class__",
        "__import__('os')",
        "open('data.txt')",
        "x[0]",
    ],
)
def test_arbitrary_python_features_are_rejected(frame, expression):
    with pytest.raises(ValueError):
        ExpressionEngine.evaluate(expression, frame)
