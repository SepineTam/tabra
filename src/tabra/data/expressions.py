"""Restricted expression evaluation for Stata-like data commands."""

from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd


class ExpressionEngine(ast.NodeVisitor):
    """Evaluate arithmetic expressions without executing arbitrary Python code."""

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
    }
    _unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Invert: operator.invert,
        ast.Not: np.logical_not,
    }
    _comparison_operators = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
    }

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame
        self._functions = {
            "exp": np.exp,
            "log": self._log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": np.abs,
        }

    @classmethod
    def evaluate(cls, expression: str, frame: pd.DataFrame):
        """Evaluate a supported expression against DataFrame columns."""
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-empty string")
        normalized = expression.replace("^", "**")
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression: {expression}") from exc
        return cls(frame).visit(tree.body)

    @staticmethod
    def _log(value, base=None):
        if base is None:
            return np.log(value)
        return np.log(value) / np.log(base)

    def visit_Name(self, node: ast.Name):
        if node.id in self._frame.columns:
            return self._frame[node.id]
        if node.id in {"True", "False"}:
            return node.id == "True"
        raise ValueError(f"Unknown variable or function: {node.id}")

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float, bool, str)) or node.value is None:
            return node.value
        raise ValueError("Unsupported constant type")

    def visit_BinOp(self, node: ast.BinOp):
        operation = self._binary_operators.get(type(node.op))
        if operation is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operation(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operation = self._unary_operators.get(type(node.op))
        if operation is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operation(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        combined = None
        for operator_node, comparator_node in zip(node.ops, node.comparators):
            operation = self._comparison_operators.get(type(operator_node))
            if operation is None:
                raise ValueError(
                    f"Unsupported comparison: {type(operator_node).__name__}"
                )
            right = self.visit(comparator_node)
            current = operation(left, right)
            combined = current if combined is None else operator.and_(combined, current)
            left = right
        return combined

    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            operation = operator.and_
        elif isinstance(node.op, ast.Or):
            operation = operator.or_
        else:
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        values = [self.visit(value) for value in node.values]
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in self._functions:
            raise ValueError("Only approved math functions are supported")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in expressions")
        function = self._functions[node.func.id]
        arguments = [self.visit(argument) for argument in node.args]
        return function(*arguments)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")
