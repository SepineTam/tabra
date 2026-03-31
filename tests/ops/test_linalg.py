#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : test_linalg.py
#

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tabra.ops.linalg import (
    mat_mul,
    mat_transpose,
    identity,
    lu_decompose,
    solve_linear,
    mat_inv,
    det,
    qr_decompose,
)


class TestMatMul:
    """测试矩阵乘法"""

    def test_square_matrix(self):
        """方阵乘法"""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        expected = np.dot(A, B)
        result = mat_mul(A, B)
        assert_allclose(result, expected)

    def test_rectangular_matrix(self):
        """矩形矩阵乘法"""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        B = np.array([[7, 8], [9, 10], [11, 12]])  # 3x2
        expected = np.dot(A, B)
        result = mat_mul(A, B)
        assert_allclose(result, expected)


class TestMatTranspose:
    """测试矩阵转置"""

    def test_square_transpose(self):
        """方阵转置"""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = A.T
        result = mat_transpose(A)
        assert_allclose(result, expected)

    def test_row_vector_transpose(self):
        """行向量转置为列向量"""
        A = np.array([[1, 2, 3, 4]])
        expected = A.T
        result = mat_transpose(A)
        assert_allclose(result, expected)


class TestIdentity:
    """测试单位矩阵"""

    def test_basic_identity(self):
        """基本单位矩阵"""
        result = identity(3)
        expected = np.eye(3)
        assert_allclose(result, expected)

    def test_identity_size_1(self):
        """1x1 单位矩阵"""
        result = identity(1)
        expected = np.eye(1)
        assert_allclose(result, expected)


class TestLUDecompose:
    """测试 LU 分解"""

    def test_2x2_exact_values(self):
        """2x2 矩阵精确值验证"""
        A = np.array([[4, 3], [6, 3]], dtype=float)
        L, U = lu_decompose(A)
        # L 应该是下三角且对角线为1
        assert_allclose(L, np.tril(L, -1) + np.eye(2))
        # U 应该是上三角
        assert_allclose(U, np.triu(U))
        # 验证 LU = A
        assert_allclose(L @ U, A)

    def test_3x3_verify_lu_equals_a(self):
        """3x3 矩阵验证 LU=A"""
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        L, U = lu_decompose(A)
        assert_allclose(L @ U, A, rtol=1e-10)

    def test_singular_matrix_raises_error(self):
        """奇异矩阵抛出 ValueError"""
        A = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="singular"):
            lu_decompose(A)


class TestSolveLinear:
    """测试线性方程组求解"""

    def test_2x2_system(self):
        """2x2 线性方程组"""
        A = np.array([[2, 1], [1, 2]], dtype=float)
        b = np.array([3, 3])
        expected = np.linalg.solve(A, b)
        result = solve_linear(A, b)
        assert_allclose(result, expected)

    def test_3x3_system(self):
        """3x3 线性方程组"""
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        b = np.array([4, 10, 24])
        expected = np.linalg.solve(A, b)
        result = solve_linear(A, b)
        assert_allclose(result, expected)

    def test_singular_system_raises_error(self):
        """奇异系统抛出 ValueError"""
        A = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([1, 2])
        with pytest.raises(ValueError):
            solve_linear(A, b)


class TestMatInv:
    """测试矩阵求逆"""

    def test_2x2_inverse(self):
        """2x2 矩阵求逆"""
        A = np.array([[4, 3], [3, 2]], dtype=float)
        expected = np.linalg.inv(A)
        result = mat_inv(A)
        assert_allclose(result, expected, atol=1e-10)

    def test_3x3_inverse(self):
        """3x3 矩阵求逆"""
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        expected = np.linalg.inv(A)
        result = mat_inv(A)
        assert_allclose(result, expected, atol=1e-10)

    def test_identity_inverse(self):
        """单位阵逆矩阵是其本身"""
        A = np.eye(4)
        result = mat_inv(A)
        assert_allclose(result, A, atol=1e-10)

    def test_singular_matrix_raises_error(self):
        """奇异矩阵抛出 ValueError"""
        A = np.array([[1, 1], [1, 1]], dtype=float)
        with pytest.raises(ValueError):
            mat_inv(A)

    def test_non_square_raises_error(self):
        """非方阵抛出 ValueError"""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="square"):
            mat_inv(A)


class TestDet:
    """测试行列式计算"""

    def test_2x2_determinant(self):
        """2x2 行列式"""
        A = np.array([[4, 3], [3, 2]], dtype=float)
        expected = np.linalg.det(A)
        result = det(A)
        assert_allclose(result, expected)

    def test_3x3_determinant(self):
        """3x3 行列式"""
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        expected = np.linalg.det(A)
        result = det(A)
        assert_allclose(result, expected)

    def test_identity_determinant(self):
        """单位阵行列式为1"""
        A = np.eye(5)
        result = det(A)
        assert_allclose(result, 1.0)

    def test_non_square_raises_error(self):
        """非方阵抛出 ValueError"""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="square"):
            det(A)


class TestQRDecompose:
    """测试 QR 分解"""

    def test_rectangular_matrix(self):
        """矩形矩阵 (3x2) 验证 Q^TQ=I 和 QR=A"""
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Q, R = qr_decompose(A)
        # Q^T Q = I
        assert_allclose(Q.T @ Q, np.eye(2), atol=1e-10)
        # QR = A
        assert_allclose(Q @ R, A, atol=1e-10)

    def test_square_matrix(self):
        """方阵 (3x3) 验证 Q^TQ=I 和 QR=A"""
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        Q, R = qr_decompose(A)
        # Q 是正交矩阵
        assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
        assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)
        # QR = A
        assert_allclose(Q @ R, A, atol=1e-10)
