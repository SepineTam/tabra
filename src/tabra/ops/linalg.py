#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : linalg.py
"""
Linear algebra operations for tabra.

Provides core matrix operations including matrix multiplication,
transpose, LU decomposition, QR decomposition, linear system solving,
matrix inversion, and determinant calculation.
"""

import numpy as np


def mat_mul(A, B):
    """Matrix multiplication using np.dot.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Product matrix (m x p)
    """
    return np.dot(A, B)


def mat_transpose(A):
    """Matrix transpose.

    Args:
        A: Input matrix (m x n)

    Returns:
        Transposed matrix (n x m)
    """
    return A.T


def identity(n):
    """Create an n x n identity matrix.

    Args:
        n: Size of the identity matrix

    Returns:
        n x n identity matrix
    """
    return np.eye(n)


def lu_decompose(A):
    """LU decomposition using Doolittle's method without pivoting.

    Decomposes A into L (lower triangular with unit diagonal)
    and U (upper triangular) such that A = LU.

    Args:
        A: Square matrix (n x n)

    Returns:
        Tuple of (L, U) where L is lower triangular with ones on diagonal,
        U is upper triangular matrix.

    Raises:
        ValueError: If matrix is singular (zero pivot encountered).
    """
    n = A.shape[0]
    A = A.astype(float, copy=True)
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # U[i, j] = A[i, j] - sum(L[i, k] * U[k, j])
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - s

        # Check for zero pivot
        if abs(U[i, i]) < 1e-14:
            raise ValueError("Matrix is singular, cannot perform LU decomposition")

        # L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i])) / U[i, i]
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - s) / U[i, i]

    return L, U


def solve_linear(A, b):
    """Solve linear system Ax = b using LU decomposition.

    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n,)

    Returns:
        Solution vector x.

    Raises:
        ValueError: If matrix is singular.
    """
    L, U = lu_decompose(A)
    n = A.shape[0]
    b = b.astype(float, copy=True)

    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = b[i] - s

    # Backward substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]

    return x


def mat_inv(A):
    """Compute matrix inverse by solving AX = I column by column.

    Args:
        A: Square matrix (n x n)

    Returns:
        Inverse matrix (n x n).

    Raises:
        ValueError: If matrix is non-square or singular.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for inversion")

    n = A.shape[0]
    I = np.eye(n)
    inv = np.zeros((n, n))

    for i in range(n):
        inv[:, i] = solve_linear(A, I[:, i])

    return inv


def det(A):
    """Compute determinant using LU decomposition.

    det(A) = det(L) * det(U) = 1 * product(diag(U))

    Args:
        A: Square matrix (n x n)

    Returns:
        Determinant of A.

    Raises:
        ValueError: If matrix is non-square.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for determinant")

    _, U = lu_decompose(A)
    return np.prod(np.diag(U))


def qr_decompose(A):
    """QR decomposition using Modified Gram-Schmidt orthogonalization.

    Decomposes A into Q (orthonormal columns) and R (upper triangular)
    such that A = QR.

    Args:
        A: Input matrix (m x n), m >= n

    Returns:
        Tuple of (Q, R) where Q has orthonormal columns (m x n),
        R is upper triangular (n x n).
    """
    m, n = A.shape
    A = A.astype(float, copy=True)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Start with j-th column of A
        v = A[:, j].copy()

        # Orthogonalize against previous columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        # Normalize
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-14:
            Q[:, j] = v
        else:
            Q[:, j] = v / R[j, j]

    return Q, R
