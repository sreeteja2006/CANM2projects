import numpy as np
import matplotlib.pyplot as plt
import Project2.Test.Teja
import copy


def infnorm(A):
    n, m = A.shape
    norm = 0
    for i in range(n):
        s = 0
        for j in range(m):
            v = A[i, j]
            s += v if v >= 0 else -v
        norm = s if s > norm else norm
    return norm


def diag(vec, k):
    n = vec.size
    if k == -1:
        M = np.zeros((n + 1, n + 1), dtype=float)
        for i in range(1, n + 1):
            M[i, i - 1] = vec[i - 1]
        return M
    if k == 0:
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            M[i, i] = vec[i]
        return M
    if k == 1:
        M = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            M[i, i + 1] = vec[i]
        return M
    raise ValueError("k must be -1, 0, or 1")


def matmul(A, B):
    n, p = A.shape
    p2, m = B.shape
    if p != p2:
        raise ValueError("shape mismatch")
    C = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def getminormat(A, i, j):
    m = copy.deepcopy(A)
    m.pop(i)
    for row in m:
        row.pop(j)
    return m


def determinant(A):
    numcols = len(A[0])
    numrows = len(A)

    if (numrows != numcols):
        raise ValueError("Please input a sqaure matrix")
    if numcols == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    sum = 0
    for i in range(numcols):
        sum += (A[0, i])*((-1)**(i))*determinant(getminormat(A, 0, i))
    return sum


def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def inv(A):
    cofactors = []
    n = len(A[0])
    for i in range(n):
        rows = []
        for j in range(n):
            minor = getminormat(A, i, j)
            cofactor = ((-1)**(i+j))*determinant(minor)
            rows.append(cofactor)
        cofactors.append(rows)

    Adjancencymatrix = transpose(cofactors)

    if (determinant(A)):
        return Adjancencymatrix/determinant(A)
    else:
        return -1


def condnum(J):
    return infnorm(J)*(infnorm(inv(J)))
