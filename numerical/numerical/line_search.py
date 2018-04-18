#!/usr/bin/env python
# encoding: utf-8
"""
@File   : line_search
@author : wuziheng
@Date   : 4/10/18 
@license: 
"""
import numpy as np
import math


def backtracing_line_search(f, xk, pk, scale, c, init_length=1):
    """
    :param f: function object, has f.eval(vector), f.diff(vector),
    :param xk: start_point = (R^n vector)
    :param pk: decrease_vector = (R^n vector)
    :param scale: back tracing contraction factor
    :param c: Armijo condition scale factor c1
    :param init_length: usually 1
    :return: search length at point xk
    """
    step_length = init_length
    while f.eval(xk + pk * step_length) <= f.eval(xk) + c * step_length * f.diff(xk) * pk:
        step_length *= scale
    return step_length


def chol_with_added_mult_identity(A, beta=1e-3):
    """
    Cholesky decompositon of a Matrix, to try the smallest scalar of Indetity to add which will make the original input
     A  positive definite and Cholesky decompose it.
    Cholesky decompose: positive definite matrix A can be decompose as A = LL^T, L is a Lower trigonometric matrix.
    :param A:=[n,n] numpy array, Input square matrix, may not be positive definite.
    :param beta:=float, heuristic param, the initial minimum scalar.
    :return: L after Cholesky decomposition
    """
    I = np.zeros(A.shape)
    for i in range(0, I.shape[0]):
        I[i, i] = 1

    min_trace_a = min([A[i, i] for i in range(0, A.shape[0])])
    if min_trace_a > 0:
        tau = 0
    else:
        tau = -min_trace_a + beta

    while True:
        flag, L = chol_d(A + tau * I)
        if flag == True:
            return L
        else:
            tau = max(2 * tau, beta)


def chol_d(A):
    """
    Cholesky decomposition : A = LL^T
    :param A: Input square matrix
    :return:flag: A is positive definite or not
            L :decomposed Lower trigonometric matrix
    """

    m, n = A.shape[0], A.shape[1]
    flag = (m == n)

    if not flag or len(A.shape) != 2:
        print "A is not a square matrix"
        return False, A

    L = np.zeros(A.shape)
    for k in range(0, n):
        s = 0
        for i in range(0, k):
            s += L[k, i] ** 2
        s = A[k, k] - s
        # s = s if s > 0 else 0
        # L[k, k] = math.sqrt(s)
        if s < 0:
            flag = False
            print 'sqrt error', i, j
            break
        else:
            L[k, k] = math.sqrt(s)

        for i in range(k + 1, n):
            s1 = 0
            for j in range(0, k):
                s1 += L[i, j] * L[k, j]
            if L[k, k] == 0:
                flag = False
                print 'divide 0:', i, j
                break
            L[i, k] = (A[i, k] - s1) / L[k, k]

        for j in range(0, k):
            L[j, k] = 0

    return flag, L


def chol_f(A, beta =10 ,sigma = 1e-3):
    """
        Cholesky Factorization : A = LDL^T
        return L, using the Gaussian elimination method

    :param A: Input Square Matrix
    :param beta: threshold to hold the M(L*sqrt(D)) matrix element
    :param sigma: min threshold of main Matrix D trace element
    :return: Cholesky Factor L, main matrix D
    """
    m, n = A.shape[0], A.shape[1]
    flag = (m == n)

    if not flag or len(A.shape) != 2:
        print "A is not a square matrix"
        return False, A

    L = np.zeros(A.shape)
    C = np.zeros(A.shape)
    D = np.zeros(A.shape[0])

    for j in range(0, n):
        L[j, j] = 1

    D[0] = A[0,0]

    for j in range(0, n):
        tmp_s1 = 0
        for s in range(0, j):
            tmp_s1 += D[s] * (L[j, s] ** 2)
        C[j, j] = A[j, j] - tmp_s1

        # D[j] = C[j, j]
        # modification version
        try:
            theta = max([abs(C[i, j]) for i in range(j + 1, n)])
        except:
            theta = abs(C[n-1,j])

        D[j] = max([abs(C[j, j]), sigma, (theta / beta) ** 2])

        for i in range(j + 1, n):
            tmp_s2 = 0
            for s in range(0, j):
                tmp_s2 += D[s] * L[i, s] * L[j, s]
            C[i, j] = A[i, j] - tmp_s2
            L[i, j] = C[i, j] / D[j]

    return L, D


if __name__ == "__main__":
    A = np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 2], [1, 2, 3, 3, 3], [1, 2, 3, 4, 4], [1, 2, 3, 4, 5]])
    B = np.array([[10, 0, 0], [0, 3, 0], [0, 0, -1]])

    test_chol_d = 0
    test_chol_f = 1

if test_chol_d:
    LA = chol_with_added_mult_identity(A)
    C = np.dot(LA, LA.T)
    print C - A

    LB = chol_with_added_mult_identity(B)
    D = np.dot(LB, LB.T)
    print D - B

if test_chol_f:
    X=B

    LA, DA = chol_f(X)

    DB = np.zeros(X.shape)

    for i in range(0, X.shape[0]):
        DB[i, i] = DA[i]
    # A = LDL^T
    print np.dot(np.dot(LA, DB), LA.T)

# if X is positive definite, we could varify: chol_d(A) = LA * sqrt(DA)
    flag = True
    for i in range(X.shape[0]):
        flag = flag and DA[i]>=0
    if flag:
        for i in range(0, X.shape[0]):
            DB[i, i] = math.sqrt(DA[i])
        print np.dot(LA, DB) - chol_d(X)[1]
