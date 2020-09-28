import numpy as np
import copy


def prod(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(n):
        for j in range(n):
            x[i] += A[i][j] * b[j]
    return x


def make_symmetric(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            A[i, j] = A[j, i]
    return A


def transposing(A):
    n = A.shape[0]
    Res = copy.deepcopy(A)
    for i in range(n):
        for j in range(i, n):
            Res[i, j], Res[j, i] = Res[j, i], Res[i, j]
    return Res


def my_solve_triang(T, b, flag):
    n = b.shape[0]
    x = np.zeros(n)
    if flag == 'l':
        for i in range(n):
            x[i] = b[i]
            for j in range(i):
                x[i] -= x[j] * T[i, j]
            x[i] = x[i] / T[i, i]
    else:
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(n - i - 1):
                x[i] -= x[i + j + 1] * T[i, i + j + 1]
            x[i] = x[i] / T[i, i]
    return x


n = int(input())
while 1:
    counter = 0
    A = np.random.rand(n, n)
    A = make_symmetric(A)
    eig = np.linalg.eigvals(A)
    for i in eig:
        if i > 0:
            counter += 1
    if counter == n:
        break
b = np.random.rand(n)


def my_kholetsky_solve(A, b):
    C = np.zeros((n, n))  # lower triangular matrix
    C[0, 0] = (A[0, 0]) ** 0.5
    for i in range(1, n):
        C[i, 0] = A[i, 0] / C[0, 0]
    for k in range(1, n):
        C[k, k] = (A[k, k] - sum(C[k, 0: k] ** 2)) ** 0.5           # diag elem
        for i in range(k + 1, n):
            C[i, k] = A[i, k] - sum(C[i, 0: k] * C[k, 0: k])
            C[i, k] = C[i, k] / C[k, k]
    C_T = transposing(C)  # upper triangular matrix
    y = my_solve_triang(C, b, 'l')
    x = my_solve_triang(C_T, y, 'u')
    return C, x


x = my_kholetsky_solve(A, b)[1]
linalg_solution = np.linalg.solve(A, b)
print("Difference = ", np.linalg.norm(x - linalg_solution))
