import numpy as np
import copy
import matplotlib.pyplot as plot


def prod(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(n):
        for j in range(n):
            x[i] += A[i][j] * b[j]
    return x


def my_symmetric_zeidel_solve(A, b, max_iter, tol):
    n = A.shape[0]
    xkp1 = np.zeros(n)
    iter_number = 0
    discrepancy = []
    for k in range(max_iter):
        xk = copy.deepcopy(xkp1)
        if k % 2 == 0:
            for i in range(n):
                xkp1[i] = b[i]
                for j in range(i):
                    xkp1[i] -= A[i, j] * xkp1[j]
                for j in range(i + 1, n):
                    xkp1[i] -= A[i, j] * xk[j]
                xkp1[i] /= A[i, i]
        else:
            for i in range(n - 1, -1, -1):
                xkp1[i] = b[i]
                for j in range(n - i - 1):
                    xkp1[i] -= A[i, i + j + 1] * xkp1[i + j + 1]
                for j in range(i):
                    xkp1[i] -= A[i, j] * xk[j]
                xkp1[i] /= A[i, i]
        iter_number += 1
        discrepancy.append(np.linalg.norm(prod(A, xkp1) - b))
        if discrepancy[-1] < tol:
            break
    return xkp1, discrepancy, iter_number


n = int(input())
max_iter = int(input())
tol = float(input())
while 1:
    A = np.random.rand(n, n)
    det = np.linalg.det(A)
    if det != 0:
        A = A @ A.T
        break
b = np.random.rand(n)


solution = my_symmetric_zeidel_solve(A, b, max_iter, tol)
appr_x = solution[0]
discrepancy = solution[1]
iter_number = solution[2]
iterations_array = np.arange(iter_number)
fig, ax = plot.subplots(figsize=(16, 9))
plot.grid()
plot.xlabel('Номер итерации', fontsize='xx-large')
plot.ylabel('Логарифм невязки', fontsize='xx-large')
ax.plot(iterations_array, np.log(discrepancy))
print("Число итераций: ", iter_number)
print("Точная ошибка: ", np.linalg.norm(appr_x - np.linalg.solve(A, b)))
fig.tight_layout()
plot.show()
