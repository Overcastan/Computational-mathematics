import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

n = 21
u = lambda x: x ** 4 + 3*x ** 3 - 6*x + 5
f = lambda x: 12*x ** 2 + 18 * x
a = u(-1)
b = u(1)
x = np.linspace(-1, 1, 200)

def solution (n, f):
    # Chebyshev
    j = np.arange(n - 2)
    xic = np.cos(np.pi / (2 * (n - 2)) + np.pi * j / (n - 2))
    xic = xic[::-1]
    xic = np.insert(xic, 0, -1)
    xic = np.insert(xic, n - 1, 1)
    yic = f(xic)
    coef_matrix = np.zeros((n - 2, n))
    zzozs = np.zeros(n)
    zzozs[2] = 1

    for i in range(1, n - 1):
        Van_matr = np.zeros((n, n))
        x = xic[i]
        for j in range(n):
            for k in range(n):
                Van_matr[j][k] = ((xic[k] - x) ** j) / np.math.factorial(j)

        coef_line = np.linalg.solve(Van_matr, zzozs)
        for z in range(n):
            coef_matrix[i - 1][z] = coef_line[z]

    f_arr = np.arange(n - 2)
    for i in range(1, n - 1):
        f_arr[i - 1] = yic[i]
    for i in range(n - 2):
        f_arr[i] -= a * coef_matrix[i][0]
        f_arr[i] -= b * coef_matrix[i][n - 1]

    final_coef_m = np.zeros((n-2, n-2))
    for i in range(n - 2):
        for j in range(n - 2):
            final_coef_m[i][j] = coef_matrix[i][j + 1]
    U = np.linalg.solve(final_coef_m, f_arr)
    return xic, U


xic, U = solution(n, f)
diff_array = np.arange(5, 14, dtype=np.float64)
x_dif_arr = np.arange(5, 14)
for i in range(5, 5 + len(x_dif_arr)):
    xic_dif, U_dif = solution(i, f)
    xic_dif = np.delete(xic_dif, 0)
    xic_dif = np.delete(xic_dif, i - 2)
    abc = abs(U_dif - u(xic_dif))
    diff_array[i - 5] = np.log(np.average(abc))
U = np.insert(U, 0, a)
U = np.insert(U, n - 1, b)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))
ax1.plot(x_dif_arr, diff_array, label = "Error")
ax2.plot(x, u(x), color = "cyan", label = "Exact")
ax2.plot(xic, U, color = "red", label = "Calculated")
ax2.legend()
ax1.legend()
plt.show()
