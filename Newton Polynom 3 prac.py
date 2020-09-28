%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt

f = lambda x: np.sin(2 * x) * np.exp(-x)
n = 20
x = np.linspace(-1, 1, 200)

# uniform
x_i = np.linspace(-1, 1, n)
y_i = f(x_i)

# Chebyshev
j = np.arange(n)
xic = np.cos(np.pi / (2 * n) + np.pi * j / n)
xic = xic[::-1]
yic = f(xic)


def Table_of_dd(x_i, y_i):
    n = x_i.shape[0]
    table_of_dd = np.random.rand(n, n)
    for i in range(n):
        table_of_dd[i, 0] = y_i[i]

    for j in range(1, n):
        for i in range(j, n):
            table_of_dd[i, j] = (table_of_dd[i, j - 1] - table_of_dd[i - 1, j - 1]) / (x_i[i] - x_i[i - j])
    return table_of_dd


table_of_dd_uni = Table_of_dd(x_i, y_i)
table_of_dd_cheb = Table_of_dd(xic, yic)


def Newton_pol_calc(table_of_dd, x_i, x):
    L = table_of_dd[0, 0]
    prod = 1
    for i in range(1, n):
        prod *= (x - x_i[i - 1])
        L += table_of_dd[i, i] * prod
    return L


def derivative_Newton_pol_calc(table_of_dd, x_i, x):
    n = x_i.shape[0]
    dL = table_of_dd[1, 1]
    pr1 = x - x_i[1]
    pr2 = x - x_i[0]
    for i in range(2, n):
        dL += table_of_dd[i, i] * (pr1 + pr2)
        pr1 = (pr1 + pr2) * (x - x_i[i])
        pr2 = (x - x_i[i - 1]) * pr2
    return dL


y_uni = []
y_cheb = []
dy_uni = []
dy_cheb = []

for value in x:
    y_uni.append(Newton_pol_calc(table_of_dd_uni, x_i, value))
    y_cheb.append(Newton_pol_calc(table_of_dd_cheb, xic, value))
    dy_uni.append(derivative_Newton_pol_calc(table_of_dd_uni, x_i, value))
    dy_cheb.append(derivative_Newton_pol_calc(table_of_dd_cheb, xic, value))

df = lambda x: 2 * np.cos(2 * x) * np.exp(-x) - np.sin(2 * x) * np.exp(-x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
ax1.plot(x, y_uni, label="Uniform", color='cyan')
ax1.plot(x, y_cheb, label="Chebyshev", color='green')
ax1.plot(x, f(x), label="Exact", color='red')
ax1.plot(x_i, y_i, 'o')
ax1.legend()

ax2.plot(x, dy_uni, label="derivative Uniform", color='cyan')
ax2.plot(x, dy_cheb, label="derivative Chebyshev", color='green')
ax2.plot(x, df(x), label="derivative Exact", color='red')
ax2.legend()

plt.show()
