import numpy as np
import scipy.integrate as integrate

n = 20
a = -5
b = 5

f1 = lambda x: x ** 3 + x - 2
f2 = np.sin


def integral_solution (f, a, b, n):
    betta = [(i + 1) / (2 * i + 1) for i in range(1, n + 1)]
    Lej_f_bot = betta * np.eye(n, k = 1)
    Lej_f_upp = betta * np.eye(n, k = -1)
    Lej_f = Lej_f_bot + Lej_f_upp
    nodes = np.linalg.eigvals(Lej_f)
    weights = []
    for i in range(len(nodes)):
        t_j = np.delete(nodes, i)
        w_i = integrate.quad(lambda t: ((b - a) / 2) * ((t - t_j) / (nodes[i] - t_j)).prod(), -1, 1)
        weights.append(w_i[0])

    x = ((a + b) / 2) + ((b - a) / 2) * nodes
    return (f(x) * weights).sum()


Gauss1 = integral_solution(f1, -1, 1, 2)
print("Gaussian quadrature for polynomial: ", Gauss1)
print("Difference for polynomial: ", Gauss1 - integrate.quad(f1, -1, 1)[0])

Gauss2 = integral_solution(f2, -1, 1, 5)
print("Gaussian quadrature for non polynomial: ", Gauss2)
print("Difference for non polynomial: ", Gauss2 - integrate.quad(f2, -1, 1)[0])
