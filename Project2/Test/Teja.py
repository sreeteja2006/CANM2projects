import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def matrix_gen(n: int, alpha, beta, h):
    N = n + 1
    w_grid = np.zeros(N)
    for i in range(N):
        w_grid[i] = i * (beta - alpha) / n + alpha

    w_cur = globals().get("w", None)
    if isinstance(w_cur, np.ndarray) and w_cur.shape == (N,):
        w = w_cur
    else:
        w = w_grid.copy()

    d = np.zeros(N)
    u = np.zeros(n)
    l = np.zeros(n)

    d[0] = 1.0
    d[-1] = 1.0
    u[0] = 0.0
    l[-1] = 0.0

    for i in range(1, n):
        d[i] = 2 + ((w_grid[i+1] - w_grid[i-1])**2) / (4 * (w[i] + 1e-4)**2)
        l[i-1] = -1 + (w_grid[i+1] - w_grid[i-1]) / (w[i] + 1e-4)
        u[i] = -1 - (w_grid[i+1] - w_grid[i-1]) / (w[i] + 1e-4)

    return d, l, u, w


def TDMA(d, l, u, b):
    n = len(d)
    scratch = np.zeros(n)
    y = np.zeros(n)

    scratch[0] = u[0] / d[0]
    y[0] = b[0] / d[0]

    for i in range(1, n):
        denom = d[i] - l[i-1] * scratch[i-1]
        if i < n - 1:
            scratch[i] = u[i] / denom
        y[i] = (b[i] - l[i-1] * y[i-1]) / denom

    x = np.zeros(n)
    x[-1] = y[-1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] - scratch[i] * x[i + 1]

    return x


d, l, u, w = matrix_gen(1000, 0, 1, 1e-4)
n = 1000
b = np.zeros(n + 1)

b[0] = w[0]
b[-1] = w[-1] - 1


def f(w):
    for i in range(1, n):
        b[i] = w[i + 1] - 2 * w[i] + w[i - 1] + \
            ((w[i] - w[i - 1])**2) / (4 * (w[i] + 1e-4))
    b[0] = w[0]
    b[-1] = w[-1] - 1
    return b


b = f(w)

for i in range(100):
    x = TDMA(d, l, u, b)
    w += x
    if (np.linalg.norm(w) > 1e2):
        break
    b = f(w)
    d, l, u, _ = matrix_gen(1000, 0, 1, 1e-4)
    if np.linalg.norm(x, np.inf) < 1e-10:
        break


plt.plot(w)
plt.savefig("solution_plot.png")
