import numpy as np

def TDMA(u, l, d, b):
    n = len(d)
    Q = np.zeros(n)
    P = np.zeros(n - 1)
    Q[0] = b[0] / d[0]
    P[0] = u[0] / d[0]
    for i in range(1, n - 1):
        P[i] = u[i] / (d[i] - l[i - 1] * P[i - 1])
        Q[i] = (b[i] - l[i - 1] * Q[i - 1]) / (d[i] - l[i - 1] * P[i - 1])

    Q[n - 1] = (b[n - 1] - l[n - 2] * Q[n - 2]) / (d[n - 1] - l[n - 2] * P[n - 2])
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = Q[i] - P[i] * x[i + 1]
    return x