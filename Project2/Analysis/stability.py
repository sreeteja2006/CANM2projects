import numpy as np
from Project2.Test.Teja import TDMA


def residual(w, eps, alpha, beta):
    n = len(w) - 1
    F = np.zeros_like(w)

    for i in range(1, n):
        F[i] = (
            w[i + 1]
            - 2 * w[i]
            + w[i - 1]
            + ((w[i] - w[i - 1]) ** 2) / (4 * (w[i] + eps))
        )

    F[0] = w[0]-alpha
    F[n] = w[n] - beta
    return F


def jacobian(w, eps):
    n = len(w) - 1

    d = np.zeros(n + 1)
    l = np.zeros(n)
    u = np.zeros(n)

    d[0] = 1.0
    d[n] = 1.0

    for i in range(1, n):
        wi = w[i]
        wim1 = w[i - 1]

        nonlinear = (wi - wim1) ** 2
        denom = 4 * (wi + eps)

        u[i] = 1.0
        l[i - 1] = 1.0 - (2 * (wi - wim1)) / denom
        d[i] = (
            -2
            + (2 * (wi - wim1)) / denom
            - nonlinear / (4 * (wi + eps) ** 2)
        )

    return d, l, u


def infnorm_tridiag(d, l, u):
    n = len(d)
    max_row_sum = 0.0

    for i in range(n):
        row_sum = abs(d[i])
        if i > 0:
            row_sum += abs(l[i - 1])
        if i < n - 1:
            row_sum += abs(u[i])
        max_row_sum = max(max_row_sum, row_sum)

    return max_row_sum


def inverse_infnorm_tridiag(d, l, u):
    n = len(d)
    max_norm = 0.0

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0

        x = TDMA(d.copy(), l.copy(), u.copy(), e)
        max_norm = max(max_norm, np.linalg.norm(x, np.inf))

    return max_norm


def cond_tridiag(d, l, u):
    return infnorm_tridiag(d, l, u) * inverse_infnorm_tridiag(d, l, u)


n = 100
eps = 1e-4

w = np.linspace(0, 1, n + 1)

for k in range(50):

    F = residual(w, eps, 0, 1)

    d, l, u = jacobian(w, eps)

    cond = cond_tridiag(d, l, u)
    print(f"Iter {k}: cond(J) = {cond:.3e}")

    delta = TDMA(d.copy(), l.copy(), u.copy(), -F)

    w += delta

    if np.linalg.norm(delta, np.inf) < 1e-10:
        print(f"Converged in {k+1} iterations")
        break
else:
    print("Did not converge")
