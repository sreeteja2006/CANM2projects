import numpy as np

def ABM2(sol,u0):
    n = int((sol.xend - sol.xstart) // sol.h)
    u = np.zeros((n+1, len(u0)))
    x = np.zeros(n+1)

    u[0] = u0
    x[0] = sol.xstart

    def RK4_step(f, u, x, h):
        k1 = h*f(u, x)
        k2 = h*f(u + k1/2, x + h/2)
        k3 = h*f(u + k2/2, x + h/2)
        k4 = h*f(u + k3, x + h)
        return u + (k1 + 2*k2 + 2*k3 + k4)/6

    u[1] = RK4_step(sol.func, u[0], x[0], sol.h)
    x[1] = x[0] + sol.h

    for i in range(1, n):
        x[i+1] = x[i] + sol.h
        up = u[i] + sol.h/2*(3*sol.func(u[i], x[i]) - sol.func(u[i-1], x[i-1]))
        u[i+1] = u[i] + sol.h/2*(sol.func(up, x[i+1]) + sol.func(u[i], x[i]))
        if(np.any(np.isnan(u[i+1])) or np.any(np.isinf(u[i+1])) or np.any(np.abs(u[i+1]) > 1e10)):
            return u[:i+2], x[:i+2], True

    return u, x, False