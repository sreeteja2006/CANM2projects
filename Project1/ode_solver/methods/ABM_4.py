import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings



def RK4_step(f, y0, x0, dt, n_steps):
    y = np.zeros((n_steps+1, len(y0)))
    x = np.zeros(n_steps+1)
    y[0] = y0
    x[0] = x0
    blowup = False
    
    for i in range(n_steps):
        x[i+1] = x[i] + dt
    
    for i in tqdm.tqdm(range(n_steps)):
        k1 = dt * f(y[i], x[i])
        k2 = dt * f(y[i] + k1/2, x[i] + dt/2)
        k3 = dt * f(y[i] + k2/2, x[i] + dt/2)
        k4 = dt * f(y[i] + k3, x[i] + dt)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])) or np.any(np.abs(y[i+1]) > 1e10):
            blowup = True
            break
    
    return y, x, blowup

def ABM4(sol, u0):
    n = int((sol.xend - sol.xstart) // sol.h)
    u = np.zeros((n+1, len(u0)))
    x = np.zeros(n+1)

    bootstrap_result = RK4_step(sol.func, u0, sol.xstart, sol.h, 3)
    u[:4], x[:4], bootstrap_blowup = bootstrap_result
    
    if bootstrap_blowup:
        sol.blowup = True
        return u, x, True
    
    for i in range(3, n):
        x[i+1] = x[i] + sol.h

        f_i = sol.func(u[i], x[i])
        f_i1 = sol.func(u[i-1], x[i-1])
        f_i2 = sol.func(u[i-2], x[i-2])
        f_i3 = sol.func(u[i-3], x[i-3])

        up = u[i] + sol.h/24 * (55*f_i - 59*f_i1 + 37*f_i2 - 9*f_i3)
        f_up = sol.func(up, x[i+1])
        u[i+1] = u[i] + sol.h/24 * (9*f_up + 19*f_i - 5*f_i1 + f_i2)

        if np.any(np.isnan(u[i+1])) or np.any(np.isinf(u[i+1])) or np.any(np.abs(u[i+1]) > 1e10):
            sol.blowup = True
            return u[:i+2], x[:i+2], True

    return u, x, False
# def shooting_ABM4(f, y0, y1, num_steps, h, tol, eps=1e-3, debug=False):

#     x_start = eps
#     s0, s1 = 0.5, 1
#     iteration = 0

#     def F(s):
#         y_start = s * eps 
#         u0 = np.array([y_start, s])
#         u, _ = ABM4(f, u0, x_start, h, num_steps, debug=False)
#         return u[-1, 0] - y1

#     F0, F1 = F(s0), F(s1)
    
#     while abs(F1) > tol:
#         iteration += 1
#         if abs(F1 - F0) < 1e-15:
#             break
#         s2 = s1 - F1*(s1 - s0)/(F1 - F0)
#         s0, s1 = s1, s2
#         F0, F1 = F1, F(s1)
#         if iteration > 50:
#             break
        
#     print(f"Final slope: s = {s1:.10f} after {iteration} iterations")
    
#     y_start = s1 * eps
#     u0 = np.array([y_start, s1])
#     return ABM4(f, u0, x_start, h, num_steps, debug=debug)



