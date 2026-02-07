import numpy as np

def RK2(sol, u0):
    nx = int((sol.xend - sol.xstart) // sol.h)
    u = np.zeros((nx+1,len(u0)))
    x = np.zeros(nx+1)
    u[0] = u0
    x[0] = sol.xstart
    for i in range(nx):
        k1 = sol.h*sol.func(u[i], x[i])
        k2 = sol.h*sol.func(u[i] + k1, x[i] + sol.h)
        u[i+1] = u[i] + (k1+k2)/2
        x[i+1] = x[i] + sol.h
    return u, x, False

# def RK2_Backshot(sol, u0):
#     nx = int((sol.xend - sol.xstart) // sol.h)
#     u_0 = np.zeros(nx+1)
#     u_1 = np.zeros(nx+1)
#     u_0[-1] = u0[0]
#     u_1[-1] = u0[1]
#     for i in range(nx, 0, -1):
#         k1_1 = sol.h*sol.func(u_0[i], u_1[i])
#         k2_1 = sol.h*sol.func(u_0[i], u_1[i])
#         k1_2 = sol.h*sol.func(u_0[i] - k1_1, u_1[i] - k2_1)
#         k2_2 = sol.h*sol.func(u_0[i] - k1_1, u_1[i] - k2_1)
#         u_0[i-1] = u_0[i] - 0.5*(k1_1 + k1_2)
#         u_1[i-1] = u_1[i] - 0.5*(k2_1 + k2_2)
#     return u_0, u_1