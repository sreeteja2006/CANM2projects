import sys
import os



import numpy as np
from Project1.ode_solver.core.core import odesolver
from methods.ABM_4 import ABM4
from methods.RK_2 import RK2
import matplotlib.pyplot as plt
from methods.RK_4 import RK4
from methods.ABM2 import ABM2

def f(u, x):
    return np.array([u[1], -u[1]**2/(u[0] + 1e-4)])

eps = 1e-3
bc = ((1, 0, 0), (1, 0, -1))

def analytical_solution(x):
    return 1e-4 * (np.sqrt(100020000*x + 1) - 1)
ob = odesolver(order=2, method=ABM2, bc=bc, tol=1e-8, max_iter=1000, func=f, guess=0.5, xstart=0, xend=1, h=1e-5)

solution, x,_,_,_ = ob.solve()

error = np.abs(solution[:, 0] - analytical_solution(x))
print(f"Max error: {np.max(error)}")
plt.subplot(2, 1, 1)
plt.plot(x, solution[:, 0], label="ABM2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("ODE Solution using ABM2 Method")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x, error, label="Error", color='red')
plt.xlabel("x")
plt.ylabel("Absolute Error")    
plt.title("Error of RK4 Solution")
plt.legend()
plt.grid(True)
plt.show()

