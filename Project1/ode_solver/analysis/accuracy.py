import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from core.core import odesolver
from methods.RK_2 import RK2
from methods.RK_4 import RK4
from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4

def f(u, x):
    return np.array([u[1], -u[1]**2/(u[0] + 1e-4)])

eps = 1e-3
bc = ((1, 0, 0), (1, 0, -1))

def analytical_solution(x, eps=0):
    return (np.sqrt(100020000*(x) + 1) - 1)


ob = odesolver(order=2, method=RK2, bc=bc, tol=1e-8, max_iter=1000, func=f, guess=[0.5, 1], xstart=0, xend=1, h=1e-5)
sol_RK2, x,_,_,_ = ob.solve()

ob.set_method(RK4)
sol_RK4, x,_,_,_ = ob.solve()
ob.set_method(ABM2)
sol_ABM2, x,_,_,_ = ob.solve()
ob.set_method(ABM4)
sol_ABM4, x,_,_,_ = ob.solve()

# analytic evaluated on a fine grid and on the solver's x for alignment
x_analytical = np.linspace(0, 1, 1000)
# account for eps being added to initial x before the IVP solver starts
# shift analytic evaluation back by eps; clip to >= 0 to avoid invalid sqrt
x_shifted = np.maximum(x - eps, 0)
x_analytical_shifted = np.maximum(x_analytical - eps, 0)
sol_analytical_highres = analytical_solution(x_analytical_shifted, eps=eps)
sol_analytical_on_x = analytical_solution(x_shifted, eps=eps)

plt.figure(figsize=(10, 6))

# RK2 subplot
plt.subplot(2, 2, 1)
delta = sol_RK2[0, 0] - sol_analytical_on_x[0]
<<<<<<< Updated upstream
plt.plot(x, sol_RK2[:, 0], label='RK2', linestyle='--')
=======
plt.plot(x, sol_RK2[300:, 0], label='RK2', linestyle='--')
>>>>>>> Stashed changes
plt.plot(x_analytical, sol_analytical_highres + delta, label='Analytical (shifted, x-offset)', linestyle='-')
plt.title('RK2 Solution vs Analytical (aligned)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# RK4 subplot
plt.subplot(2, 2, 2)
delta = sol_RK4[0, 0] - sol_analytical_on_x[0]
<<<<<<< Updated upstream
plt.plot(x, sol_RK4[:, 0], label='RK4', linestyle='--')
=======
plt.plot(x, sol_RK4[200:, 0], label='RK4', linestyle='--')
>>>>>>> Stashed changes
plt.plot(x_analytical, sol_analytical_highres + delta, label='Analytical (shifted, x-offset)', linestyle='-')
plt.title('RK4 Solution vs Analytical (aligned)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# ABM2 subplot
plt.subplot(2, 2, 3)
delta = sol_ABM2[0, 0] - sol_analytical_on_x[0]
<<<<<<< Updated upstream
plt.plot(x, sol_ABM2, label='ABM2', linestyle='--')
=======
plt.plot(x, sol_ABM2[100:,0], label='ABM2', linestyle='--')
>>>>>>> Stashed changes
plt.plot(x_analytical, sol_analytical_highres + delta, label='Analytical (shifted, x-offset)', linestyle='-')
plt.title('ABM2 Solution vs Analytical (aligned)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# ABM4 subplot
plt.subplot(2, 2, 4)
delta = sol_ABM4[0, 0] - sol_analytical_on_x[0]
plt.plot(x, sol_ABM4[:, 0], label='ABM4', linestyle='--')
plt.plot(x_analytical, sol_analytical_highres + delta, label='Analytical (shifted, x-offset)', linestyle='-')
plt.title('ABM4 Solution vs Analytical (aligned)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
# save figure to PNG next to this script (use high DPI)
outpath = os.path.join(os.path.dirname(__file__), 'accuracy_plot.png')
plt.savefig(outpath, dpi=300)
plt.show()


