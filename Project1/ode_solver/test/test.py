import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.core import odesolver
from methods.RK_2 import RK2
import numpy as np
from methods.RK_2 import RK2
import matplotlib.pyplot as plt
from methods.RK_4 import RK4
from methods.ABM_2 import ABM2
from analysis.analysis import analysis

def f(u, x):
    return np.array([u[1], -u[1]**2/(u[0] + 1e-4)])

eps = 1e-3
bc = ((1, 0, 0), (1, 0, -1))

def analytical_solution(x):
    return 1e-4 * (np.sqrt(100020000*x + 1) - 1)

ob = odesolver(order=2, method=ABM2, bc=bc, tol=1e-10, max_iter=1000, func=f, guess=[10,20], xstart=0, xend=1, h=1e-5)
solution, x,_,_,_ = ob.solve()

# error = np.abs(solution[:, 0] - analytical_solution(x))
# print(f"Max error: {np.max(error)}")
# plt.subplot(2, 1, 1)
# plt.plot(x, solution[:, 0], label="ABM2")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("ODE Solution using ABM2 Method")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(x, error, label="Error", color='red')
# plt.xlabel("x")
# plt.ylabel("Absolute Error")    
# plt.title("Error of ABM2 Solution")
# plt.legend()
# plt.grid(True)
# plt.show()

# Create analysis object and test accuracy
analyzer = analysis(ob)
results = analyzer.h_refinement(1e-3)
for name, data in results.items():
    print(f"\nMethod: {name}")
    print(f"  Local orders : {np.round(data['orders'], 3)}")
    print(f"  Avg order    : {data['order_avg']:.3f}")
    print(f"  Std dev      : {data['order_std']:.3f}")

analyzer.plot_loglog_convergence(results)