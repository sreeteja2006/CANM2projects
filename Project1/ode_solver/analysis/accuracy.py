import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
<<<<<<< Updated upstream
=======
from pathlib import Path
>>>>>>> Stashed changes
import matplotlib.pyplot as plt
from core.core import odesolver
from methods.RK_2 import RK2
from methods.RK_4 import RK4
from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4

<<<<<<< Updated upstream
=======
# Try to load precomputed analytical solution values from y_rk2.txt
_y_data = None
_x_data = None
data_path = Path(__file__).resolve().parent / "y_rk2.txt"
if data_path.exists():
    try:
        loaded = np.loadtxt(data_path)
        if loaded.ndim == 1:
            _y_data = loaded
        elif loaded.ndim == 2:
            if loaded.shape[1] == 1:
                _y_data = loaded.flatten()
            else:
                _x_data = loaded[:, 0]
                _y_data = loaded[:, 1]
    except Exception:
        _y_data = None
        _x_data = None

>>>>>>> Stashed changes
def f(u, x):
    return np.array([u[1], -u[1]**2/(u[0] + 1e-4)])

eps = 1e-3
bc = ((1, 0, 0), (1, 0, -1))

<<<<<<< Updated upstream
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
# <<<<<<< Updated upstream
# plt.plot(x, sol_RK2[:, 0], label='RK2', linestyle='--')
# =======
plt.plot(x, sol_RK2[300:, 0], label='RK2', linestyle='--')
# >>>>>>> Stashed changes
plt.plot(x_analytical, sol_analytical_highres + delta, label='Analytical (shifted, x-offset)', linestyle='-')
plt.title('RK2 Solution vs Analytical (aligned)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# RK4 subplot
plt.subplot(2, 2, 2)
delta = sol_RK4[0, 0] - sol_analytical_on_x[0]
# =======
plt.plot(x, sol_RK4[200:, 0], label='RK4', linestyle='--')
# >>>>>>> Stashed changes
plt.plot(x_analytical, sol_analytical_highres + delta, label='Analytical (shifted, x-offset)', linestyle='-')
plt.title('RK4 Solution vs Analytical (aligned)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# ABM2 subplot
plt.subplot(2, 2, 3)
delta = sol_ABM2[0, 0] - sol_analytical_on_x[0]
# <<<<<<< Updated upstream
# plt.plot(x, sol_ABM2, label='ABM2', linestyle='--')
# =======
plt.plot(x, sol_ABM2[100:,0], label='ABM2', linestyle='--')
# >>>>>>> Stashed changes
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
=======
def analytical_solution(x, small=1e-4):
    # Use precomputed data from y_rk2.txt if available, otherwise fall back
    # to the analytical formula.
    global _y_data, _x_data
    x_arr = np.asarray(x)
    scalar_input = False
    if x_arr.ndim == 0:
        scalar_input = True
        x_arr = x_arr[np.newaxis]

    if '_y_data' in globals() and _y_data is not None:
        if _x_data is not None:
            y_interp = np.interp(x_arr, _x_data, _y_data)
            return y_interp[0] if scalar_input else y_interp
        if _y_data.shape[0] == x_arr.shape[0]:
            return _y_data[0] if scalar_input else _y_data

    result = small * (np.sqrt(100020000 * x_arr + 1) - 1)
    return result[0] if scalar_input else result


ob = odesolver(order=2, method=RK2, bc=bc, tol=1e-8, max_iter=1000, func=f, guess=[0.5, 1], xstart=0, xend=1, h=1e-5)

# Methods and labels
methods = [(RK2, 'RK2'), (RK4, 'RK4'), (ABM2, 'ABM2'), (ABM4, 'ABM4')]

# Step sizes to test
h_values = [1e-3, 1e-4, 1e-5, 1e-6]

# Keep originals to restore later
original_h = ob.get_h()
original_method = ob.get_method()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()

for idx, (method, name) in enumerate(methods):
    ax = axes[idx]
    for h in h_values:
        ob.set_method(method)
        ob.set_h(h)
        sol, x, _, blew, _ = ob.solve()
        if blew:
            print(f"Method {name} blew up at h={h}; skipping")
            continue
        y_analytical_on_x = analytical_solution(x)
        error = np.abs(sol[:, 0] - y_analytical_on_x)
        ax.plot(x, error, label=f'h={h}')

    ax.set_title(f'Absolute Error |y_numerical - y_analytical| ({name})')
    ax.set_xlabel('x')
    ax.set_ylabel('Absolute Error')
    ax.legend(fontsize='small')
    ax.grid(True)

# Restore solver state
ob.set_method(original_method)
ob.set_h(original_h)

# Save and show
outpath = os.path.join(os.path.dirname(__file__), 'accuracy_error_plot.png')
>>>>>>> Stashed changes
plt.savefig(outpath, dpi=300)
plt.show()


