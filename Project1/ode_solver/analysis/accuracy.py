import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from pathlib import Path
from pathlib import Path
import matplotlib.pyplot as plt
from core.core import odesolver
from methods.RK_2 import RK2
from methods.RK_4 import RK4
from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4
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
def f(u, x):
    return np.array([u[1], -u[1]**2/(u[0] + 1e-4)])

bc = ((1, 0, 0), (1, 0, -1))

# Load precomputed reference solution from y_ref.txt (if available)
y_ref_data = None
y_ref_path = Path(__file__).resolve().parent / "y_ref.txt"
if y_ref_path.exists():
    try:
        y_ref_data = np.loadtxt(y_ref_path)
    except Exception:
        y_ref_data = None


def analytical_solution(x):
    """Return reference/analytical values at points x.

    If `y_ref.txt` is present it will be used (interpolated across [0,1]).
    Otherwise fall back to the analytic formula used in the reference run.
    """
    x_arr = np.asarray(x)
    if y_ref_data is not None:
        x_ref = np.linspace(0, 1, y_ref_data.size)
        return np.interp(x_arr, x_ref, y_ref_data)
    # fallback analytical formula (matches reference.py)
    return 0.0001 * (np.sqrt(100020000 * x_arr + 1) - 1)


# ob = odesolver(order=2, method=RK2, bc=bc, tol=1e-8, max_iter=1000, func=f, guess=[0.5, 1], xstart=0, xend=1, h=1e-5)
# sol_RK2, x,_,_,_ = ob.solve()

# ob.set_method(RK4)
# sol_RK4, x,_,_,_ = ob.solve()
# ob.set_method(ABM2)
# sol_ABM2, x,_,_,_ = ob.solve()
# ob.set_method(ABM4)
# sol_ABM4, x,_,_,_ = ob.solve()

# # analytic evaluated on a fine grid and on the solver's x for alignment
# x_analytical = np.linspace(0, 1, 1000)
# sol_analytical_highres = analytical_solution(x_analytical)
# sol_analytical_on_x = analytical_solution(x)

# plt.figure(figsize=(10, 6))

# # RK2 subplot
# plt.subplot(2, 2, 1)
# plt.plot(x, sol_RK2[:, 0], label='RK2', linestyle='--')
# plt.plot(x_analytical, sol_analytical_highres, label='Analytical', linestyle='-')
# plt.title('RK2 Solution vs Analytical (aligned)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# # RK4 subplot
# plt.subplot(2, 2, 2)
# plt.plot(x, sol_RK4[:, 0], label='RK4', linestyle='--')
# plt.plot(x_analytical, sol_analytical_highres, label='Analytical', linestyle='-')
# plt.title('RK4 Solution vs Analytical (aligned)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# # ABM2 subplot
# plt.subplot(2, 2, 3)
# plt.plot(x, sol_ABM2[:,0], label='ABM2', linestyle='--')
# plt.plot(x_analytical, sol_analytical_highres, label='Analytical', linestyle='-')
# plt.title('ABM2 Solution vs Analytical (aligned)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# # ABM4 subplot
# plt.subplot(2, 2, 4)
# plt.plot(x, sol_ABM4[:, 0], label='ABM4', linestyle='--')
# plt.plot(x_analytical, sol_analytical_highres, label='Analytical', linestyle='-')
# plt.title('ABM4 Solution vs Analytical (aligned)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

plt.tight_layout()
# save figure to PNG next to this script (use high DPI)
outpath = os.path.join(os.path.dirname(__file__), 'accuracy_plot.png')
 # (y_ref loading and analytical_solution moved earlier)
ob = odesolver(order=2, method=RK2, bc=bc, tol=1e-8, max_iter=1000, func=f, guess=[0.5, 1], xstart=0, xend=1, h=1e-5)

# Methods and labels
methods = [(RK2, 'RK2'), (RK4, 'RK4'), (ABM2, 'ABM2'), (ABM4, 'ABM4')]

# Step sizes to test
h_values = [1e-3, 5e-3, 1e-4, 5e-4]

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
plt.savefig(outpath, dpi=300)
plt.show()


