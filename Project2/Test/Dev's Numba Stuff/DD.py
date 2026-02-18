import json
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
# from Core.FDM_Solver import FDM_Solver
from core.Loader import load_config
from core.FDM_Solver import FDM_Solver

# --- Main Execution ---

config_path = 'configs/config.json'
try:
    N, domain, BC, F, Fy, Fyp, analytical_solution, tol, max_iter, error = load_config(config_path)
    print(f"Successfully loaded configuration from {config_path}")
except FileNotFoundError:
    print(f"Error: Could not find {config_path}")
    exit()
import numpy as np
from typing import Callable


#minor change
# Initialize Solver
solver = FDM_Solver(F, None, None, N, domain, BC)
solution = solver.solver(tol=tol, max_iter=max_iter)

print("Numerical Solution:", solution)

# --- Plotting Section ---

# 1. Define x-axis grid (Needed for both plots)
x = np.linspace(domain[0], domain[1], N+1)

plt.figure(figsize=(10, 6))

# 2. Plot Numerical Solution (Always runs)
plt.plot(x, solution, label='Numerical (FDM)',marker = 'x' ,linestyle='--', color='blue')

# 3. Plot Analytical Solution (Only runs if available)
if analytical_solution:
    try:
        analytical_values = analytical_solution(x)
        print("Analytical Solution:", analytical_values)
        print("Max Error:", np.max(np.abs(analytical_values - solution)))
        
        plt.plot(x, analytical_values, label='Analytical', marker='o', linestyle='-', alpha=0.6, color='orange')
    except Exception as e:
        print(f"Warning: Could not evaluate analytical solution: {e}")

# 4. Finalize Plot
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'BVP Solver Output (N={N})')
plt.legend()
plt.grid(True)
plt.show()