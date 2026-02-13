import json
import numpy as np
import matplotlib.pyplot as plt
from Core.FDM_Solver import FDM_Solver

def load_config(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 1. Convert standard parameters
    N = data['N']
    domain = tuple(data['domain'])
    BC = np.array(data['BC'])
    
    # 2. Convert function strings to executable Python functions
    funcs = data['functions']
    
    # Compile F(x, y, yp)
    F_func = eval(f"lambda x, y, yp: {funcs['F']}", {"np": np})
    
    # Compile Fy(x, y, yp)
    Fy_func = eval(f"lambda x, y, yp: {funcs['Fy']}", {"np": np})
    
    # Compile Fyp(x, y, yp)
    Fyp_func = eval(f"lambda x, y, yp: {funcs['Fyp']}", {"np": np})
    
    # Compile Analytical Solution (Optional)
    if 'analytical' in funcs:
        ana_func = eval(f"lambda x: {funcs['analytical']}", {"np": np})
    else:
        ana_func = None

    return N, domain, BC, F_func, Fy_func, Fyp_func, ana_func

# --- Main Execution ---

config_path = 'config.json'
try:
    N, domain, BC, F, Fy, Fyp, analytical_solution = load_config(config_path)
    print(f"Successfully loaded configuration from {config_path}")
except FileNotFoundError:
    print(f"Error: Could not find {config_path}")
    exit()

# Initialize Solver
solver = FDM_Solver(F, Fy, Fyp, N, domain, BC)
solution = solver.solver()

print("Numerical Solution:", solution)

# --- Plotting Section ---

# 1. Define x-axis grid (Needed for both plots)
x = np.linspace(domain[0], domain[1], N+1)

plt.figure(figsize=(10, 6))

# 2. Plot Numerical Solution (Always runs)
plt.plot(x, solution, label='Numerical (FDM)', linestyle='--', color='blue')

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