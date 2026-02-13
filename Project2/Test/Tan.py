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
    # We prefix "lambda x, y, yp: " so Python treats the string as a function definition.
    # We pass {'np': np} to eval() so it knows what "np" means in your formulas.
    
    funcs = data['functions']
    
    # Compile F(x, y, yp)
    F_func = eval(f"lambda x, y, yp: {funcs['F']}", {"np": np})
    
    # Compile Fy(x, y, yp)
    Fy_func = eval(f"lambda x, y, yp: {funcs['Fy']}", {"np": np})
    
    # Compile Fyp(x, y, yp)
    Fyp_func = eval(f"lambda x, y, yp: {funcs['Fyp']}", {"np": np})
    
    # Compile Analytical Solution (only takes x)
    # Check if 'analytical' exists in json to avoid errors if you remove it later
    if 'analytical' in funcs:
        ana_func = eval(f"lambda x: {funcs['analytical']}", {"np": np})
    else:
        ana_func = None

    return N, domain, BC, F_func, Fy_func, Fyp_func, ana_func

# --- Main Execution ---

# Load everything from the file
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

# Plotting
if analytical_solution:
    x = np.linspace(domain[0], domain[1], N+1)
    analytical_values = analytical_solution(x)
    print("Analytical Solution:", analytical_values)
    print("Max Error:", np.max(np.abs(analytical_values - solution)))

    plt.figure(figsize=(10, 6))
    plt.plot(x, analytical_values, label='Analytical', marker='o', linestyle='-', alpha=0.6)
    plt.plot(x, solution, label='Numerical (FDM)', marker='x', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'BVP Solver Output (N={N})')
    plt.legend()
    plt.grid(True)
    plt.show()