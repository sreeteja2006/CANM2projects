import numpy as np
import matplotlib.pyplot as plt
import os
from core.core import odesolver
from methods.RK_4 import RK4
from methods.RK_2 import RK2
from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4
from analysis.analysis import analysis

print("Welcome to the ODE Solver for Boundary Value Problems (BVPs) using the Shooting Method!")

def parse_func(funcstr):
    funcstr = funcstr.strip()
    if funcstr.startswith("["):
        funcstr = funcstr[1:]
    if funcstr.endswith("]"):
        funcstr = funcstr[:-1]
    
    funcstr = funcstr.replace("y'", "u[1]").replace("y", "u[0]")
    return eval("lambda u,x: np.array([" + funcstr + "])")

def getinput():
    print("\n" + "="*60)
    print("  ODE INPUT")
    print("="*60)
    
    while True:
        try:
            order = int(input("Order of ODE (1 or 2) [default=2]: ") or "2")
            if order not in [1, 2]:
                print("Error: Order must be 1 or 2.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer (1 or 2).")

    print("\nEnter function as: [y', f(y,y',x)]")
    print("Example: y'' + 2y' + 3y = 0  →  [y', -2*y' - 3*y]")
    while True:
        try:
            funcstr = input("f(u,x) = ")
            if not funcstr.strip():
                print("Error: Function cannot be empty.")
                continue
            func = parse_func(funcstr)
            func(np.array([1.0, 1.0]), 0.0)
            break
        except Exception as e:
            print(f"Error: Invalid function. {e}")

    # Domain input
    print("\n" + "-"*40)
    print("  DOMAIN")
    print("-"*40)
    while True:
        try:
            domain = input("x range (start, end, step) [e.g. 0, 1, 0.1]: ")
            parts = [p.strip() for p in domain.split(",")]
            if len(parts) != 3:
                print("Error: Enter exactly 3 values (start, end, step).")
                continue
            xstart, xend, h = float(parts[0]), float(parts[1]), float(parts[2])
            if xend <= xstart:
                print("Error: end must be greater than start.")
                continue
            if h <= 0 or h > (xend - xstart):
                print("Error: step must be positive and smaller than the range.")
                continue
            break
        except ValueError:
            print("Error: Please enter valid numbers.")

    # Boundary conditions
    print("\n" + "-"*40)
    print("  BOUNDARY CONDITIONS (ay + by' + c = 0)")
    print("-"*40)
    while True:
        try:
            bc_start = input("At x=start (a, b, c) [e.g. 1, 0, -1]: ")
            parts = [p.strip() for p in bc_start.split(",")]
            if len(parts) != 3:
                print("Error: Enter exactly 3 values (a, b, c).")
                continue
            a0, b0, c0 = float(parts[0]), float(parts[1]), float(parts[2])
            if a0 == 0 and b0 == 0:
                print("Error: a and b cannot both be zero.")
                continue
            break
        except ValueError:
            print("Error: Please enter valid numbers.")
    
    while True:
        try:
            bc_end = input("At x=end   (a, b, c) [e.g. 1, 0, -2]: ")
            parts = [p.strip() for p in bc_end.split(",")]
            if len(parts) != 3:
                print("Error: Enter exactly 3 values (a, b, c).")
                continue
            a1, b1, c1 = float(parts[0]), float(parts[1]), float(parts[2])
            if a1 == 0 and b1 == 0:
                print("Error: a and b cannot both be zero.")
                continue
            break
        except ValueError:
            print("Error: Please enter valid numbers.")

    # Guesses
    print("\n" + "-"*40)
    print("  SHOOTING METHOD")
    print("-"*40)
    while True:
        try:
            guesses = input("Two initial guesses (g1, g2) [e.g. 0, 1]: ")
            parts = [p.strip() for p in guesses.split(",")]
            if len(parts) != 2:
                print("Error: Enter exactly 2 values.")
                continue
            guess1, guess2 = float(parts[0]), float(parts[1])
            if guess1 == guess2:
                print("Error: Guesses must be different.")
                continue
            break
        except ValueError:
            print("Error: Please enter valid numbers.")

    # Method choice
    print("\nMethods: 1=RK4, 2=RK2, 3=ABM2, 4=ABM4")
    while True:
        try:
            method_choice = int(input("Choose method [default=1]: ") or "1")
            if method_choice not in [1, 2, 3, 4]:
                print("Error: Choose 1, 2, 3, or 4.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer.")
    
    methods = {1: RK4, 2: RK2, 3: ABM2, 4: ABM4}
    method = methods.get(method_choice, RK4)
    print("="*60 + "\n")
    
    return {'order': order, 'func': func, 'xstart': xstart, 'xend': xend, 'h': h,
            'bc_start': (a0, b0, c0), 'bc_end': (a1, b1, c1), 'guess': [guess1, guess2], 'method': method}

def load_from_file(filepath):
    methods = {
        1: RK4,
        2: RK2,
        3: ABM2,
        4: ABM4
    }
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    
    order = int(lines[0])
    funcstr = lines[1]
    if funcstr.startswith("["):
        funcstr = funcstr[1:]
    if funcstr.endswith("]"):
        funcstr = funcstr[:-1]
    funcstr = funcstr.replace("y'", "u[1]").replace("y", "u[0]")
    func = eval("lambda u,x: np.array([" + funcstr + "])")
    xstart = float(lines[2])
    xend = float(lines[3])
    h = float(lines[4])
    a0 = float(lines[5])
    b0 = float(lines[6])
    c0 = float(lines[7])
    a1 = float(lines[8])
    b1 = float(lines[9])
    c1 = float(lines[10])
    guess1 = float(lines[11])
    guess2 = float(lines[12])
    method_choice = int(lines[13])
    method = methods.get(method_choice, RK4)
    
    show_stability = lines[14].lower() == 'y' if len(lines) > 14 else False
    show_convergence = lines[15].lower() == 'y' if len(lines) > 15 else False
    
    params = {'order': order, 'func': func, 'xstart': xstart, 'xend': xend, 'h': h,
              'bc_start': (a0, b0, c0), 'bc_end': (a1, b1, c1), 'guess': [guess1, guess2], 'method': method}
    
    return params, show_stability, show_convergence

def solve_bvp(params):
    solver = odesolver(
        order=params['order'],
        method=params['method'],
        bc=(params['bc_start'], params['bc_end']),
        tol=1e-8,
        max_iter=100,
        func=params['func'],
        xstart=params['xstart'],
        xend=params['xend'],
        h=params['h'],
        guess=params['guess']
    )
    u, x, _, _, _ = solver.solve()
    return u, x

def stability_plots(params):
    solver = odesolver(
        order=params['order'],
        method=params['method'],
        bc=(params['bc_start'], params['bc_end']),
        tol=1e-8,
        max_iter=100,
        func=params['func'],
        xstart=params['xstart'],
        xend=params['xend'],
        h=params['h'],
        guess=params['guess']
    )
    
    analysis_obj = analysis(solver)
    analysis_obj.run_full_stability_analysis()

def convergence_plots(params):
    solver = odesolver(
        order=params['order'],
        method=params['method'],
        bc=(params['bc_start'], params['bc_end']),
        tol=1e-8,
        max_iter=100,
        func=params['func'],
        xstart=params['xstart'],
        xend=params['xend'],
        h=params['h'],
        guess=params['guess']
    )
    analysis_obj = analysis(solver)
    results = analysis_obj.h_refinement(params['h'])
    analysis_obj.plot_loglog_convergence(results)

def plotting(params, u, x):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    method_name = params['method'].__name__ if hasattr(params['method'], '__name__') else str(params['method'])
    ax1 = axes[0]
    ax1.plot(x, u[:, 0], 'b-', linewidth=2, marker='o', markersize=3, label='y(x)')
    ax1.set_title(f'Solution y(x) using {method_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2 = axes[1]
    ax2.plot(x, u[:, 1], 'r-', linewidth=2, marker='x', markersize=3, label="y'(x)")
    ax2.set_title(f"Derivative y'(x) using {method_name}", fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("y'", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.suptitle('BVP Solution using Shooting Method', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # saving the plots to respective directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'solution_plot.png'), dpi=150)
    plt.show()

if __name__ == "__main__":
    print("\nChoose input mode:")
    print("1. Use default inputs (from input.txt)")
    print("2. Enter custom inputs")
    choice = input("Enter your choice (1 or 2): ").strip()

    show_stability = None
    show_convergence = None

    if choice == '1':
        input_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input.txt')
        if os.path.exists(input_file):
            print(f"\nLoading inputs from {input_file}...")
            params, show_stability, show_convergence = load_from_file(input_file)
        else:
            print(f"\nError: {input_file} not found. Switching to custom input mode.")
            params = getinput()
    else:
        params = getinput()
        
    u, x = solve_bvp(params)
    print("\nSolution u(x):")
    for xi, ui in zip(x, u):
        print(f"x: {xi:.4f}, u: {ui[0]:.4f}, u': {ui[1]:.4f}")

    plotting(params, u, x)
    if show_stability is None:
        print("Do you want to see stability plots? (y/n)")
        show_stability = input().lower() == 'y'
    if show_stability:
        stability_plots(params)
    if show_convergence is None:
        show_convergence = input("\nDo you want to see convergence plots? (y/n)\n").lower() == 'y'
    if show_convergence:
        print("\nGenerating convergence plots...\n")
        convergence_plots(params)





        
        