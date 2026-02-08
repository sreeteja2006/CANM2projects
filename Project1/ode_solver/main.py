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
def getinput():
    order = int(input("\nEnter the order of the ODE (1 or 2): "))


    print("\n Enter the function in form of f(u,x) = [u[0],u[1],....,f(u,x)]\n")
    print("\n For example, for u'' + 2u' + 3u = 0, you can enter: [u[1], -2*u[1] - 3*u[0]]\n")

    funcstr = input("f(u,x) = [")
    funcstr =funcstr.rstrip("]")
    func = eval("lambda u,x: np.array([" + funcstr + "])")


    xstart = float(input("Enter the start value of x: "))
    xend = float(input("Enter the end value of x: "))
    h = float(input("Enter the step size h: "))


    print("\n Enter boundary conditions in the form of ay+by'+c=0:")
    print("\n At xstart:")
    a0 = float(input("a: "))
    b0 = float(input("b: "))
    c0 = float(input("c: "))
    print("\n At xend:")
    a1 = float(input("a: "))
    b1 = float(input("b: "))
    c1 = float(input("c: "))

    print('\n give the two guesses for shooting method:')
    guess1 = float(input("guess 1: "))
    guess2 = float(input("guess 2: "))

    guess = [guess1, guess2]

    methods = {
        1: RK4,
        2: RK2,
        3: ABM2,
        4: ABM4
    }

    print("\n Choose the method to solve the ODE:")
    print("1. Runge-Kutta 4th order (RK4)")
    print("2. Runge-Kutta 2nd order (RK2)")
    print("3. Adams-Bashforth-Moulton 2nd order (ABM2)")
    print("4. Adams-Bashforth-Moulton 4th order (ABM4)")

    method_choice = int(input("Enter the number corresponding to the method: "))
    method = methods.get(method_choice, RK4)

    return {'order': order, 'func': func, 'xstart': xstart, 'xend': xend, 'h': h,
            'bc_start': (a0, b0, c0), 'bc_end': (a1, b1, c1), 'guess': guess, 'method': method}

def load_from_file(filepath):
    """Load inputs from a file."""
    methods = {
        1: RK4,
        2: RK2,
        3: ABM2,
        4: ABM4
    }
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    order = int(lines[0])
    funcstr = lines[1].rstrip("]")
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
    
    # Store plot preferences for later use
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
    
    if show_stability is None:
        print("Do you want to see stability plots? (y/n)")
        show_stability = input().lower() == 'y'
    if show_stability:
        stability_plots(params)
    
    if show_convergence is None:
        show_convergence = input("\nDo you want to see convergence plots? (y/n)").lower() == 'y'
    if show_convergence:
        print("\nGenerating convergence plots...")
        convergence_plots(params)






        
        