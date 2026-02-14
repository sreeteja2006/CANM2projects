import json
import numpy as np
from Project2.Core.Loader import load_config
from Project2.Core.FDM_Solver import FDM_Solver


def save_config(data, filepath="config.json"):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print("\nConfiguration saved to config.json\n")


def manual_input():

    print("\nEnter Problem Parameters\n")

    N = int(input("Number of grid intervals N: "))
    x0 = float(input("Domain start: "))
    x1 = float(input("Domain end: "))

    print("\nLeft Boundary Condition (a*y + b*y' + c = 0)")
    aL = float(input("a: "))
    bL = float(input("b: "))
    cL = float(input("c: "))

    print("\nRight Boundary Condition (a*y + b*y' + c = 0)")
    aR = float(input("a: "))
    bR = float(input("b: "))
    cR = float(input("c: "))

    print("\nEnter Functions (use Python syntax, np allowed)")
    F = input("F(x,y,yp) = ")
    Fy = input("Fy(x,y,yp) = ")
    Fyp = input("Fyp(x,y,yp) = ")

    analytical = input("Analytical solution (optional, press Enter to skip): ")

    data = {
        "problem": {
            "equation": "y'' = F(x,y,y')",
            "domain": {"start": x0, "end": x1},
            "grid": {"N": N}
        },
        "boundary_conditions": {
            "left": {"a": aL, "b": bL, "c": cL},
            "right": {"a": aR, "b": bR, "c": cR}
        },
        "functions": {
            "F": F,
            "Fy": Fy,
            "Fyp": Fyp
        },
        "solver": {
            "tolerance": 1e-8,
            "max_iterations": 50
        },
        "postprocessing": {}
    }

    if analytical.strip():
        data["postprocessing"]["analytical_solution"] = analytical

    save_config(data)

    return load_config("config.json")


def main():

    print("Nonlinear BVP Solver (FDM + Newton)\n")

    choice = input("Load problem from config.json? (y/n): ").lower()

    if choice == 'y':
        params = load_config("config.json")
    else:
        params = manual_input()

    N, domain, BC, F, Fy, Fyp, ana, tol, max_iter = params

    solver = FDM_Solver(F, Fy, Fyp, N, domain, BC)
    solution = solver.solver(tol=tol, max_iter=max_iter)

    print("\nSolver finished.\n")

    if ana is not None:
        x = np.linspace(domain[0], domain[1], N+1)
        error = np.max(np.abs(solution - ana(x)))
        print(f"Max error: {error:.6e}")


if __name__ == "__main__":
    main()
