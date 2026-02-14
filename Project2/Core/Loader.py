import json
import numpy as np

def load_config(filepath="config.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # ---- Problem ----
    N = data["problem"]["grid"]["N"]
    domain = (
        data["problem"]["domain"]["start"],
        data["problem"]["domain"]["end"]
    )

    # ---- Boundary Conditions ----
    left = data["boundary_conditions"]["left"]
    right = data["boundary_conditions"]["right"]

    BC = np.array([
        [left["a"], left["b"], left["c"]],
        [right["a"], right["b"], right["c"]]
    ])

    # ---- Functions ----
    funcs = data["functions"]

    safe_env = {"np": np}

    F_func = eval(f"lambda x, y, yp: {funcs['F']}", safe_env)
    Fy_func = eval(f"lambda x, y, yp: {funcs['Fy']}", safe_env)
    Fyp_func = eval(f"lambda x, y, yp: {funcs['Fyp']}", safe_env)

    # ---- Analytical (optional) ----
    ana_func = None
    if "analytical_solution" in data.get("postprocessing", {}):
        ana_expr = data["postprocessing"]["analytical_solution"]
        ana_func = eval(f"lambda x: {ana_expr}", safe_env)

    # ---- Solver Settings ----
    tol = data["solver"]["tolerance"]
    max_iter = data["solver"]["max_iterations"]

    return N, domain, BC, F_func, Fy_func, Fyp_func, ana_func, tol, max_iter