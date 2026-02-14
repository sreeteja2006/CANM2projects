import json
import numpy as np

def load_config(filepath="config.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)

    N = data["problem"]["grid"]["N"]
    domain = (
        data["problem"]["domain"]["start"],
        data["problem"]["domain"]["end"]
    )

    left = data["boundary_conditions"]["left"]
    right = data["boundary_conditions"]["right"]

    BC = np.array([
        [left["a"], left["b"], left["c"]],
        [right["a"], right["b"], right["c"]]
    ])

    funcs = data["functions"]
    safe_env = {"np": np}

    # ---- F is mandatory ----
    F_func = eval(f"lambda x, y, yp: {funcs['F']}", safe_env)

    # ---- Fy optional ----
    if funcs.get("Fy", "").strip() == "":
        Fy_func = None
    else:
        Fy_func = eval(f"lambda x, y, yp: {funcs['Fy']}", safe_env)

    # ---- Fyp optional ----
    if funcs.get("Fyp", "").strip() == "":
        Fyp_func = None
    else:
        Fyp_func = eval(f"lambda x, y, yp: {funcs['Fyp']}", safe_env)

    # ---- Analytical optional ----
    # ---- Analytical optional ----
    ana_func = None
    post = data.get("postprocessing", {})

    if post.get("analytical_solution", "").strip() != "":
        ana_func = eval(
            f"lambda x: {post['analytical_solution']}",
            safe_env
        )

    compute_error = post.get("compute_error", False)


    tol = data["solver"]["tolerance"]
    max_iter = data["solver"]["max_iterations"]

    return N, domain, BC, F_func, Fy_func, Fyp_func, ana_func, tol, max_iter,compute_error
