import json
import numpy as np
import os
from .constants import MAX_GRID_SIZE, MAX_ITERATIONS

def load_config(filepath="config.json"):
    """
    Load and validate configuration from JSON file.
    
    Parameters:
    filepath: Path to configuration JSON file
    
    Returns:
    Tuple of configuration parameters
    """
    # Validate filepath
    if filepath is None or filepath == "":
        raise ValueError("filepath must be provided")
    
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath).__name__}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")
    
    # Load JSON with error handling
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"Cannot read configuration file: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading configuration file: {e}") from e
    
    # Validate top-level structure
    required_keys = ["problem", "boundary_conditions", "functions", "solver"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key in config: '{key}'")
    
    # Validate and extract N
    try:
        N = data["problem"]["grid"]["N"]
    except KeyError as e:
        raise KeyError(f"Missing required key: {e}") from e
    
    if not isinstance(N, int):
        try:
            N = int(N)
        except (ValueError, TypeError) as e:
            raise TypeError(f"N must be an integer, got {type(N).__name__}") from e
    
    if N < 1:
        raise ValueError(f"N must be positive, got {N}")
    
    if N > MAX_GRID_SIZE:
        raise ValueError(f"N is too large (max {MAX_GRID_SIZE}), got {N}")
    
    # Validate and extract domain
    try:
        domain_start = data["problem"]["domain"]["start"]
        domain_end = data["problem"]["domain"]["end"]
    except KeyError as e:
        raise KeyError(f"Missing required domain key: {e}") from e
    
    # Convert to float and validate
    try:
        domain_start = float(domain_start)
        domain_end = float(domain_end)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Domain values must be numeric: {e}") from e
    
    if np.isnan(domain_start) or np.isinf(domain_start):
        raise ValueError(f"Domain start is NaN or infinite: {domain_start}")
    if np.isnan(domain_end) or np.isinf(domain_end):
        raise ValueError(f"Domain end is NaN or infinite: {domain_end}")
    
    if domain_start >= domain_end:
        raise ValueError(f"Domain start must be less than end: [{domain_start}, {domain_end}]")
    
    domain = (domain_start, domain_end)
    
    # Validate and extract boundary conditions
    try:
        left = data["boundary_conditions"]["left"]
        right = data["boundary_conditions"]["right"]
    except KeyError as e:
        raise KeyError(f"Missing required boundary condition: {e}") from e
    
    # Extract BC coefficients with validation
    bc_keys = ["a", "b", "c"]
    for side, bc in [("left", left), ("right", right)]:
        for key in bc_keys:
            if key not in bc:
                raise KeyError(f"Missing '{key}' in {side} boundary condition")
            try:
                bc[key] = float(bc[key])
            except (ValueError, TypeError) as e:
                raise TypeError(f"BC {side}.{key} must be numeric: {e}") from e
            if np.isnan(bc[key]) or np.isinf(bc[key]):
                raise ValueError(f"BC {side}.{key} is NaN or infinite")
    
    BC = np.array([
        [left["a"], left["b"], left["c"]],
        [right["a"], right["b"], right["c"]]
    ])
    
    # Validate functions section
    if "functions" not in data:
        raise KeyError("Missing 'functions' section in config")
    
    funcs = data["functions"]
    safe_env = {"np": np}
    
    # ---- F is mandatory ----
    if "F" not in funcs or not funcs["F"].strip():
        raise ValueError("Function 'F' is required and cannot be empty")
    
    try:
        F_func = eval(f"lambda x, y, yp: {funcs['F']}", safe_env)
        # Test if function is callable
        if not callable(F_func):
            raise ValueError("F_func is not callable")
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in function F: {e}") from e
    except Exception as e:
        raise ValueError(f"Error creating function F: {e}") from e
    
    # ---- Fy optional ----
    Fy_func = None
    if funcs.get("Fy", "").strip() != "":
        try:
            Fy_func = eval(f"lambda x, y, yp: {funcs['Fy']}", safe_env)
            if not callable(Fy_func):
                raise ValueError("Fy_func is not callable")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in function Fy: {e}") from e
        except Exception as e:
            raise ValueError(f"Error creating function Fy: {e}") from e
    
    # ---- Fyp optional ----
    Fyp_func = None
    if funcs.get("Fyp", "").strip() != "":
        try:
            Fyp_func = eval(f"lambda x, y, yp: {funcs['Fyp']}", safe_env)
            if not callable(Fyp_func):
                raise ValueError("Fyp_func is not callable")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in function Fyp: {e}") from e
        except Exception as e:
            raise ValueError(f"Error creating function Fyp: {e}") from e
    
    # ---- Analytical optional ----
    ana_func = None
    post = data.get("postprocessing", {})
    
    if post.get("analytical_solution", "").strip() != "":
        try:
            ana_func = eval(
                f"lambda x: {post['analytical_solution']}",
                safe_env
            )
            if not callable(ana_func):
                raise ValueError("analytical_solution is not callable")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in analytical_solution: {e}") from e
        except Exception as e:
            raise ValueError(f"Error creating analytical_solution: {e}") from e
    
    compute_error = post.get("compute_error", False)
    if not isinstance(compute_error, bool):
        raise TypeError(f"compute_error must be boolean, got {type(compute_error).__name__}")
    
    # Validate solver parameters
    try:
        tol = data["solver"]["tolerance"]
        max_iter = data["solver"]["max_iterations"]
    except KeyError as e:
        raise KeyError(f"Missing required solver parameter: {e}") from e
    
    # Validate tolerance
    try:
        tol = float(tol)
    except (ValueError, TypeError) as e:
        raise TypeError(f"tolerance must be numeric: {e}") from e
    
    if tol <= 0:
        raise ValueError(f"tolerance must be positive, got {tol}")
    if tol > 1:
        raise ValueError(f"tolerance seems too large (>1), got {tol}")
    if np.isnan(tol) or np.isinf(tol):
        raise ValueError("tolerance is NaN or infinite")
    
    # Validate max_iter
    if not isinstance(max_iter, int):
        try:
            max_iter = int(max_iter)
        except (ValueError, TypeError) as e:
            raise TypeError(f"max_iterations must be an integer: {e}") from e
    
    if max_iter < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iter}")
    if max_iter > MAX_ITERATIONS:
        raise ValueError(f"max_iterations is too large (max {MAX_ITERATIONS}), got {max_iter}")
    
    return N, domain, BC, F_func, Fy_func, Fyp_func, ana_func, tol, max_iter, compute_error
