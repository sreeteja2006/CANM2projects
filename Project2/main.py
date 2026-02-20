"""
Project2: Finite Difference Method Solver for Nonlinear BVPs
Main driver program for solving y'' = F(x, y, y') with boundary conditions
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from core.Loader import load_config
from core.FDM_Solver import FDM_Solver
from analysis.Convergence import ConvergenceStudy
from analysis.stability import StabilitySolver
from analysis.Accuracy import AccuracyAnalysis

def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)


def print_header():
    """Display welcome message."""
    print_separator()
    print("FDM Solver for Nonlinear Boundary Value Problems".center(70))
    print("Solving: y'' = F(x, y, y') with boundary conditions".center(70))
    print_separator()
    print()


def get_yes_no(prompt):
    """Get yes/no response from user."""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def get_float_input(prompt, allow_empty=False):
    """Get and validate float input."""
    while True:
        try:
            user_input = input(prompt).strip()
            if allow_empty and user_input == "":
                return None
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_int_input(prompt, min_val=None):
    """Get and validate integer input."""
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def get_function_input(prompt, param_str, example):
    """Get and validate function input."""
    print(f"\n{prompt}")
    print(f"Parameters: {param_str}")
    print(f"Example: {example}")
    
    while True:
        func_str = input("Enter function: ").strip()
        if func_str == "":
            return None
        
        # Test the function
        try:
            safe_env = {"np": np}
            test_func = eval(f"lambda {param_str}: {func_str}", safe_env)
            # Test evaluation
            if param_str == "x":
                _ = test_func(0.5)
            else:
                _ = test_func(0.5, 0.5, 0.5)
            return func_str
        except Exception as e:
            print(f"Error in function: {e}")
            print("Please try again.")


def manual_input():
    """Collect all parameters manually from user."""
    print("\n--- Problem Configuration ---\n")
    
    # Domain
    print("Domain: [x_start, x_end]")
    x_start = get_float_input("  Enter x_start: ")
    x_end = get_float_input("  Enter x_end: ")
    domain = (x_start, x_end)
    
    # Grid size
    N = get_int_input("  Enter number of grid intervals (N): ", min_val=2)
    
    # Boundary conditions
    print("\nBoundary Conditions: a*y + b*y' + c = 0")
    print("Left boundary (at x_start):")
    a0 = get_float_input("  a: ")
    b0 = get_float_input("  b: ")
    c0 = get_float_input("  c: ")
    
    print("Right boundary (at x_end):")
    a1 = get_float_input("  a: ")
    b1 = get_float_input("  b: ")
    c1 = get_float_input("  c: ")
    
    BC = np.array([[a0, b0, c0], [a1, b1, c1]])
    
    # Function F
    F_str = get_function_input(
        "\nDefine F(x, y, y') where y'' = F(x, y, y')",
        "x, y, yp",
        "-(yp**2)/(y + 1e-4)"
    )
    
    # Ask for Fy and Fyp
    provide_partials = get_yes_no("\nProvide partial derivatives Fy and Fyp? (y/n): ")
    
    Fy_str = None
    Fyp_str = None
    if provide_partials:
        Fy_str = get_function_input(
            "Define ∂F/∂y (Fy)",
            "x, y, yp",
            "(yp**2)/((y + 1e-4)**2)"
        )
        Fyp_str = get_function_input(
            "Define ∂F/∂y' (Fyp)",
            "x, y, yp",
            "-(2*yp)/(y + 1e-4)"
        )
    
    # Analytical solution
    has_analytical = get_yes_no("\nDoes an analytical solution exist? (y/n): ")
    analytical_str = None
    if has_analytical:
        analytical_str = get_function_input(
            "Define analytical solution y(x)",
            "x",
            "1e-4*(np.sqrt(100020000*x + 1) - 1)"
        )
    
    # Solver parameters
    print("\nSolver Parameters:")
    tol = get_float_input("  Tolerance (default 1e-8, press Enter): ", allow_empty=True)
    if tol is None:
        tol = 1e-8
    
    max_iter = get_int_input("  Max iterations (default 100): ", min_val=1)
    if max_iter is None:
        max_iter = 100
    
    # Create functions
    safe_env = {"np": np}
    F_func = eval(f"lambda x, y, yp: {F_str}", safe_env)
    Fy_func = eval(f"lambda x, y, yp: {Fy_str}", safe_env) if Fy_str else None
    Fyp_func = eval(f"lambda x, y, yp: {Fyp_str}", safe_env) if Fyp_str else None
    analytical_func = eval(f"lambda x: {analytical_str}", safe_env) if analytical_str else None
    
    compute_error = has_analytical
    
    return N, domain, BC, F_func, Fy_func, Fyp_func, analytical_func, tol, max_iter, compute_error


def analysis_menu():
    """Display analysis options menu."""
    print("\n--- Analysis Options ---")
    print("Select tests to perform:")
    
    do_accuracy = get_yes_no("  1. Accuracy test? (y/n): ")
    do_stability = get_yes_no("  2. Stability test? (y/n): ")
    do_convergence = get_yes_no("  3. Convergence test? (y/n): ")
    
    return do_accuracy, do_stability, do_convergence


def output_menu(x, w, analytical_func):
    """Handle output options."""
    print("\n--- Output Options ---")
    
    # Print data
    print_data = get_yes_no("Print solution data to console? (y/n): ")
    if print_data:
        print("\nSolution Data:")
        print(f"{'x':>12} {'y':>15}")
        print("-" * 30)
        for xi, wi in zip(x, w):
            print(f"{xi:12.6f} {wi:15.8e}")
    
    # Save to CSV
    save_csv = get_yes_no("\nSave solution to CSV file? (y/n): ")
    if save_csv:
        filename = input("Enter filename (default 'solution.csv'): ").strip()
        if filename == "":
            filename = "solution.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = os.path.join("outputs", "data", filename)
        os.makedirs(os.path.join("outputs", "data"), exist_ok=True)
        
        if analytical_func is not None:
            y_exact = analytical_func(x)
            data = np.column_stack((x, w, y_exact))
            header = "x,y_numerical,y_analytical"
        else:
            data = np.column_stack((x, w))
            header = "x,y_numerical"
        
        np.savetxt(filepath, data, delimiter=',', header=header, comments='')
        print(f"Solution saved to: {filepath}")


def plot_solution(x, w, analytical_func, problem_desc="Solution"):
    """Plot numerical and analytical solutions."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, w, 'b-x', label='Numerical Solution', markersize=6)
    
    if analytical_func is not None:
        y_exact = analytical_func(x)
        plt.plot(x, y_exact, color='lightcoral', linestyle='--', label='Analytical Solution', linewidth=2)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'FDM Solution: {problem_desc}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    save_path = os.path.join("outputs", "plots", "solution.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def compute_and_display_error(x, w, analytical_func):
    """Compute and display error if analytical solution exists."""
    if analytical_func is None:
        return
    
    y_exact = analytical_func(x)
    error = np.abs(w - y_exact)
    max_error = np.max(error)
    
    print_separator("-")
    print(f"Maximum Error: {max_error:.6e}")
    print_separator("-")
    
    # Plot error
    plt.figure(figsize=(10, 5))
    plt.plot(x, error, 'g-o', markersize=4)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Error Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    error_path = os.path.join("outputs", "plots", "error.png")
    plt.savefig(error_path, dpi=150)
    print(f"Error plot saved to: {error_path}")
    plt.show()


def main():
    """Main driver function."""
    # Welcome message
    print_header()
    
    # Load configuration
    use_config = get_yes_no("Load configuration from configs/config.json? (y/n): ")
    
    if use_config:
        try:
            print("\nLoading configuration from configs/config.json...")
            N, domain, BC, F_func, Fy_func, Fyp_func, analytical_func, tol, max_iter, compute_error = load_config("configs/config.json")
            print("Configuration loaded successfully.\n")
        except Exception as e:
            print(f"\nError loading config file: {e}")
            print("Switching to manual input.\n")
            N, domain, BC, F_func, Fy_func, Fyp_func, analytical_func, tol, max_iter, compute_error = manual_input()
    else:
        N, domain, BC, F_func, Fy_func, Fyp_func, analytical_func, tol, max_iter, compute_error = manual_input()
    
    # Display configuration summary
    print("\n--- Configuration Summary ---")
    print(f"Domain: [{domain[0]}, {domain[1]}]")
    print(f"Grid intervals (N): {N}")
    print(f"Tolerance: {tol:.2e}")
    print(f"Max iterations: {max_iter}")
    print(f"BC Left:  {BC[0][0]}*y + {BC[0][1]}*y' + {BC[0][2]} = 0")
    print(f"BC Right: {BC[1][0]}*y + {BC[1][1]}*y' + {BC[1][2]} = 0")
    print()
    
    # Solve the problem
    print("\n--- Solving BVP ---")
    try:
        solver = FDM_Solver(N=N, domain=domain, F=F_func, Fy=Fy_func, Fyp=Fyp_func, BC=BC)
        w = solver.solver(tol=tol, max_iter=max_iter)
        x = np.linspace(domain[0], domain[1], N + 1)
        print("Solution completed successfully.\n")
    except Exception as e:
        print(f"\nError during solving: {e}")
        sys.exit(1)
    
    # Output handling
    output_menu(x, w, analytical_func)
    
    # Plotting
    plot_solution(x, w, analytical_func)
    
    # Error analysis
    if compute_error and analytical_func is not None:
        compute_and_display_error(x, w, analytical_func)
    # Analysis options
    do_accuracy, do_stability, do_convergence = analysis_menu()
    
    if do_accuracy:
        print("\nPerforming accuracy analysis...")
        acc_analysis = AccuracyAnalysis("configs/config.json", "analysis/y_ref.txt")
        acc_analysis.run()
    
    if do_stability:
        stab_solver = StabilitySolver(F_func, Fy_func, Fyp_func, N, domain, BC)
        w, it_fdm, hist_fdm = stab_solver.solve_fdm()

        if input("\nPlot stability metrics? (y/n): ").strip().lower() == "y":
            stab_solver.plot_fdm_all(hist_fdm)
    
    if do_convergence is None:
        do_convergence = input("\nDo you want to see convergence plots? (y/n)\n").lower() == 'y'
    if do_convergence:
        print("\nGenerating convergence plots...\n")
        study = ConvergenceStudy()
        study.run()
    else:
        print("\nSkipping convergence plots.\n")
    

    # Completion message
    print("\n")
    print_separator()
    print("Program completed successfully.".center(70))
    print_separator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
