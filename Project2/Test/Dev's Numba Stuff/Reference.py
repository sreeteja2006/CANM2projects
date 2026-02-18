"""
FDM Solver with Numba @njit for CPU JIT Compilation
=====================================================

This example demonstrates how to properly use @njit from Numba to accelerate
the Finite Difference Method (FDM) solver for Boundary Value Problems (BVPs).

Key Points:
-----------
1. @njit compiles the code to machine code at runtime (CPU JIT compilation)
   This gives 10-100x speedup compared to pure Python
   
2. ALL functions decorated with @njit must be "numba-safe":
   - Only NumPy arrays (not lists, dicts, custom objects)
   - Only basic Python operations
   - No .format(), f-strings with expressions, etc.
   
3. For ACTUAL GPU computation, use @cuda.jit (requires CUDA-capable GPU)
   This is shown in the comments at the end
   
4. The trick is to wrap all compute-heavy loops in @njit functions
   that take ONLY arrays and scalars as arguments

Problem Setup (hardcoded):
--------------------------
Boundary Value Problem: y'' = F(x, y, y')
Domain: [0, 1]
N: 8192 grid points
Left BC: y(0) = 0     (Dirichlet)
Right BC: y(1) = 1    (Dirichlet)
Equation: y'' = -(y')^2 / (y + 1e-4)
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
from pathlib import Path


# ensure output plots go into Project2/Plots regardless of where this script is run
PLOTS_DIR = Path(__file__).parent.parent.parent / "Plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PART 1: Problem Definition (HARDCODED)
# =============================================================================

# Domain parameters
X_START = 0.0
X_END = 1.0
N = 2**25  # number of interior grid points

# Boundary conditions (Dirichlet: y(x0) = c0, y(x1) = c1)
BC_LEFT_VALUE = 0.0      # y(0) = 0
BC_RIGHT_VALUE = 1.0     # y(1) = 1


# =============================================================================
# PART 2: PDE Functions (Must be @njit compatible)
# =============================================================================

@njit
def F(x, y, yp):
    """
    Main ODE: y'' = F(x, y, y')
    This is: y'' = -(y')^2 / (y + 1e-4)
    """
    return -(yp**2) / (y + 1e-4)


@njit
def Fy(x, y, yp):
    """
    Partial derivative: dF/dy
    From F = -(y')^2 / (y + 1e-4)
    dF/dy = (y')^2 / (y + 1e-4)^2
    """
    delta = 1e-9 * max(1.0, abs(y))  # relative perturbation
    return (F(x, y+delta, yp) - F(x, y-delta, yp)) / (2.0 * delta)


@njit
def Fyp(x, y, yp):
    """
    Partial derivative: dF/dy'
    From F = -(y')^2 / (y + 1e-4)
    dF/dy' = -2*y' / (y + 1e-4)
    """
    delta = 1e-9 * max(1.0, abs(yp))  # relative perturbation
    return (F(x, y, yp+delta) - F(x, y, yp-delta)) / (2.0 * delta)


@njit
def analytical_solution(x):
    """Analytical solution for error checking"""
    return 1e-4*(np.sqrt(100020000* x + 1) - 1)


# =============================================================================
# PART 3: Core FDM Assembly (Fully @njit)
# =============================================================================

@njit
def build_residual(w, x, h, F):
    """
    Build the residual vector r where we solve r(w) = 0
    
    For Dirichlet BCs and interior points:
    r[0] = y(x0) - BC_left
    r[i] = -y[i-1] + 2*y[i] - y[i+1] + h^2 * F(x[i], y[i], y'[i])   for i in 1..N-1
    r[N] = y(xN) - BC_right
    """
    N = len(w) - 1
    res = np.zeros(N + 1)
    
    # Left boundary (Dirichlet)
    res[0] = w[0] - BC_LEFT_VALUE
    
    # Right boundary (Dirichlet)
    res[N] = w[N] - BC_RIGHT_VALUE
    
    # Interior points
    for i in range(1, N):
        # Central difference approximation: y'[i] ≈ (y[i+1] - y[i-1]) / (2*h)
        yp_i = (w[i+1] - w[i-1]) / (2.0 * h)
        yi = w[i]
        xi = x[i]
        
        res[i] = -w[i-1] + 2.0*w[i] - w[i+1] + h**2 * F(xi, yi, yp_i)
    
    return res


@njit
def build_jacobian(w, x, h, Fy, Fyp):
    """
    Build the Jacobian matrix for Newton's method
    
    For a tridiagonal system (which our FDM produces), we only need:
    - u: upper diagonal
    - l: lower diagonal  
    - d: main diagonal
    
    Format: J[i, i-1] = l[i-1], J[i, i] = d[i], J[i, i+1] = u[i]
    """
    N = len(w) - 1
    
    u = np.zeros(N)      # upper diagonal (length N)
    l = np.zeros(N)      # lower diagonal (length N)
    d = np.zeros(N + 1)  # main diagonal (length N+1)
    
    # Row 0 (left BC, Dirichlet): d*y0 = value
    d[0] = 1.0
    u[0] = 0.0
    
    # Row N (right BC, Dirichlet): l*yN-1 + d*yN = value
    d[N] = 1.0
    l[N-1] = 0.0
    
    # Interior rows (Newton linearization of residual)
    for i in range(1, N):
        yp_i = (w[i+1] - w[i-1]) / (2.0 * h)
        yi = w[i]
        xi = x[i]
        
        # Jacobian of residual[i] w.r.t. w[i-1], w[i], w[i+1]
        # ∂r[i]/∂y[i-1] = -1 - (h/2) * Fyp * ∂yp/∂y[i-1]
        #               = -1 - (h/2) * Fyp * (-1/(2h))
        #               = -1 + (h/2) * Fyp / (2h)
        # Actually, simpler: ∂yp[i]/∂y[i-1] = -1/(2h)
        
        # From FDM: r[i] = -y[i-1] + 2*y[i] - y[i+1] + h^2*F(...)
        # ∂r[i]/∂y[i-1] = -1 - (h^2/2)*Fyp*(-1/(2h)) = -1 + (h*Fyp)/4
        # Wait, let me reconsider:
        # yp_i = (y[i+1] - y[i-1])/(2h)
        # ∂yp_i/∂y[i-1] = -1/(2h)
        # ∂yp_i/∂y[i+1] = 1/(2h)
        # ∂F/∂y[i-1] = Fy * ∂y/∂y[i-1] + Fyp * ∂yp/∂y[i-1] = 0 + Fyp * (-1/(2h))
        
        fyp_val = Fyp(xi, yi, yp_i)
        fy_val = Fy(xi, yi, yp_i)
        
        l[i-1] = -1.0 - (h / 2.0) * fyp_val * (-1.0 / (2.0 * h))
        l[i-1] = -1.0 + (h**2) * fy_val / (4.0)  # Corrected
        
        u[i] = -1.0 - (h / 2.0) * fyp_val * (1.0 / (2.0 * h))
        u[i] = -1.0 - (h**2) * fy_val / (4.0)
        
        d[i] = 2.0 + h**2 * fy_val
    
    return u, l, d


@njit
def TDMA_solver(u, l, d, b):
    """
    Tridiagonal Matrix Algorithm (Thomas Algorithm)
    Solves the system: Ax = b where A is tridiagonal
    
    u: upper diagonal (length N, represents superdiagonal)
    l: lower diagonal (length N, represents subdiagonal)
    d: main diagonal (length N+1)
    b: right-hand side (length N+1)
    """
    N = len(d) - 1
    
    # Forward elimination
    x = np.zeros(N + 1)
    P = np.zeros(N)  # Thomas algorithm workspace
    Q = np.zeros(N + 1)
    
    # First row
    P[0] = u[0] / d[0]
    Q[0] = b[0] / d[0]
    
    # Rows 1 to N-1
    for i in range(1, N):
        denom = d[i] - l[i-1] * P[i-1]
        if denom == 0.0:
            denom = 1e-15  # avoid division by zero
        P[i] = u[i] / denom
        Q[i] = (b[i] - l[i-1] * Q[i-1]) / denom
    
    # Last row
    denom = d[N] - l[N-1] * P[N-1]
    if denom == 0.0:
        denom = 1e-15
    Q[N] = (b[N] - l[N-1] * Q[N-1]) / denom
    
    # Back substitution
    x[N] = Q[N]
    for i in range(N-1, -1, -1):
        x[i] = Q[i] - P[i] * x[i+1]
    
    return x


# =============================================================================
# PART 4: Newton's Method Wrapped in @njit
# =============================================================================

@njit
def newton_solve_jit(w0, x, h, tol=1e-8, max_iter=100):
    """
    Newton's method for solving the nonlinear BVP (jitted kernel)
    
    w0: initial guess
    x: grid points
    h: grid spacing
    tol: convergence tolerance
    max_iter: max iterations
    """
    w = w0.copy()
    
    for iteration in range(max_iter):
        # Build residual and Jacobian
        # print(iteration)
        r = build_residual(w, x, h, F, Fy, Fyp)
        u, l, d = build_jacobian(w, x, h, Fy, Fyp)
        
        # Solve J * delta = -r for delta
        delta = TDMA_solver(u, l, d, -r)
        
        # Update solution
        w = w + delta
        # Check convergence
        error = np.max(np.abs(delta))
        if error < tol:
            return w, iteration + 1
    
    return w, max_iter





# =============================================================================
# PART 5: Analytical Solution for Error Analysis
# =============================================================================

@njit
def compute_error(w, x):
    """Compute max absolute error against analytical solution"""
    max_err = 0.0
    rms_err = 0.0
    for i in range(len(w)):
        exact = analytical_solution(x[i])
        error = np.abs(w[i] - exact)
        if error > max_err:
            max_err = error
        rms_err += error**2
    rms_err = np.sqrt(rms_err / len(w))
    return max_err, rms_err


# =============================================================================
# PART 6: Main Execution
# =============================================================================

if __name__ == "__main__":
    print("FDM Solver with Numba @njit to reduce runtime - BVP: y'' = -(y')^2 / (y + 1e-4)")
    
    # Setup grid
    x = np.linspace(X_START, X_END, N + 1)
    h = (X_END - X_START) / N
    
    # print(f"\nGrid Configuration:")
    # print(f"  Domain: [{X_START}, {X_END}]")
    # print(f"  Number of points: {N + 1}")
    # print(f"  Grid spacing: {h:.8e}")
    # print(f"  BC: y({X_START}) = {BC_LEFT_VALUE}, y({X_END}) = {BC_RIGHT_VALUE}")
    
    # Initial guess (linear interpolation between BCs)
    w0 = np.linspace(BC_LEFT_VALUE, BC_RIGHT_VALUE, N + 1)
    
    # print(f"\nSolver Configuration:")
    # print(f"  Method: Newton with Numba @njit acceleration")
    # print(f"  Tolerance: 1e-8")
    # print(f"  Max iterations: 100")
    
    # First call compiles the jitted functions (warmup)
    # print(f"\nCompiling with Numba (first call)...")
    # t0 = time.time()
    w_solution, n_iters = newton_solve_jit(w0, x, h)
    # compile_time = time.time() - t0
    
    # print(f"  Compilation + Solve: {compile_time:.4f} seconds")
    print(f"  Converged in {n_iters} iterations")
    
    # Error analysis
    print(f"\nError Analysis:")
    max_error, rms_error = compute_error(w_solution, x)
    print(f"  Max absolute error: {max_error:.8e}")
    print(f"  RMS error: {rms_error:.8e}") 
    
    # Plotting
    plt.figure(figsize=(10, 5))
    errors = np.abs(w_solution - analytical_solution(x))
    plt.semilogy(x, errors, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Reference Error')
    plt.title('Reference Solution Absolute Error (log scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # save into Project2 Plots folder
    save_path = PLOTS_DIR / 'Reference_Solution_Error.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    # Save final solution vector to Analysis folder
    ANALYSIS_DIR = Path(__file__).parent.parent.parent / "Analysis"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    solution_file = ANALYSIS_DIR / 'y_ref.txt'
    np.savetxt(solution_file, w_solution[::2**10])  # save every 1024th point to reduce file size
    print(f"Solution vector saved to: {solution_file}")