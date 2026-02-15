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
from tqdm import tqdm

# =============================================================================
# PART 1: Problem Definition (HARDCODED)
# =============================================================================

# Domain parameters
X_START = 0.0
X_END = 1.0
N = 2**26  # number of interior grid points

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
    temp = y + 1e-4
    return (yp**2) / (temp * temp)


@njit
def Fyp(x, y, yp):
    """
    Partial derivative: dF/dy'
    From F = -(y')^2 / (y + 1e-4)
    dF/dy' = -2*y' / (y + 1e-4)
    """
    return -2.0 * yp / (y + 1e-4)


@njit
def analytical_solution(x):
    """Analytical solution for error checking"""
    return 1e-4 * (np.sqrt(100020000 * x + 1) - 1)


# =============================================================================
# PART 3: Core FDM Assembly (Fully @njit)
# =============================================================================

@njit
def build_residual(w, x, h, F, Fy, Fyp):
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


def newton_solve(w0, x, h, tol=1e-8, max_iter=100, show_progress=True):
    """
    Newton's method with progress bar (wrapper around jitted version)
    
    w0: initial guess
    x: grid points
    h: grid spacing
    tol: convergence tolerance
    max_iter: max iterations
    show_progress: if True, display tqdm progress bar
    """
    if show_progress:
        print("Running Newton solver with progress tracking...")
    
    # Manual iteration loop to show progress with tqdm
    w = w0.copy()
    with tqdm(total=max_iter, disable=not show_progress, desc="Newton iterations") as pbar:
        for iteration in range(max_iter):
            # Build residual and Jacobian
            r = build_residual(w, x, h, F, Fy, Fyp)
            u, l, d = build_jacobian(w, x, h, Fy, Fyp)
            
            # Solve J * delta = -r for delta
            delta = TDMA_solver(u, l, d, -r)
            
            # Update solution
            w = w + delta
            
            # Check convergence
            error = np.max(np.abs(delta))
            pbar.update(1)
            pbar.set_postfix({"error": f"{error:.2e}"})
            
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
    for i in range(len(w)):
        exact = analytical_solution(x[i])
        error = np.abs(w[i] - exact)
        if error > max_err:
            max_err = error
    return max_err


# =============================================================================
# PART 6: Main Execution
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FDM Solver with Numba @njit - BVP: y'' = -(y')^2 / (y + 1e-4)")
    print("="*70)
    
    # Setup grid
    x = np.linspace(X_START, X_END, N + 1)
    h = (X_END - X_START) / N
    
    print(f"\nGrid Configuration:")
    print(f"  Domain: [{X_START}, {X_END}]")
    print(f"  Number of points: {N + 1}")
    print(f"  Grid spacing: {h:.8e}")
    print(f"  BC: y({X_START}) = {BC_LEFT_VALUE}, y({X_END}) = {BC_RIGHT_VALUE}")
    
    # Initial guess (linear interpolation between BCs)
    w0 = np.linspace(BC_LEFT_VALUE, BC_RIGHT_VALUE, N + 1)
    
    print(f"\nSolver Configuration:")
    print(f"  Method: Newton with Numba @njit acceleration")
    print(f"  Tolerance: 1e-8")
    print(f"  Max iterations: 100")
    
    # First call compiles the jitted functions (warmup)
    print(f"\nCompiling with Numba (first call)...")
    t0 = time.time()
    w_solution, n_iters = newton_solve(w0, x, h, show_progress=False)
    compile_time = time.time() - t0
    
    print(f"  Compilation + Solve: {compile_time:.4f} seconds")
    print(f"  Converged in {n_iters} iterations")
    
    # Second call measures actual performance (no compilation overhead)
    print(f"\nSolving again (no compilation overhead)...")
    w0_fresh = np.linspace(BC_LEFT_VALUE, BC_RIGHT_VALUE, N + 1)
    t0 = time.time()
    w_solution, n_iters = newton_solve(w0_fresh, x, h, show_progress=True)
    solve_time = time.time() - t0
    
    print(f"  Pure compute time: {solve_time:.4f} seconds")
    print(f"  Converged in {n_iters} iterations")
    
    # Error analysis
    print(f"\nError Analysis:")
    max_error = compute_error(w_solution, x)
    print(f"  Max absolute error: {max_error:.8e}")
    
    # Print sample values
    print(f"\nSample Solution Values:")
    indices = [0, N//4, N//2, 3*N//4, N]
    for idx in indices:
        exact = analytical_solution(x[idx])
        print(f"  x={x[idx]:.4f}: numerical={w_solution[idx]:.8f}, " +
              f"exact={exact:.8f}, error={abs(w_solution[idx] - exact):.2e}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Numerical solution
    plt.subplot(1, 2, 1)
    plt.plot(x, w_solution, 'b-', label='Numerical (FDM + Numba)')
    x_fine = np.linspace(X_START, X_END, 1000)
    y_fine = analytical_solution(x_fine)
    plt.plot(x_fine, y_fine, 'r--', label='Analytical', linewidth=2, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error
    plt.subplot(1, 2, 2)
    errors = np.abs(w_solution - analytical_solution(x))
    plt.semilogy(x, errors, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Analytical Solution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fdm_njit_solution.png', dpi=150)
    print(f"\nPlot saved to: fdm_njit_solution.png")
    plt.show()
    
    print("\n" + "="*70)
    print("PERFORMANCE NOTES:")
    print("="*70)
    print("""
The @njit decorator creates compiled machine code (CPU JIT) which gives:
  - ~10-100x speedup vs pure Python (N=8192 typically ~0.01-0.1s)
  - First call is slower due to compilation overhead
  - Subsequent calls run at compiled speed

For EVEN FASTER execution with large N (100k+ points), consider:
  1. Use @cuda.jit for GPU acceleration (requires NVIDIA GPU + CUDA)
  2. Use parallel=True in @njit (multi-threaded CPU)
  3. Adjust problem: larger h or smaller N
  
GPU Code Template (requires: pip install numba[cuda]):
  from numba import cuda
  
  @cuda.jit
  def my_kernel(array):
      idx = cuda.grid(1)
      if idx < array.size:
          array[idx] *= 2
    """)
    print("="*70)
