import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import sys

# Ensure workspace root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.Loader import load_config

# Ensure output plots go into Project2/outputs/plots
PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

YREF_DIR = Path(__file__).parent
YREF_DIR.mkdir(parents=True, exist_ok=True)
# =============================================================================
# PART 1: LOAD CONFIGURATION
# =============================================================================
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
(
    _, domain, BC,
    F_func, Fy_func, Fyp_func,
    analytical_solution,
    tol_config, max_iter_config,
    compute_error_flag
) = load_config(config_path)
N = 25000000
X_START, X_END = domain

# Extract BC coefficients as plain Python floats.
# These are passed explicitly into @njit functions — never read as globals.
BC_LEFT_A  = float(BC[0, 0])
BC_LEFT_B  = float(BC[0, 1])
BC_LEFT_C  = float(BC[0, 2])
BC_RIGHT_A = float(BC[1, 0])
BC_RIGHT_B = float(BC[1, 1])
BC_RIGHT_C = float(BC[1, 2])


# =============================================================================
# PART 2: BUILD @njit-COMPILED F, Fy, Fyp DISPATCHERS
#
# Rules enforced here:
#   (a) F_func is a plain Python callable from load_config.
#       njit(F_func) compiles it to a CPUDispatcher Numba can call.
#
#   (b) Fy_func / Fyp_func may be None.
#       We resolve the None branch HERE in plain Python and always produce
#       a concrete @njit function. This means @njit kernels downstream
#       never have to test for None (which Numba cannot type).
#
#   (c) The fallback numerical Fy/Fyp approximations call F_jit, which is
#       already a CPUDispatcher, so they are fully @njit safe.
# =============================================================================

F_jit = njit(F_func)

if Fy_func is not None:
    Fy_jit = njit(Fy_func)
else:
    @njit
    def Fy_jit(x, y, yp):
        """Numerical central-difference approximation of dF/dy."""
        eps = 1e-7 * max(1.0, abs(y))
        if eps < 1e-12:
            eps = 1e-7
        return (F_jit(x, y + eps, yp) - F_jit(x, y - eps, yp)) / (2.0 * eps)

if Fyp_func is not None:
    Fyp_jit = njit(Fyp_func)
else:
    @njit
    def Fyp_jit(x, y, yp):
        """Numerical central-difference approximation of dF/dy'."""
        eps = 1e-7 * max(1.0, abs(yp))
        if eps < 1e-12:
            eps = 1e-7
        return (F_jit(x, y, yp + eps) - F_jit(x, y, yp - eps)) / (2.0 * eps)


# =============================================================================
# PART 3: BOUNDARY CONDITION HELPERS
#
# Mirrors Boundary_Conditions class.
# General BC: a*y + b*y' + c = 0
#
# Residuals:
#   Left:  y'(x0) approximated by forward  difference: (w[1]  - w[0])  / h
#   Right: y'(xN) approximated by backward difference: (w[N]  - w[N-1])/ h
#
# Jacobian entries (from differentiating the residual):
#   Left row 0:
#     d(res)/d(w[0]) = a - b/h   -> main diagonal d[0]
#     d(res)/d(w[1]) = b/h       -> super-diagonal u[0]
#   Right row N:
#     d(res)/d(w[N-1]) = -b/h   -> sub-diagonal  l[N-1]
#     d(res)/d(w[N])   = a+b/h  -> main diagonal  d[N]
#
# All BC coefficients (a, b, c) passed as explicit scalar arguments.
# =============================================================================

@njit
def bc_left_residual(w0, w1, h, a, b, c):
    """Residual for left BC: a*w[0] + b*(w[1]-w[0])/h + c = 0"""
    yp0 = (w1 - w0) / h
    return a * w0 + b * yp0 + c


@njit
def bc_right_residual(wNm1, wN, h, a, b, c):
    """Residual for right BC: a*w[N] + b*(w[N]-w[N-1])/h + c = 0"""
    ypN = (wN - wNm1) / h
    return a * wN + b * ypN + c


@njit
def bc_left_jac_coeffs(h, a, b):
    """Jacobian coefficients for left BC row."""
    d0 = a - b / h
    u0 = b / h
    return d0, u0


@njit
def bc_right_jac_coeffs(h, a, b):
    """Jacobian coefficients for right BC row."""
    lNm1 = -b / h
    dN   =  a + b / h
    return lNm1, dN


# =============================================================================
# PART 4: INTERIOR ROW HELPERS
#
# Mirrors Function_Generator.build_tridiagonal_terms() and
# Function_Generator.build_residual_function().
#
# Interior FDM residual for row i:
#   r[i] = -w[i-1] + 2*w[i] - w[i+1] + h^2 * F(x[i], w[i], yp[i])
#   where yp[i] = (w[i+1] - w[i-1]) / (2*h)  (central difference)
#
# Newton linearisation (Jacobian entries for row i):
#   fl[i-1] = dr[i]/dw[i-1] = -1 - (h/2)*Fyp
#   fd[i]   = dr[i]/dw[i]   =  2 + h^2*Fy
#   fu[i]   = dr[i]/dw[i+1] = -1 + (h/2)*Fyp
#
# F, Fy, Fyp are @njit CPUDispatchers passed as arguments.
# =============================================================================

@njit
def interior_residual(F, xi, yi, yp_i, wim1, wi, wip1, h):
    """Interior residual for row i."""
    return -wim1 + 2.0 * wi - wip1 + h**2 * F(xi, yi, yp_i)


@njit
def interior_jac_coeffs(Fy, Fyp, xi, yi, yp_i, h):
    """Jacobian tridiagonal entries for interior row i."""
    fy_val  = Fy(xi, yi, yp_i)
    fyp_val = Fyp(xi, yi, yp_i)
    fl_i = -1.0 - (h / 2.0) * fyp_val
    fd_i =  2.0 + h**2 * fy_val
    fu_i = -1.0 + (h / 2.0) * fyp_val
    return fl_i, fd_i, fu_i


# =============================================================================
# PART 5: FULL RESIDUAL VECTOR
#
# Mirrors Residual.build(w, x).
# Layout:
#   r[0]      = left  BC residual
#   r[1..N-1] = interior FDM residuals
#   r[N]      = right BC residual
# =============================================================================

@njit
def build_residual(F, w, x, h,
                   bc_la, bc_lb, bc_lc,
                   bc_ra, bc_rb, bc_rc):
    """Build the full residual vector r(w)."""
    N = len(w) - 1
    r = np.zeros(N + 1)

    # Left boundary condition
    r[0] = bc_left_residual(w[0], w[1], h, bc_la, bc_lb, bc_lc)

    # Interior points
    for i in range(1, N):
        yp_i = (w[i + 1] - w[i - 1]) / (2.0 * h)
        r[i] = interior_residual(F, x[i], w[i], yp_i, w[i - 1], w[i], w[i + 1], h)

    # Right boundary condition
    r[N] = bc_right_residual(w[N - 1], w[N], h, bc_ra, bc_rb, bc_rc)

    return r


# =============================================================================
# PART 6: FULL JACOBIAN (TRIDIAGONAL)
#
# Mirrors Jacobian.build(w, x).
# Returns three arrays for the tridiagonal system:
#   u : super-diagonal, length N   (u[i] = J[i,   i+1])
#   l : sub-diagonal,   length N   (l[i] = J[i+1, i])
#   d : main diagonal,  length N+1 (d[i] = J[i,   i])
# =============================================================================

@njit
def build_jacobian(Fy, Fyp, w, x, h,
                   bc_la, bc_lb,
                   bc_ra, bc_rb):
    """Build the tridiagonal Jacobian of the residual."""
    N = len(w) - 1

    u = np.zeros(N)
    l = np.zeros(N)
    d = np.zeros(N + 1)

    # Left BC row (row 0)
    d[0], u[0] = bc_left_jac_coeffs(h, bc_la, bc_lb)

    # Interior rows (rows 1 .. N-1)
    for i in range(1, N):
        yp_i = (w[i + 1] - w[i - 1]) / (2.0 * h)
        fl_i, fd_i, fu_i = interior_jac_coeffs(Fy, Fyp, x[i], w[i], yp_i, h)
        l[i - 1] = fl_i
        d[i]     = fd_i
        u[i]     = fu_i

    # Right BC row (row N)
    l[N - 1], d[N] = bc_right_jac_coeffs(h, bc_ra, bc_rb)

    return u, l, d


# =============================================================================
# PART 7: TDMA (THOMAS ALGORITHM)
#
# Mirrors the TDMA module used in FDM_Solver.solver().
# Solves A*x = b where A is tridiagonal:
#   u : super-diagonal (length N,   u[i] = A[i,   i+1])
#   l : sub-diagonal   (length N,   l[i] = A[i+1, i])
#   d : main diagonal  (length N+1, d[i] = A[i,   i])
#   b : RHS vector     (length N+1)
# =============================================================================

@njit
def TDMA_solver(u, l, d, b):
    """Tridiagonal Matrix Algorithm (Thomas Algorithm)."""
    N = len(d) - 1
    x = np.zeros(N + 1)
    P = np.zeros(N)
    Q = np.zeros(N + 1)

    # Forward sweep — first row
    denom = d[0]
    if denom == 0.0:
        denom = 1e-15
    P[0] = u[0] / denom
    Q[0] = b[0] / denom

    # Forward sweep — rows 1 .. N-1
    for i in range(1, N):
        denom = d[i] - l[i - 1] * P[i - 1]
        if denom == 0.0:
            denom = 1e-15
        P[i] = u[i] / denom
        Q[i] = (b[i] - l[i - 1] * Q[i - 1]) / denom

    # Forward sweep — last row
    denom = d[N] - l[N - 1] * P[N - 1]
    if denom == 0.0:
        denom = 1e-15
    Q[N] = (b[N] - l[N - 1] * Q[N - 1]) / denom

    # Back substitution
    x[N] = Q[N]
    for i in range(N - 1, -1, -1):
        x[i] = Q[i] - P[i] * x[i + 1]

    return x


# =============================================================================
# PART 8: NEWTON'S METHOD
#
# Mirrors FDM_Solver.solver():
#   - Builds Jacobian and residual each iteration
#   - Solves J * delta = -r via TDMA
#   - Convergence check: Norm_inf(delta) = max|delta| < tol
#   - Returns (w, iterations, converged)
#
# All functions and BC scalars passed explicitly — no global captures.
# =============================================================================

@njit
def newton_solve_jit(F, Fy, Fyp,
                     w0, x, h,
                     bc_la, bc_lb, bc_lc,
                     bc_ra, bc_rb, bc_rc,
                     tol=1e-6, max_iter=100):
    """
    Newton's method for the nonlinear BVP.
    Mirrors FDM_Solver.solver() exactly.
    """
    w = w0.copy()
    converged  = False
    norm_delta = np.inf

    for k in range(max_iter):
        r = build_residual(
            F, w, x, h,
            bc_la, bc_lb, bc_lc,
            bc_ra, bc_rb, bc_rc
        )
        u, l, d = build_jacobian(
            Fy, Fyp, w, x, h,
            bc_la, bc_lb,
            bc_ra, bc_rb
        )
        delta = TDMA_solver(u, l, d, -r)
        w = w + delta

        norm_delta = np.max(np.abs(delta))
        if norm_delta < tol:
            converged = True
            return w, k + 1, converged

    return w, max_iter, converged


# =============================================================================
# PART 9: ERROR ANALYSIS (plain Python — analytical_solution is a Python callable)
# =============================================================================

def compute_error(w, x):
    """
    Max absolute error and RMS error vs analytical solution.
    Returns (max_err, rms_err) or (None, None) if no analytical solution.
    """
    if analytical_solution is None:
        return None, None

    exact  = np.array([analytical_solution(xi) for xi in x])
    errors = np.abs(w - exact)
    return float(np.max(errors)), float(np.sqrt(np.mean(errors**2)))


def compute_analytical_error(x, h):
    """
    Analytical truncation error estimate.
    Returns array or None if no analytical solution.
    """
    if analytical_solution is None:
        return None

    b_c = 0.0001
    a_c = 1.0 + 2.0 * b_c
    C_1 = 1.0 / b_c**2 - 1.0 / (1.0 + b_c)**2
    C_2 = -1.0 / b_c**2

    out = np.zeros(len(x))
    for idx, xi in enumerate(x):
        y_val = analytical_solution(xi)
        denom = y_val + b_c
        if abs(denom) < 1e-10:
            out[idx] = 0.0
        else:
            out[idx] = (h**2) * (1.0 / denom) * (a_c**2 / 128.0) * (
                (1.0 / (a_c * xi + b_c**2)) + C_1 * xi + C_2
            )
    return out


# =============================================================================
# PART 10: WARM-UP HELPERS
#
# Numba compiles every @njit function on its first call.
# We warm up on a tiny 10-interval problem so the reported solve time
# reflects only computation, not compilation overhead.
#
# Warm-up functions are self-contained (no dependency on config values)
# so they compile independently from the main solve path.
# =============================================================================

# @njit
# def _F_warm(x, y, yp):
#     return -(yp * yp) / (y + 1e-4)

# @njit
# def _Fy_warm(x, y, yp):
#     eps = 1e-7 * max(1.0, abs(y))
#     if eps < 1e-12:
#         eps = 1e-7
#     return (_F_warm(x, y + eps, yp) - _F_warm(x, y - eps, yp)) / (2.0 * eps)

# @njit
# def _Fyp_warm(x, y, yp):
#     eps = 1e-7 * max(1.0, abs(yp))
#     if eps < 1e-12:
#         eps = 1e-7
#     return (_F_warm(x, y, yp + eps) - _F_warm(x, y, yp - eps)) / (2.0 * eps)


# =============================================================================
# PART 11: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # print("FDM Solver with Numba @njit (aligned with FDM_Solver class)")
    # print("=" * 60)
    # print(f"Domain:      [{X_START}, {X_END}]")
    # print(f"Grid points: {N + 1}  (N = {N} intervals)")
    # print(f"Left  BC:    {BC_LEFT_A}*y + {BC_LEFT_B}*y' + {BC_LEFT_C} = 0")
    # print(f"Right BC:    {BC_RIGHT_A}*y + {BC_RIGHT_B}*y' + {BC_RIGHT_C} = 0")
    # print(f"Config:      {config_path}\n")

    # Setup grid
    x = np.linspace(X_START, X_END, N + 1)
    h = (X_END - X_START) / N

    # Initial guess — same default as FDM_Solver: linspace over domain
    w0 = np.linspace(X_START, X_END, N + 1)

    # ------------------------------------------------------------------
    # Warm-up: trigger JIT compilation on a tiny problem
    # ------------------------------------------------------------------
    # print("Warming up Numba JIT (compiling all kernels)...")
    # _x_w = np.linspace(0.0, 1.0, 11)
    # _w_w = np.linspace(0.0, 1.0, 11)
    # _h_w = 0.1
    # newton_solve_jit(
    #     _F_warm, _Fy_warm, _Fyp_warm,
    #     _w_w, _x_w, _h_w,
    #     1.0, 0.0,  0.0,   # left  BC: y(0) = 0
    #     1.0, 0.0, -1.0,   # right BC: y(1) = 1
    #     tol=1e-6, max_iter=5
    # )
    # print("JIT warm-up complete.\n")

    # ------------------------------------------------------------------
    # Solve the actual problem
    # ------------------------------------------------------------------
    # print(f"Solving with tolerance={tol_config}, max_iter={max_iter_config}")
    t0 = time.time()
    w_solution, n_iters, converged = newton_solve_jit(
        F_jit, Fy_jit, Fyp_jit,
        w0, x, h,
        BC_LEFT_A,  BC_LEFT_B,  BC_LEFT_C,
        BC_RIGHT_A, BC_RIGHT_B, BC_RIGHT_C,
        tol=tol_config, max_iter=max_iter_config
    )
    solve_time = time.time() - t0

    if converged:
        print(f"Converged in {n_iters} iterations ({solve_time:.4f} seconds)\n")
    else:
        print(
            f"WARNING: Did not converge in {n_iters} iterations "
            f"({solve_time:.4f} seconds).\n"
        )
    ref_path = YREF_DIR / "y_ref.txt"
    np.savetxt(ref_path, w_solution[::250], fmt="%.15e")
    # print(f"Reference solution saved to: {ref_path}\n")
    # ------------------------------------------------------------------
    # Error analysis and plotting
    # ------------------------------------------------------------------
    if analytical_solution is not None:
        # print("Error Analysis:")
        max_error, rms_error = compute_error(w_solution, x)
        print(f"  Max absolute error: {max_error:.8e}")
        print(f"  RMS error:          {rms_error:.8e}\n")

        numerical_errors = np.abs(
            w_solution - np.array([analytical_solution(xi) for xi in x])
        )

        plt.figure(figsize=(8.85, 6))
        plt.semilogy(x, numerical_errors, 'g-', linewidth=2, label='Numerical Error')
        plt.xlabel('x')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error (log scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = PLOTS_DIR / 'AbsoluteError.png'
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
        plt.show()

        plt.figure(figsize=(8.85, 6))
        plt.semilogy(x, numerical_errors, 'g-', linewidth=2, label='Numerical Error')
        try:
            analytical_errors = np.abs(compute_analytical_error(x, h))
            if analytical_errors is not None:
                valid = (
                    (~np.isnan(analytical_errors))
                    & (~np.isinf(analytical_errors))
                    & (analytical_errors > 0)
                )
                if np.any(valid):
                    plt.semilogy(
                        x[valid], analytical_errors[valid],
                        'r--', linewidth=2, label='Analytical Error Estimate'
                    )
                else:
                    print("  Warning: Analytical error estimate produced no valid values.")
        except Exception as e:
            print(f"  Warning: Could not compute analytical error: {e}")

        plt.xlabel('x')
        plt.ylabel('Absolute Error')
        plt.title('Error Comparison (log scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = PLOTS_DIR / 'ErrorsComparison.png'
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
        plt.show()

    else:
        print("No analytical solution available\n")