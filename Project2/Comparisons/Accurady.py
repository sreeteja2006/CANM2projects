from numba import njit
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Core.Loader import load_config

config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
(
    _, domain, BC,
    F_func, Fy_func, Fyp_func,
    _,
    tol_config, max_iter_config,
    _
) = load_config(config_path)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Project2.Core.FDM_Solver import FDM_Solver

PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ODE functions
# =============================================================================

f_2 = njit(F_func)

# @njit
# def f_2(x, u_0, u_1):
#     return -(u_1)**2/(u_0+1e-4)
@njit
def f_1(x, u_0, u_1):
    return u_1

# =============================================================================
# RK2
# =============================================================================

@njit
def RK2(x, nx, h, y_0, ydash_0):
    u_0 = np.zeros(nx + 1)
    u_1 = np.zeros(nx + 1)
    u_0[0] = y_0
    u_1[0] = ydash_0
    for i in range(nx):
        u0_i = u_0[i]
        u1_i = u_1[i]
        k1_1 = h * f_1(x[i], u0_i, u1_i)
        k1_2 = h * f_2(x[i], u0_i, u1_i)
        u0_k2 = u0_i + k1_1
        u1_k2 = u1_i + k1_2
        k2_1 = h * f_1(x[i] + h, u0_k2, u1_k2)
        k2_2 = h * f_2(x[i] + h, u0_k2, u1_k2)
        u_0[i + 1] = u0_i + 0.5 * (k1_1 + k2_1)
        u_1[i + 1] = u1_i + 0.5 * (k1_2 + k2_2)
    return u_0, u_1

# =============================================================================
# RK4
# =============================================================================

@njit
def RK4(x, nx, h, y_0, ydash_0):
    u_0 = np.zeros(nx + 1)
    u_1 = np.zeros(nx + 1)
    u_0[0] = y_0
    u_1[0] = ydash_0
    for i in range(nx):
        u0_i = u_0[i]
        u1_i = u_1[i]
        k1_1 = h * f_1(x[i], u0_i, u1_i)
        k1_2 = h * f_2(x[i], u0_i, u1_i)
        u0_k2 = u0_i + 0.5 * k1_1
        u1_k2 = u1_i + 0.5 * k1_2
        k2_1 = h * f_1(x[i] + 0.5 * h, u0_k2, u1_k2)
        k2_2 = h * f_2(x[i] + 0.5 * h, u0_k2, u1_k2)
        u0_k3 = u0_i + 0.5 * k2_1
        u1_k3 = u1_i + 0.5 * k2_2
        k3_1 = h * f_1(x[i] + 0.5 * h, u0_k3, u1_k3)
        k3_2 = h * f_2(x[i] + 0.5 * h, u0_k3, u1_k3)
        u0_k4 = u0_i + k3_1
        u1_k4 = u1_i + k3_2
        k4_1 = h * f_1(x[i] + h, u0_k4, u1_k4)
        k4_2 = h * f_2(x[i] + h, u0_k4, u1_k4)
        u_0[i + 1] = u0_i + (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6.0
        u_1[i + 1] = u1_i + (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6.0
    return u_0, u_1

# =============================================================================
# ABM2
# =============================================================================

@njit
def ABM2(x, nx, h, y_0, ydash_0):
    u_0 = np.zeros(nx + 1)
    u_1 = np.zeros(nx + 1)
    u_0[0] = y_0
    u_1[0] = ydash_0
    # Startup: RK2 for step 0->1
    u0_i  = u_0[0]
    u1_i  = u_1[0]
    k1_1  = h * f_1(x[0], u0_i, u1_i)
    k1_2  = h * f_2(x[0], u0_i, u1_i)
    u0_k2 = u0_i + k1_1
    u1_k2 = u1_i + k1_2
    k2_1  = h * f_1(x[0] + h, u0_k2, u1_k2)
    k2_2  = h * f_2(x[0] + h, u0_k2, u1_k2)
    u_0[1] = u0_i + 0.5 * (k1_1 + k2_1)
    u_1[1] = u1_i + 0.5 * (k1_2 + k2_2)
    for i in range(1, nx):
        f1_i   = f_1(x[i],     u_0[i],     u_1[i])
        f2_i   = f_2(x[i],     u_0[i],     u_1[i])
        f1_im1 = f_1(x[i - 1], u_0[i - 1], u_1[i - 1])
        f2_im1 = f_2(x[i - 1], u_0[i - 1], u_1[i - 1])
        u0_pred = u_0[i] + h * (3.0 * f1_i - f1_im1) / 2.0
        u1_pred = u_1[i] + h * (3.0 * f2_i - f2_im1) / 2.0
        f1_pred = f_1(x[i + 1], u0_pred, u1_pred)
        f2_pred = f_2(x[i + 1], u0_pred, u1_pred)
        u_0[i + 1] = u_0[i] + h * (f1_pred + f1_i) / 2.0
        u_1[i + 1] = u_1[i] + h * (f2_pred + f2_i) / 2.0
    return u_0, u_1

# =============================================================================
# ABM4
# =============================================================================

@njit
def ABM4(x, nx, h, y_0, ydash_0):
    u_0 = np.zeros(nx + 1)
    u_1 = np.zeros(nx + 1)
    u_0[0] = y_0
    u_1[0] = ydash_0
    # Startup: RK4 for steps 0->1, 1->2, 2->3
    for s in range(3):
        u0_i  = u_0[s]
        u1_i  = u_1[s]
        k1_1  = h * f_1(x[s], u0_i, u1_i)
        k1_2  = h * f_2(x[s], u0_i, u1_i)
        u0_k2 = u0_i + 0.5 * k1_1
        u1_k2 = u1_i + 0.5 * k1_2
        k2_1  = h * f_1(x[s] + 0.5 * h, u0_k2, u1_k2)
        k2_2  = h * f_2(x[s] + 0.5 * h, u0_k2, u1_k2)
        u0_k3 = u0_i + 0.5 * k2_1
        u1_k3 = u1_i + 0.5 * k2_2
        k3_1  = h * f_1(x[s] + 0.5 * h, u0_k3, u1_k3)
        k3_2  = h * f_2(x[s] + 0.5 * h, u0_k3, u1_k3)
        u0_k4 = u0_i + k3_1
        u1_k4 = u1_i + k3_2
        k4_1  = h * f_1(x[s] + h, u0_k4, u1_k4)
        k4_2  = h * f_2(x[s] + h, u0_k4, u1_k4)
        u_0[s + 1] = u0_i + (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6.0
        u_1[s + 1] = u1_i + (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6.0
    for i in range(3, nx):
        f1_i   = f_1(x[i],     u_0[i],     u_1[i])
        f2_i   = f_2(x[i],     u_0[i],     u_1[i])
        f1_im1 = f_1(x[i - 1], u_0[i - 1], u_1[i - 1])
        f2_im1 = f_2(x[i - 1], u_0[i - 1], u_1[i - 1])
        f1_im2 = f_1(x[i - 2], u_0[i - 2], u_1[i - 2])
        f2_im2 = f_2(x[i - 2], u_0[i - 2], u_1[i - 2])
        f1_im3 = f_1(x[i - 3], u_0[i - 3], u_1[i - 3])
        f2_im3 = f_2(x[i - 3], u_0[i - 3], u_1[i - 3])
        u0_pred = u_0[i] + h * (55.0 * f1_i - 59.0 * f1_im1 + 37.0 * f1_im2 -  9.0 * f1_im3) / 24.0
        u1_pred = u_1[i] + h * (55.0 * f2_i - 59.0 * f2_im1 + 37.0 * f2_im2 -  9.0 * f2_im3) / 24.0
        f1_pred = f_1(x[i + 1], u0_pred, u1_pred)
        f2_pred = f_2(x[i + 1], u0_pred, u1_pred)
        u_0[i + 1] = u_0[i] + h * (9.0 * f1_pred + 19.0 * f1_i - 5.0 * f1_im1 + f1_im2) / 24.0
        u_1[i + 1] = u_1[i] + h * (9.0 * f2_pred + 19.0 * f2_i - 5.0 * f2_im1 + f2_im2) / 24.0
    return u_0, u_1

# =============================================================================
# Secant Method (Shooting) — with robustness guards
# =============================================================================

def SecantMethod(x, nx, h, y_0, ydash1_0, ydash2_0, y_1, tol=1e-6, N=100, Method=RK4, eps=1e-3):
    x_0 = eps
    s_0 = ydash1_0
    s_1 = ydash2_0
    def F(s):
        y_0_scaled = s * eps
        u0, _ = Method(x, nx, h, y_0_scaled, s)
        return u0[nx] - y_1
    F0 = F(s_0)
    F1 = F(s_1)

    no_of_iterations = 0
    for _ in range(N):
        if abs(F1) <= tol:
            break

        denom = F1 - F0

        # Guard: denominator too small => secant step undefined
        if abs(denom) < 1e-14:
            print(f"    WARNING: Secant denominator near zero ({denom:.2e}), stopping early")
            break

        s_2 = s_1 - ((F1) * (s_1 - s_0)) / denom
        s_0, s_1 = s_1, s_2
        F0, F1 = F1, F(s_1)
        no_of_iterations += 1

    ystart = s_1*eps
    sol,x = RK4(x, nx, h, ystart, s_1)
    print(f"    Secant iterations: {no_of_iterations},  final BC error: {abs(y_1 - sol[nx]):.2e}")
    return sol,x

# =============================================================================
# Load reference solution
# =============================================================================

ref_path     = os.path.join(os.path.dirname(__file__),"..","Analysis", "y_ref.txt")
y_ref        = np.loadtxt(ref_path)
x_ref        = np.linspace(domain[0], domain[1], len(y_ref))

y_left  = -BC[0, 2] / BC[0, 0] if BC[0, 0] != 0 else 0.0
y_right = -BC[1, 2] / BC[1, 0] if BC[1, 0] != 0 else 1.0
eps     = 1e-3

ydash1_0 = 100.0
ydash2_0 = 200.0

shooting_methods = {
    "RK2":  RK2,
    "RK4":  RK4,
    "ABM2": ABM2,
    "ABM4": ABM4,
}

method_styles = {
    "FDM":  {"color": "#1f77b4", "lw": 2.0, "ls": "-"},
    "RK2":  {"color": "#ff7f0e", "lw": 1.5, "ls": "--"},
    "RK4":  {"color": "#2ca02c", "lw": 1.5, "ls": "-."},
    "ABM2": {"color": "#d62728", "lw": 1.5, "ls": ":"},
    "ABM4": {"color": "#9467bd", "lw": 1.5, "ls": (0, (3, 1, 1, 1))},
}

fdm_solver = FDM_Solver(F=F_func, Fy=Fy_func, Fyp=Fyp_func,
                        N=2, domain=domain, BC=BC)

# =============================================================================
# Compute all errors
# =============================================================================

Ns = [1000, 10000, 100000, 1000000]

# all_errors[N][method_name] = (x_grid, abs_error_array)
all_errors = {N: {} for N in Ns}

for N in Ns:
    print(f"\n{'='*50}")
    print(f"  N = {N}")
    print(f"{'='*50}")
    x_grid       = np.linspace(domain[0], domain[1], N + 1)
    h            = (domain[1] - domain[0]) / N
    y_ref_interp = np.interp(x_grid, x_ref, y_ref)

    # ---- FDM ----
    print("  FDM...")
    fdm_solver.set_N(N)
    w_fdm = fdm_solver.solver(tol=tol_config, max_iter=max_iter_config)
    err_fdm = np.abs(w_fdm - y_ref_interp)
    print(f"    max={np.max(err_fdm):.2e},  RMS={np.sqrt(np.mean(err_fdm**2)):.2e}")
    all_errors[N]["FDM"] = (x_grid, err_fdm)

    # ---- Shooting methods ----
    for name, Method in shooting_methods.items():
        print(f"  {name}...")
        # x grid starts at eps (singularity avoidance)
        x_shoot    = np.linspace(eps, domain[1], N + 1)
        h_shoot = (domain[1] - eps) / N      # <-- matched to shooting grid

        w_shoot, _ = SecantMethod(
            x_shoot, N, h_shoot,             # <-- h_shoot not h
            y_left, ydash1_0, ydash2_0, y_right,
            tol=tol_config, N=max_iter_config,
            Method=Method, eps=eps
        )

        # Replace any non-finite values with NaN so they don't corrupt interpolation
        w_shoot = np.where(np.isfinite(w_shoot), w_shoot, np.nan)

        # Interpolate back onto the uniform reference grid
        w_interp = np.interp(x_grid, x_shoot, w_shoot)
        err      = np.abs(w_interp - y_ref_interp)

        finite_err = err[np.isfinite(err) & (err > 0)]
        if len(finite_err) > 0:
            print(f"    max={np.max(finite_err):.2e},  RMS={np.sqrt(np.mean(finite_err**2)):.2e},  valid_pts={len(finite_err)}/{N+1}")
        else:
            print(f"    WARNING: No valid error points — method likely diverged at this N")
        print("Value of w_interp at x=0:", w_interp[0], "(should be close to y_left =", y_left, ")")
        print("Value of w_interp at x=1:", w_interp[-1], "(should be close to y_right =", y_right, ")")
        all_errors[N][name] = (x_grid, err)

# =============================================================================
# Plot: 2x2 grid, one subplot per N, all 5 methods overlaid on each
# =============================================================================

method_order = ["FDM", "RK2", "RK4", "ABM2", "ABM4"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Absolute Error vs Reference Solution — All Methods",
             fontsize=14, fontweight="bold")

for ax, N in zip(axes.flat, Ns):
    any_plotted = False
    for name in method_order:
        x_grid, err = all_errors[N][name]
        s = method_styles[name]

        # Only plot finite, positive values — semilogy silently drops the
        # entire line if any NaN/Inf/zero is present without this mask
        valid = np.isfinite(err) & (err > 0)
        if not np.any(valid):
            print(f"  PLOT WARNING: Skipping {name} at N={N} (no valid points)")
            continue

        ax.semilogy(x_grid[valid], err[valid],
                    label=name,
                    color=s["color"],
                    linewidth=s["lw"],
                    linestyle=s["ls"])
        any_plotted = True

    ax.set_title(f"N = {N}", fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("|error|")
    ax.grid(True, which="both", alpha=0.3)
    if any_plotted:
        ax.legend(fontsize=8, loc="best")

plt.tight_layout()
save_path = PLOTS_DIR / "Error_Comparison_All_Methods.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {save_path}")
plt.show()

# =============================================================================
# Summary table: RMS and Max error for every method x N
# =============================================================================

print("\n" + "=" * 85)
print(f"{'Method':<8}", end="")
for N in Ns:
    print(f"  {'N='+str(N):<24}", end="")
print()
print(f"{'':8}", end="")
for N in Ns:
    print(f"  {'RMS':<12}{'Max':<12}", end="")
print()
print("-" * 85)

for name in method_order:
    print(f"{name:<8}", end="")
    for N in Ns:
        _, err = all_errors[N][name]
        finite_err = err[np.isfinite(err)]
        if len(finite_err) > 0:
            rms_e = np.sqrt(np.mean(finite_err ** 2))
            max_e = np.max(finite_err)
            print(f"  {rms_e:<12.2e}{max_e:<12.2e}", end="")
        else:
            print(f"  {'DIVERGED':<12}{'DIVERGED':<12}", end="")
    print()

print("=" * 85)
# print(f_1(0,0,1))