import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods.ABM_4 import ABM4
from methods.RK_2 import RK2
from methods.RK_4 import RK4
from methods.ABM_2 import ABM2
from core.core import odesolver

def f(u, x):
    return np.array([u[1], -u[1]**2/(u[0] + 1e-4)])

bc = ((1, 0, 0), (1, 0, -1))
beta = 1.0

methods = [RK2, RK4, ABM2, ABM4]
methods_str = ['RK2', 'RK4', 'ABM2', 'ABM4']

def compute_R(method, s, h=0.001):
    eps = 1e-3
    u0 = np.array([eps * s, s])
    solver = odesolver(order=2, method=method, bc=bc, tol=1e-12, max_iter=1,
                       func=f, guess=[s, s], xstart=eps, xend=1, h=h)
    u, x, blew = method(solver, u0)
    if blew: return np.inf
    return u[-1, 0] - beta

def secant_solve(method, s0, s1, h=0.001, tol=1e-8, max_iter=20):
    R0 = compute_R(method, s0, h)
    R1 = compute_R(method, s1, h)
    for i in range(max_iter):
        if np.isinf(R0) or np.isinf(R1):
            return np.nan, i, False
        if abs(R1 - R0) < 1e-15:
            return np.nan, i, False
        s2 = s1 - R1 * (s1 - s0) / (R1 - R0)
        if s2 < 0.01 or s2 > 500:
            return np.nan, i, False
        R2 = compute_R(method, s2, h)
        
        if abs(R2) < tol:
            return s2, i+1, True
        s0, s1 = s1, s2
        R0, R1 = R1, R2
    
    return s1, max_iter, abs(R1) < tol * 100

def compute_condition_number_2d(method, s0, s1, h=0.001):
    delta = 1e-4
    
    s_star, _, converged = secant_solve(method, s0, s1, h)
    if not converged or np.isnan(s_star):
        return np.inf, np.inf, np.inf, np.nan
    
    s_star_s0p, _, _ = secant_solve(method, s0 * (1 + delta), s1, h)
    s_star_s0m, _, _ = secant_solve(method, s0 * (1 - delta), s1, h)
    s_star_s1p, _, _ = secant_solve(method, s0, s1 * (1 + delta), h)
    s_star_s1m, _, _ = secant_solve(method, s0, s1 * (1 - delta), h)
    
    ds_ds0 = (s_star_s0p - s_star_s0m) / (2 * delta * s0) if abs(s0) > 1e-10 else np.inf
    ds_ds1 = (s_star_s1p - s_star_s1m) / (2 * delta * s1) if abs(s1) > 1e-10 else np.inf
    
    if np.isnan(ds_ds0) or np.isnan(ds_ds1):
        return np.inf, np.inf, np.inf, s_star
    
    kappa_s0 = abs(ds_ds0) * abs(s0) / abs(s_star) if abs(s_star) > 1e-10 else np.inf
    kappa_s1 = abs(ds_ds1) * abs(s1) / abs(s_star) if abs(s_star) > 1e-10 else np.inf
    
    input_norm = np.sqrt(s0**2 + s1**2)
    gradient_norm = np.sqrt(ds_ds0**2 + ds_ds1**2)
    kappa_2d = gradient_norm * input_norm / abs(s_star) if abs(s_star) > 1e-10 else np.inf
    
    return kappa_s0, kappa_s1, kappa_2d, s_star

# IVP stability functions
def rk2_stability_function(z):
    return 1 + z + 0.5 * z**2


def rk4_stability_function(z):
    return 1 + z + 0.5 * z**2 + (z**3) / 6 + (z**4) / 24


def abm2_max_root(z):
    a = 1 + z + 0.75 * z**2
    b = 0.25 * z**2
    roots = np.roots([1, -a, b])
    return np.max(np.abs(roots))


def abm4_max_root(z):
    a = 1 + (7 * z) / 6 + (55 * z**2) / 64
    b = (-5 * z) / 24 - (59 * z**2) / 64
    c = (z / 24) + (37 * z**2) / 64
    d = -(9 * z**2) / 64
    roots = np.roots([1, -a, -b, -c, -d])
    return np.max(np.abs(roots))


def compute_ivp_stability_grid(stability_func, re_vals, im_vals):
    grid = np.zeros((len(im_vals), len(re_vals)))
    for i, im in enumerate(im_vals):
        for j, re in enumerate(re_vals):
            z = re + 1j * im
            grid[i, j] = stability_func(z)
    return grid


def plot_ivp_stability():
    re_vals = np.linspace(-5, 5, 301)
    im_vals = np.linspace(-5, 5, 301)
    re_grid, im_grid = np.meshgrid(re_vals, im_vals)

    methods_ivp = [
        ("RK2", lambda z: abs(rk2_stability_function(z))),
        ("RK4", lambda z: abs(rk4_stability_function(z))),
        ("ABM2", abm2_max_root),
        ("ABM4", abm4_max_root),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (name, func) in zip(axes, methods_ivp):
        stability_grid = compute_ivp_stability_grid(func, re_vals, im_vals)
        stable_mask = stability_grid <= 1.0
        im = ax.contourf(
            re_grid,
            im_grid,
            stable_mask.astype(float),
            levels=[-0.5, 0.5, 1.5],
            colors=["#d73027", "#1a9850"],
            alpha=0.85,
        )
        ax.contour(re_grid, im_grid, stability_grid, levels=[1], colors='black', linewidths=1.5)
        ax.axhline(0, color='white', linewidth=0.8, alpha=0.6)
        ax.axvline(0, color='white', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Re(z)', fontsize=12)
        ax.set_ylabel('Im(z)', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{name}: IVP Stability Region', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Unstable', 'Stable'])
        cbar.set_label('|R(z)| ≤ 1', fontsize=10)

    plt.suptitle('IVP Stability Regions (|R(z)| ≤ 1)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ivp_stability.png', dpi=150)
    plt.close(fig)
    print("\n" + "=" * 70)
    print("IVP stability plot saved as 'ivp_stability.png'")
    print("=" * 70)


if __name__ == "__main__":
    h = 0.001  
    s_true = 23.25 
    n_points = 50
    s0_range = np.linspace(-25, 25, n_points)
    s1_range = np.linspace(-25, 25, n_points)
    S0, S1 = np.meshgrid(s0_range, s1_range)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    kappa_grid = np.zeros_like(S0)
    for idx, (method, name) in enumerate(zip(methods, methods_str)):
        print(f"\nComputing {name}...")
        kappa_grid = np.zeros_like(S0)
        
        for i in range(n_points):
            for j in range(n_points):
                s0, s1 = s0_range[j], s1_range[i]
                _, _, k2d, _ = compute_condition_number_2d(method, s0, s1, h)
                kappa_grid[i, j] = min(k2d, 10) if not np.isinf(k2d) else 10
        
        ax = axes[idx]
        im = ax.contourf(S0, S1, kappa_grid, levels=20, cmap='RdYlGn_r', vmin=0, vmax=10)
        ax.contour(S0, S1, kappa_grid, levels=[1], colors='black', linewidths=2, linestyles='--')
        ax.plot([s_true], [s_true], 'w*', markersize=15, markeredgecolor='black', label='True s*')
        ax.axhline(y=s_true, color='white', linestyle=':', linewidth=1, alpha=0.7)
        ax.set_xlabel('s0 (first guess)', fontsize=12)
        ax.set_ylabel('s1 (second guess)', fontsize=12)
        ax.set_title(f'{name}: Condition Number κ(s0, s1)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('κ(s0, s1)', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)



    plt.suptitle('SHOOTING METHOD: 2D Condition Number Heatmaps\nAll 4 Methods Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stability_2d_heatmap.png', dpi=150)
    print("\n" + "=" * 70)
    print("Plot saved as 'stability_2d_heatmap.png'")
    print("=" * 70)
    print("\nInterpretation:")
    print("  GREEN  = Low κ (well-conditioned, stable)")
    print("  YELLOW = Medium κ (moderately sensitive)")
    print("  RED    = High κ (ill-conditioned, sensitive)")
    print("  Black dashed line = κ = 1 contour")
    print("  White star = True solution s*")
    
    # Also run IVP stability analysis
    print("\n" + "=" * 70)
    print("Running IVP Stability Analysis...")
    print("=" * 70)
    print("=" * 70)
    plot_ivp_stability()
