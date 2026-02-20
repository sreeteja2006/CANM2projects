from core.core import odesolver
from methods.RK_4 import RK4
from methods.RK_2 import RK2
from methods.ABM_4 import ABM4
from methods.ABM_2 import ABM2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('Agg')  # Non-interactive backend for saving plots


class analysis:

    def __init__(self, solver: odesolver):
        self.solver = solver
        self.methods = [RK2, RK4, ABM2, ABM4]
        self.methods_str = ['RK2', 'RK4', 'ABM2', 'ABM4']

    def restrict_solution(self, y_fine, ratio):
        """
        Restrict fine-grid solution to coarse grid by simple injection.
        ratio = h_coarse / h_fine (e.g., 2)
        """
        return y_fine[::ratio]

    def h_refinement(self, htest):
        original_h = self.solver.get_h()
        original_method = self.solver.get_method()

        h_values = [htest / (2**i) for i in range(10)]  # h, h/2, h/4

        methods = [RK2, RK4, ABM2, ABM4]
        method_names = ['RK2', 'RK4', 'ABM2', 'ABM4']


        results = {}
        for method, name in zip(methods, method_names):
            self.solver.set_method(method)

            solutions = []

            for h in h_values:
                self.solver.set_h(h)
                y, x, _, blew, _ = self.solver.solve()
                solutions.append(None if blew or y is None else y)

            if any(sol is None for sol in solutions):
                results[name] = {"order_avg": np.nan}
                continue

            errors = []
            for i in range(len(h_values) - 1):
                y_coarse = solutions[i]
                y_fine = solutions[i + 1][::2]

                min_len = min(len(y_coarse), len(y_fine))
                err = np.linalg.norm(
                    y_fine[:min_len] - y_coarse[:min_len],
                    ord=np.inf
                )
                errors.append(err)

            orders = []
            for i in range(len(errors) - 1):
                if errors[i] > 0 and errors[i + 1] > 0:
                    orders.append(np.log2(errors[i] / errors[i + 1]))

            orders = np.array(orders)
            valid = (orders > 0.5) & (orders < 6)

            results[name] = {
                "h": h_values,
                "errors": errors,
                "orders": orders,
                "order_avg": np.mean(orders[valid]),
                "order_std": np.std(orders[valid])
            }

        self.solver.set_method(original_method)
        self.solver.set_h(original_h)

        return results

    def plot_loglog_convergence(self, results):
        plt.figure(figsize=(8, 6))
        plotted = False

        for name, data in results.items():
            h = np.asarray(data["h"])
            err = np.asarray(data["errors"])

            print(f"\n{name}")
            print("  h     :", h)
            print("  errors:", err)

            if len(err) < 1:
                print("  skipped: no errors")
                continue

            h_plot = h[:-1]

            if len(h_plot) != len(err):
                print("  skipped: length mismatch")
                continue

            mask = (err > 0) & np.isfinite(err)

            h_plot = h_plot[mask]
            err = err[mask]

            if len(h_plot) < 2:
                print("  skipped: not enough points for log–log")
                continue

            plt.loglog(h_plot, err, 'o-', label=name)
            plotted = True

            slope = np.polyfit(np.log(h_plot), np.log(err), 1)[0]
            print(f"  observed slope ≈ {slope:.3f}")

        if not plotted:
            print("\n⚠️ Nothing was plotted — check errors above.")

        plt.xlabel("Step size h")
        plt.ylabel(r"$\|y_{h/2} - y_h\|$")
        plt.title("Log–Log Convergence Plot")
        plt.grid(True, which="both")
        plt.legend()
        plt.show()


    def compute_R(self, method, s, h=0.001):
        eps = 1e-3
        u0 = np.array([eps * s, s])
        bc = self.solver.get_bc()
        func = self.solver.get_func()
        beta = -bc[1][2] / bc[1][0] if abs(bc[1][0]) > 1e-10 else 1.0
        
        solver = odesolver(order=2, method=method, bc=bc, tol=1e-12, max_iter=1,
                           func=func, guess=[s, s], xstart=eps, xend=self.solver.get_xend(), h=h)
        u, x, blew = method(solver, u0)
        if blew:
            return np.inf
        return u[-1, 0] - beta

    def secant_solve(self, method, s0, s1, h=0.001, tol=1e-8, max_iter=20):
        R0 = self.compute_R(method, s0, h)
        R1 = self.compute_R(method, s1, h)
        for i in range(max_iter):
            if np.isinf(R0) or np.isinf(R1):
                return np.nan, i, False
            if abs(R1 - R0) < 1e-15:
                return np.nan, i, False
            s2 = s1 - R1 * (s1 - s0) / (R1 - R0)
            if s2 < 0.01 or s2 > 500:
                return np.nan, i, False
            R2 = self.compute_R(method, s2, h)
            
            if abs(R2) < tol:
                return s2, i+1, True
            s0, s1 = s1, s2
            R0, R1 = R1, R2
        
        return s1, max_iter, abs(R1) < tol * 100

    def compute_condition_number_2d(self, method, s0, s1, h=0.001):
        """Compute 2D condition number for shooting method."""
        delta = 1e-4
        
        s_star, _, converged = self.secant_solve(method, s0, s1, h)
        if not converged or np.isnan(s_star):
            return np.inf, np.inf, np.inf, np.nan
        
        s_star_s0p, _, _ = self.secant_solve(method, s0 * (1 + delta), s1, h)
        s_star_s0m, _, _ = self.secant_solve(method, s0 * (1 - delta), s1, h)
        s_star_s1p, _, _ = self.secant_solve(method, s0, s1 * (1 + delta), h)
        s_star_s1m, _, _ = self.secant_solve(method, s0, s1 * (1 - delta), h)
        
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

    @staticmethod
    def rk2_stability_function(z):
        return 1 + z + 0.5 * z**2

    @staticmethod
    def rk4_stability_function(z):
        return 1 + z + 0.5 * z**2 + (z**3) / 6 + (z**4) / 24

    @staticmethod
    def abm2_max_root(z):
        a = 1 + z + 0.75 * z**2
        b = 0.25 * z**2
        roots = np.roots([1, -a, b])
        return np.max(np.abs(roots))

    @staticmethod
    def abm4_max_root(z):
        a = 1 + (7 * z) / 6 + (55 * z**2) / 64
        b = (-5 * z) / 24 - (59 * z**2) / 64
        c = (z / 24) + (37 * z**2) / 64
        d = -(9 * z**2) / 64
        roots = np.roots([1, -a, -b, -c, -d])
        return np.max(np.abs(roots))

    def compute_ivp_stability_grid(self, stability_func, re_vals, im_vals):
        grid = np.zeros((len(im_vals), len(re_vals)))
        for i, im in enumerate(im_vals):
            for j, re in enumerate(re_vals):
                z = re + 1j * im
                grid[i, j] = stability_func(z)
        return grid

    def plot_ivp_stability(self, save_path='ivp_stability.png'):
        re_vals = np.linspace(-5, 5, 301)
        im_vals = np.linspace(-5, 5, 301)
        re_grid, im_grid = np.meshgrid(re_vals, im_vals)

        methods_ivp = [
            ("RK2", lambda z: abs(self.rk2_stability_function(z))),
            ("RK4", lambda z: abs(self.rk4_stability_function(z))),
            ("ABM2", self.abm2_max_root),
            ("ABM4", self.abm4_max_root),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for ax, (name, func) in zip(axes, methods_ivp):
            stability_grid = self.compute_ivp_stability_grid(func, re_vals, im_vals)
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
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print("\n" + "=" * 70)
        print(f"IVP stability plot saved as '{save_path}'")
        print("=" * 70)

    def plot_shooting_stability_heatmap(self, s_true=23.25, n_points=50, h=0.001, 
                                         s_range=(-25, 25), save_path='stability_2d_heatmap.png'):
        s0_range = np.linspace(s_range[0], s_range[1], n_points)
        s1_range = np.linspace(s_range[0], s_range[1], n_points)
        S0, S1 = np.meshgrid(s0_range, s1_range)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (method, name) in enumerate(zip(self.methods, self.methods_str)):
            print(f"\nComputing {name}...")
            kappa_grid = np.zeros_like(S0)
            
            for i in range(n_points):
                for j in range(n_points):
                    s0, s1 = s0_range[j], s1_range[i]
                    _, _, k2d, _ = self.compute_condition_number_2d(method, s0, s1, h)
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
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print("\n" + "=" * 70)
        print(f"Plot saved as '{save_path}'")
        print("=" * 70)
        print("\nInterpretation:")
        print("  GREEN  = Low κ (well-conditioned, stable)")
        print("  YELLOW = Medium κ (moderately sensitive)")
        print("  RED    = High κ (ill-conditioned, sensitive)")
        print("  Black dashed line = κ = 1 contour")
        print("  White star = True solution s*")

    def run_full_stability_analysis(self, s_true=23.25, n_points=50, h=0.001):
        """Run complete stability analysis (both IVP and shooting method)."""
        print("\n" + "=" * 70)
        print("Running Full Stability Analysis...")
        print("=" * 70)
        
        # IVP stability
        print("\n[1] IVP Stability Analysis...")
        self.plot_ivp_stability()
        
        # Shooting method stability
        print("\n[2] Shooting Method Stability Analysis...")
        self.plot_shooting_stability_heatmap(s_true=s_true, n_points=n_points, h=h)
        
        print("\n" + "=" * 70)
        print("Stability Analysis Complete!")
        print("=" * 70)

    def load_reference_solution(self, x=None):
        """
        Load reference solution from y_ref.txt.
        
        If x is provided (array of points), interpolate the reference solution to those points.
        Otherwise, return the raw reference data.
        
        Returns:
            tuple: (y_ref, x_ref) where y_ref is the solution and x_ref is the grid it was sampled on.
        """
        y_ref_path =  Path(__file__).resolve().parent / "y_ref.txt"
        
        if not y_ref_path.exists():
            raise FileNotFoundError(f"Reference solution file not found: {y_ref_path}")
        
        # Load reference data
        y_ref_data = np.loadtxt(y_ref_path)
        
        # Create uniform grid that y_ref was sampled on (every 100 points from reference.py)
        n_ref = len(y_ref_data)
        x_ref = np.linspace(0, 1, n_ref)
        
        # If no x provided, return raw reference solution
        if x is None:
            return y_ref_data, x_ref
        
        # Otherwise, interpolate to provided x points
        y_interp = np.interp(x, x_ref, y_ref_data)
        return y_interp, x_ref
    def plot_accuracy_errors(self, h_values=None):
        """
        Plot absolute errors for each numerical method compared to reference solution.
        
        Parameters:
            h_values: list of step sizes to test (default: [1e-3, 1e-4, 1e-5, 1e-6])
        """
        if h_values is None:
            h_values = [1e-3, 1e-4, 1e-5, 1e-6]
        
        # Load reference solution
        y_ref, x_ref = self.load_reference_solution()
        
        methods = [RK2, RK4, ABM2, ABM4]
        method_names = ['RK2', 'RK4', 'ABM2', 'ABM4']
        
        # Store original state
        original_h = self.solver.get_h()
        original_method = self.solver.get_method()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()
        
        for idx, (method, name) in enumerate(zip(methods, method_names)):
            ax = axes[idx]
            self.solver.set_method(method)
            
            for h in h_values:
                self.solver.set_h(h)
                sol, x, _, blew, _ = self.solver.solve()
                
                if blew or sol is None:
                    print(f"{name} blew up at h={h}; skipping")
                    continue
                
                # Interpolate reference solution to solver's x grid
                y_ref_on_x, _ = self.load_reference_solution(x)
                
                # Compute absolute error
                y_numerical = sol[:, 0]
                error = np.abs(y_numerical - y_ref_on_x)
                
                ax.plot(x, error, label=f'h={h}')
            
            ax.set_title(f'Absolute Error |y_numerical - y_reference| ({name})')
            ax.set_xlabel('x')
            ax.set_ylabel('Absolute Error')
            ax.legend(fontsize='small')
            ax.grid(True)
        
        # Restore original state
        self.solver.set_method(original_method)
        self.solver.set_h(original_h)
        
        # Save and show
        outpath = Path(__file__).resolve().parent / 'accuracy_errors.png'
        plt.savefig(outpath, dpi=300)
        print(f"Accuracy plot saved to {outpath}")
        plt.show()