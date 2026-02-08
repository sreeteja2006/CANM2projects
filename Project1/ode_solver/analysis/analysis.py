import numpy as np
from pathlib import Path

from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4
from methods.RK_2 import RK2
from methods.RK_4 import RK4
from core.core import odesolver
import matplotlib.pyplot as plt


class analysis:
    
    def __init__(self, solver : odesolver):
        self.solver = solver

    def restrict_solution(self, y_fine, ratio):
        """
        Restrict fine-grid solution to coarse grid by simple injection.
        ratio = h_coarse / h_fine (e.g., 2)
        """
        return y_fine[::ratio]

    def load_reference_solution(self, x=None):
        """
        Load reference solution from y_ref.txt.
        
        If x is provided (array of points), interpolate the reference solution to those points.
        Otherwise, return the raw reference data.
        
        Returns:
            tuple: (y_ref, x_ref) where y_ref is the solution and x_ref is the grid it was sampled on.
        """
        y_ref_path = Path(__file__).resolve().parent / "y_ref.txt"
        
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


    def h_refinement(self,htest):
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
