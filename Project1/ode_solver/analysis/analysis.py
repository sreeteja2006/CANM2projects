import numpy as np

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



