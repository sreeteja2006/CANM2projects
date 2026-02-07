import numpy as np

from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4
from methods.RK_2 import RK2
from methods.RK_4 import RK4
from core.core import odesolver
import matplotlib.pyplot as plt

class analysis:
    def __init__(self, solver: odesolver):
        self.solver = solver

    def convergence_rate(self, h_values, errors):
        log_h = np.log(h_values)
        log_errors = np.log(errors)
        coeffs = np.polyfit(log_h, log_errors, 1)
        return coeffs[0]
    
    def analytical_solution(self, x):
        return 1e-4 * (np.sqrt(100020000*x + 1) - 1)
    
    def normalized_error(self, numerical_values, analytical_values):
        return np.linalg.norm(numerical_values - analytical_values, ord=np.inf) / np.linalg.norm(analytical_values, ord=np.inf)
    
    def accuracy_analytical(self):
        original_h = self.solver.get_h()
        original_method = self.solver.get_method()   # MUST be a function
        original_tol = self.solver.get_tol()
        self.solver.set_tol(1e-12)  # Set a very tight tolerance for accuracy testing
        methods = [RK2, RK4, ABM2, ABM4]
        methods_str = ['RK2', 'RK4', 'ABM2', 'ABM4']

        error = np.zeros((len(methods), 2))

        for i, method in enumerate(methods):

            # ---- set method safely ----
            if not callable(method):
                raise TypeError(f"Method {method} is not callable")
            self.solver.set_method(method)

            # ---- solve with h ----
            self.solver.set_h(original_h)
            numerical_values, x_temp, _, _, _ = self.solver.solve()
            analytical_values = self.analytical_solution(x_temp)
            error[i, 0] = self.normalized_error(
                numerical_values[:, 0], analytical_values
            )

            # ---- solve with h/2 ----
            self.solver.set_h(original_h / 2)
            numerical_values, x_temp, _, _, _ = self.solver.solve()
            analytical_values = self.analytical_solution(x_temp)
            error[i, 1] = self.normalized_error(
                numerical_values[:, 0], analytical_values
            )

            # ---- restore h before next method ----
            self.solver.set_h(original_h)

        # ---- restore original solver state ----
        self.solver.set_method(original_method)
        self.solver.set_h(original_h)

        # ---- print convergence order ----
        for i, name in enumerate(methods_str):
            order = np.log(error[i, 0] / error[i, 1]) / np.log(2)

            print(
                f"Method: {name}, "
                f"Error(h): {error[i,0]:.2e}, "
                f"Error(h/2): {error[i,1]:.2e}, "
                f"Observed order: {order:.2f}"
            )

        return error
    def accuracy_bvp(self):
        original_h = self.solver.get_h()
        original_method = self.solver.get_method()

        methods = [RK2, RK4, ABM2, ABM4]
        methods_str = ['RK2', 'RK4', 'ABM2', 'ABM4']

        error = np.zeros(len(methods))

        ystart, yend, bc_type = self.solver.bcs()

        for i, method in enumerate(methods):
            self.solver.set_method(method)
            self.solver.set_h(original_h)

            u, x, s, blew, _ = self.solver.solve()

            if blew:
                error[i] = np.nan
                continue

            # Boundary residual only
            if bc_type in ("dd", "nd"):
                error[i] = abs(u[-1, 0] - yend)
            else:
                error[i] = abs(u[-1, 1] - yend)

        # restore solver state
        self.solver.set_method(original_method)
        self.solver.set_h(original_h)

        print("\nBVP Accuracy (boundary satisfaction):")
        for i, name in enumerate(methods_str):
            print(
                f"{name}: "
                f"Boundary residual = {error[i]:.2e}"
            )

        return error
    
    def Convergence_test(self, h_values):
        original_h = self.solver.get_h()
        original_method = self.solver.get_method()

        methods = [RK2, RK4, ABM2, ABM4]
        methods_str = ['RK2', 'RK4', 'ABM2', 'ABM4']

        errors = np.zeros((len(methods), len(h_values)))
        
        # Setup plotting grid: rows = methods, columns = 2 (1 for Solutions, 1 for Convergence)
        fig, axes = plt.subplots(len(methods), 1, figsize=(10, 4 * len(methods)), constrained_layout=True)

        for i, method in enumerate(methods):
            self.solver.set_method(method)
            ax = axes[i] # Current subplot for this method

            for j, h in enumerate(h_values):
                self.solver.set_h(h)
                u, x, _, blew, _ = self.solver.solve()

                if blew:
                    errors[i, j] = np.nan
                    print(f"Method {methods_str[i]} blew up at h={h}")
                    continue

                # --- Calculate Error ---
                analytical_values = self.analytical_solution(x)
                errors[i, j] = self.normalized_error(u[:, 0], analytical_values)

                # --- Plot Solution for this h ---
                ax.plot(x, u[:, 0], label=f'h={h}')
            
            # Formatting the Solution plot for the current method
            ax.set_title(f"Solutions for {methods_str[i]}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize='small', loc='upper right')
            ax.grid(True)

        # Restore solver state
        self.solver.set_method(original_method)
        self.solver.set_h(original_h)

        # Separate Figure for the Log-Log Convergence plot
        plt.figure(figsize=(8, 6))
        for i, name in enumerate(methods_str):
            plt.loglog(h_values, errors[i], marker='o', label=name)
        
        plt.xlabel('Step size (h)')
        plt.ylabel('Error')
        plt.title('Convergence Summary (Log-Log)')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.show()

        return errors