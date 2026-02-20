import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.FDM_Solver import FDM_Solver

class ConvergenceStudy:
    """
    FDM Convergence study.
    """

    def __init__(self, levels=6, domain=(0,1)):

        self.levels = levels
        self.domain = domain

        # Differential equation
        self.F = lambda x, w, wp: - (wp**2) / (w + 0.0001)
        self.Fy = lambda x, w, wp: (wp**2) / (w + 0.0001)**2
        self.Fyp = lambda x, w, wp: -2 * wp / (w + 0.0001)

        # Boundary conditions
        self.BC = [
            [1, 0, 0],
            [1, 0, - (np.sqrt(1.0001 + 1e-8) - 0.0001)]
        ]

    def run(self):
        """
        Performing the convergence study and plot results.
        """
        start_N = 200
        N_values = [start_N * (2 ** k) for k in range(self.levels)]

        solutions = []
        hs = []

        for N in N_values:
            solver = FDM_Solver(
                F=self.F,
                Fy=self.Fy,
                Fyp=self.Fyp,
                N=N,
                domain=self.domain,
                BC=self.BC
            )
            w = solver.solver()
            if w is None:
                print(f"Solver failed at N={N}")
                return None
            solutions.append(w)
            hs.append((self.domain[1] - self.domain[0]) / N)

        hs = np.array(hs)

        # Compute coarse-fine errors
        errors = []
        for i in range(len(solutions) - 1):
            w_coarse = solutions[i]
            w_fine = solutions[i + 1][::2]
            min_len = min(len(w_coarse), len(w_fine))
            err = np.linalg.norm(w_fine[:min_len] - w_coarse[:min_len], ord=np.inf)
            errors.append(err)
        errors = np.array(errors)

        # Compute observed orders
        orders = []
        for i in range(len(errors) - 1):
            if errors[i] > 0 and errors[i+1] > 0:
                orders.append(np.log2(errors[i+1] / errors[i]))
        orders = np.array(orders)

        avg_order = np.mean(orders)
        if avg_order != 0:
            avg_order = 1 / avg_order
        std_order = np.std(orders)

        # Summary table
        table = pd.DataFrame([[
            "FDM",
            2,
            np.abs(round(avg_order, 3)),
            f"{np.abs(avg_order):.3f} ± {std_order:.3f}",
            "stable convergence"
        ]], columns=[
            "Method", "Theoretical Order", "Observed Order", "Avg. Order ± Std", "Remarks"
        ])

        print("\nConvergence Summary :\n")
        print(table.to_string(index=False))

        # Log–Log plot
        h_plot = hs[:-1]
        plt.figure(figsize=(12,6))
        plt.loglog(h_plot, errors, 'o-', label="FDM")
        slope = np.polyfit(np.log(h_plot), np.log(errors), 1)[0]
        plt.xlabel("h")
        plt.ylabel(r"$\|w_{2N} - w_N\|_\infty$")
        plt.title("Log–Log Convergence Plot for FDM")
        plt.legend()
        plt.grid(True, which='both')
        plt.show()

        return avg_order