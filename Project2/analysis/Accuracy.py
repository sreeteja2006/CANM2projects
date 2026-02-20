
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.Loader import load_config


class AccuracyAnalysis:
    def __init__(self, config_path, ref_path, Ns=None):
        """
        Parameters:
            config_path : path to config.json
            ref_path    : path to y_ref.txt
            Ns          : list of grid sizes to test (default: [10, 100, 1000, 10000])
        """
        (
            _, self.domain, self.BC,
            self.F_func, self.Fy_func, self.Fyp_func,
            _,
            self.tol_config, self.max_iter_config,
            _
        ) = load_config(config_path)

        self.Ns      = Ns if Ns is not None else [10, 100, 1000, 10000]
        self.errors  = {}
        self.solver  = None

        self.y_ref = np.loadtxt(ref_path)
        self.x_ref = np.linspace(0.0, 1.0, len(self.y_ref))

        self.PLOTS_DIR = Path(config_path).parent.parent / "outputs" / "plots"
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        ROOT = os.path.abspath(os.path.join(os.path.dirname(config_path), "..", ".."))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)

        from Project2.core.FDM_Solver import FDM_Solver
        self.solver = FDM_Solver(
            F=self.F_func, Fy=self.Fy_func, Fyp=self.Fyp_func,
            N=2, domain=self.domain, BC=self.BC
        )

    def compute_errors(self):
        self.errors = {}
        for N in self.Ns:
            self.solver.set_N(N)
            w = self.solver.solver(tol=self.tol_config, max_iter=self.max_iter_config)
            x = np.linspace(self.domain[0], self.domain[1], N + 1)
            y_ref_on_x = np.interp(x, self.x_ref, self.y_ref)
            err = np.abs(w - y_ref_on_x)
            self.errors[N] = (x, err)
        return self.errors

    def plot(self, save=True, show=True):
        if not self.errors:
            raise RuntimeError("No errors to plot. Run compute_errors() first.")

        plt.figure(figsize=(10, 6))
        for N, (x, err) in self.errors.items():
            plt.semilogy(x, err, label=f"N={N}")
        plt.xlabel("x")
        plt.ylabel("Absolute Error |$y_N$ - $y_{ref}$|")
        plt.title("Absolute Error of FDM solutions for various N values (log scale)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.PLOTS_DIR / "Errors for various N values.png"
            plt.savefig(save_path, dpi=150)
            print(f"\nPlot saved to: {save_path}")
        if show:
            plt.show()

    def print_summary(self):
        """Print RMS and Max error for each N."""
        if not self.errors:
            raise RuntimeError("No errors computed. Run compute_errors() first.")

        print(f"\n{'N':<10} {'RMS Error':<20} {'Max Error':<20}")
        print("-" * 50)
        for N, (x, err) in self.errors.items():
            rms_error = np.sqrt(np.mean(err ** 2))
            max_error = np.max(err)
            print(f"N={N:<8} {rms_error:<20.8e} {max_error:<20.8e}")

    def run(self):
        """Convenience method: compute errors, plot, and print summary."""
        self.compute_errors()
        self.plot()
        self.print_summary()