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

    def value_at_x(self, x, y, x_star, component=0):
        idx = np.argmin(np.abs(x - x_star))
        return y[idx][component]

    
    def h_refinement_test(self, x_star):
        """
        Compute observed order using Q(h) = y_h(x_star)
        """
        original_h = self.solver.get_h()
        original_method = self.solver.get_method()

        h_values = [original_h / (2**i) for i in range(3)]  # h, h/2, h/4

        methods = [RK2, RK4, ABM2, ABM4]
        method_names = ['RK2', 'RK4', 'ABM2', 'ABM4']

        results = {}

        for method, name in zip(methods, method_names):
            self.solver.set_method(method)

            Q = []

            for h in h_values:
                self.solver.set_h(h)
                solution, x, _, blew, _ = self.solver.solve()

                if blew or solution is None:
                    Q.append(np.nan)
                else:
                    Q.append(self.value_at_x(x, solution, x_star))

            Q = np.array(Q)

            if np.any(np.isnan(Q)):
                k = np.nan
            else:
                num = Q[1] - Q[2]     # Q_{h/2} - Q_{h/4}
                den = Q[0] - Q[1]     # Q_h - Q_{h/2}

                if num == 0 or den == 0:
                    k = np.nan
                else:
                    k = np.log2(abs(num / den))

            results[name] = {
                "h_values": h_values,
                "Q_values": Q,
                "order": k
            }

        # Restore solver state
        self.solver.set_method(original_method)
        self.solver.set_h(original_h)

        return results

    def plot_loglog_convergence(self,results):
        plt.figure(figsize=(8, 6))

        for name, data in results.items():
            h = np.array(data["h_values"])
            Q = np.array(data["Q_values"])

            if np.any(np.isnan(Q)):
                continue

            Q_ref = Q[-1]                  # finest grid
            error = np.abs(Q[:-1] - Q_ref)
            h_plot = h[:-1]

            plt.loglog(h_plot, error, 'o-', label=name)

        plt.xlabel("h")
        plt.ylabel(r"$|y_h(x^*) - y_{ref}(x^*)|$")
        plt.title("Log–Log Convergence at Fixed x*")
        plt.grid(True, which="both")
        plt.legend()
        plt.show()