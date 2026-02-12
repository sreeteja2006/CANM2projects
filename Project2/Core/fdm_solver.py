import numpy as np
from callable import Callable

class FDM_Solver:

    def Norm_inf(vector :np.ndarray) -> float:
        return vector.max()

    def solver(self,Generate_Jacobian :callable, Generate_Residual :callable,TDMA : callable ,initial_guess :np.ndarray,x : np.ndarray ,tol :float = 1e-6, max_iter :int = 1000) -> np.ndarray:
        h = x[1] - x[0]
        w = initial_guess.copy()
        for k in range(max_iter):
            u, l, d = Generate_Jacobian(self)
            b = Generate_Residual(self)

            delta = TDMA(u, l, d, -b)
            w = w + delta

            if self.Norm_inf(delta) < tol:
                print(f"Converged in {k+1} iterations")
                break

        return w