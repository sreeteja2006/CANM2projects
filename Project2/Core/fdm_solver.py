import numpy as np
from typing import Callable
from Jacobian import Jacobian

class FDM_Solver:
    def __init__(self,F,Fy,Fyp,N, x,w = None):
        pass

    def Norm_inf(vector :np.ndarray) -> float:
        return vector.max()

    def solver(self, jacobian, residual, TDMA, initial_guess, x,
           tol=1e-6, max_iter=100):

        w = initial_guess.copy()

        for k in range(max_iter):

            u, l, d = jacobian.build(w, x)
            b = residual.build(w, x)

            delta = TDMA(u, l, d, -b)
            w += delta

            if self.Norm_inf(delta) < tol:
                print(f"Converged in {k+1} iterations")
                break

        return w

    def Y_FDM(i,x, w, h):
        return w[i]
    def Yp_FDM(i,x, w, h):
        return (w[i+1] - w[i-1]) / (2*h)
    def Ypp_FDM(i,x, w, h):
        return (w[i+1] - 2*w[i] + w[i-1]) / (h**2)

