import numpy as np
from typing import Callable
from Jacobian import Jacobian
from Boundary_Conditions import Boundary_Conditions, BCType
from TDMA import TDMA

class FDM_Solver:
    def __init__(self,F,Fy,Fyp,N, domain,BC,w = None):
        """_summary_

        Args:
            F (callable): y'' = F(x,y,y') in FDM
            Fy (callable): partial derivative of F wrt y
            Fyp (callable): partial derivative of F wrt y' 
            N (int): Number of grid points
            domain (Tuple): Domain
            BC (Array): Array of BC in a 2*3 matrix of the form ay + by' + c = 0
            w (Vector, optional): Initial guess vector. Defaults to None.
        """
        self.F = F
        self.Fy = Fy
        self.Fyp = Fyp
        self.N = N
        self.xstart = domain[0]
        self.xend = domain[1]
        self.h = (self.xend - self.xstart) / N
        self.BC = BC
        self.Boundary_Conditions = Boundary_Conditions(BC)
        self.left_bc_jac = self.Boundary_Conditions.build_left_bc_jac()
        self.right_bc_jac = self.Boundary_Conditions.build_right_bc_jac()
        self.left_bc_res = self.Boundary_Conditions.build_left_bc_res()
        self.right_bc_res = self.Boundary_Conditions.build_right_bc_res()

        if w is None:
            self.w = np.linspace(self.xstart,self.xend,self.N + 1)
        else:
            self.w = w
    

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

    def build_tridiagonal_terms(self):

        def fu(x, w, h, i):
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            return -1 + (h/2) * self.Fyp(xi, yi, ypi)

        def fl(x, w, h, i):
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            return -1 - (h/2) * self.Fyp(xi, yi, ypi)

        def fd(x, w, h, i):
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            return 2 + h**2 * self.Fy(xi, yi, ypi)

        return fu, fl, fd

    def interior_F(i, x, w, h):
        xi = x[i]
        yi = w[i]
        sip = (w[i+1] - w[i-1]) / (2*h)

        return (
            -w[i-1]
            + 2*w[i]
            - w[i+1]
            + h**2 * self.F(xi, yi, sip)
        )
