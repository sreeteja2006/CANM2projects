import numpy as np
from typing import Callable

from .Jacobian import Jacobian
from .Boundary_Conditions import Boundary_Conditions
from .TDMA import TDMA
from .Function_Generator import Function_Generator
from .Residual import Residual

#minor change
class FDM_Solver:
    def __init__(self,F,Fy,Fyp,N, domain,BC,w0 = None):
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
        self.Function_Generator = Function_Generator(F, Fy, Fyp)
        self.fu, self.fl, self.fd = self.Function_Generator.build_tridiagonal_terms()
        self.Residual_F = self.Function_Generator.build_residual_function()

        if w0 is None:
            self.w0 = np.linspace(self.xstart,self.xend,self.N + 1)
        else:
            self.w0 = w0
    

    def Norm_inf(self, vector) -> float:
        return float(max(abs(x) for x in vector))

    def solver(self,tol=1e-6, max_iter=100):
        x = np.linspace(self.xstart, self.xend, self.N + 1)
        w = self.w0.copy()

        residual = Residual(
            self.Residual_F,
            self.N,
            self.left_bc_res,
            self.right_bc_res
        )

        jacobian = Jacobian(
            fu=self.fu,
            fl=self.fl,
            fd=self.fd,
            bc_left_jac=self.left_bc_jac,
            bc_right_jac=self.right_bc_jac
        )

        for k in range(max_iter):

            u, l, d = jacobian.build(w, x)
            b = residual.build(w, x)

            delta = TDMA(u, l, d, -b)
            w += delta

            if self.Norm_inf(delta) < tol:
                print(f"Converged in {k+1} iterations")
                break

        return w

    
