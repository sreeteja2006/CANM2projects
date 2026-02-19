import numpy as np
import matplotlib.pyplot as plt
from core.FDM_Solver import FDM_Solver
from core.Jacobian import Jacobian
from core.Boundary_Conditions import Boundary_Conditions
from core.Function_Generator import Function_Generator
from core.Residual import Residual
from core.TDMA import TDMA

xstart = 0
xend = 1
N = 32
domain = (xstart, xend)


F = lambda x, y, yp: np.exp(20*y) - yp/x
Fy = lambda x, y, yp: 20*np.exp(20*y)
Fyp = lambda x, y, yp: -1/x
BC = np.array([[0, 1, 0], [1, 0, 0]])

def question_1_solver(N, domain, F, BC,Fy = None, Fyp = None):

    xstart, xend = domain
    h = (xend - xstart) / N
    x = np.linspace(xstart, xend, N+1)

    BoundaryConditionHandler = Boundary_Conditions(BC)
    F_residual_xN = BoundaryConditionHandler.build_right_bc_res(F, xend)
    bc_right_jac = BoundaryConditionHandler.build_right_bc_jac(Fy, Fyp, xend)

    bc_left_res = lambda w,h: -2*w[1] + 2*w[0] + np.exp(20*w[0])*h**2/2
    bc_left_jac = lambda w,h : (2 + 10*h**2*np.exp(20*w[0]), -2)
    fu,fl,fd = Function_Generator(F, Fy=Fy, Fyp=Fyp).build_tridiagonal_terms()
    F_residual = Function_Generator(F, Fy=Fy, Fyp=Fyp).build_residual_function()

    jacobian = Jacobian(fu, fl, fd, bc_left_jac, bc_right_jac)
    residual = Residual(F_residual,N,bc_left_res, F_residual_xN)


    solver = FDM_Solver(domain=domain,jacobian=jacobian, residual=residual, N=N)

    solution = solver.solver(tol=1e-10, max_iter=1000)
    return solution

solution = question_1_solver(N, domain, F, BC, Fy=Fy, Fyp=Fyp)
x = np.linspace(xstart, xend, N+1)


plt.plot(x, solution, label='Numerical Solution', marker='x', linestyle='--', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of BVP')
plt.legend()
plt.grid(True)
plt.show()