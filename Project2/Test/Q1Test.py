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
N = 8
domain = (xstart, xend)
h = (xend - xstart) / N
x = np.linspace(xstart, xend, N+1)


F_residual = lambda i,x,w,h : (-w[i+1] + 2*w[i] - w[i-1]) - h*(w[i+1] - w[i-1])/(2*x[i]) + h**2*np.exp(20*w[i])
F_residual_x0 = lambda w,h: -w[1] + w[0] + np.exp(20*w[0])*h**2/4
F_residual_xN = lambda w,h: w[-1]

fu = lambda w,x,h,i : -1 - h/(2*x[i])
fl = lambda w,x,h,i : -1 + h/(2*x[i])
fd = lambda w,x,h,i : 2 + h**2*20*np.exp(20*w[i])

bc_left_jac = lambda w,h : (1 + 5*h**2*np.exp(20*w[0])/4, -1)
bc_right_jac = lambda w,h : (0, 1)

jacobian = Jacobian(fu, fl, fd, bc_left_jac, bc_right_jac)
residual = Residual(F_residual,N,F_residual_x0, F_residual_xN)

F = lambda x, y, yp: np.exp(20*y) - yp/x

solver = FDM_Solver(domain=domain,jacobian=jacobian, residual=residual, N=N, BC=np.array([[0,1,0],[1,0,0]],dtype=float), F=F, Fy=None, Fyp=None)
solution = solver.solver(tol=1e-6, max_iter=1000)

plt.plot(x, solution, label='Numerical Solution', marker='x', linestyle='--', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of BVP')
plt.legend()
plt.grid(True)
plt.show()