from Project2.Analysis.stability import StabilitySolver
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#Group 3
def F3(x, y, yp):
    return -(yp**2) / (y + 1e-4)

def Fy3(x, y, yp):
    return (yp**2) / ((y + 1e-4)**2)

def Fyp3(x, y, yp):
    return -2.0*yp / (y + 1e-4)

domain3 = (0.0, 1.0)
BC3 = ((1, 0, 0), (1, 0, -1))  # y(0)=0, y(1)=1



#Group 2
def F2(x, y, yp):
    return 2.0*y - y**3 + 25.0*np.sin(5.0*np.pi*x/2.0)

def Fy2(x, y, yp):
    return 2.0 - 3.0*y**2

def Fyp2(x, y, yp):
    return 0.0

domain2 = (0.0, 1.0)
BC2 = ((1, 0, -1), (0, 1, 0))  # y(0)=1, y'(1)=0

#Group 5
def F5(x, y, yp):
    return -np.sin(y) - 1e-3

def Fy5(x, y, yp):
    return -np.cos(y)

def Fyp5(x, y, yp):
    return 0.0

domain5 = (0.0, float(np.pi))
BC5 = ((1, 0, 0), (1, 0, 0))  # y(0)=0, y(π)=0


N = 1000

stability = StabilitySolver(F=F5, Fy=Fy5, Fyp=Fyp5, N=N, domain=domain5, BC=BC5, w0=None)
w,iterations = stability.solve(tol=1e-10, max_iter=50, verbose=True)

print("Final solution w:", w)
print(f"Converged in {iterations} iterations")
plt.plot(np.linspace(domain5[0], domain5[1], N + 1), w, label='Numerical Solution')
plt.savefig("plot.png", dpi=200, bbox_inches="tight")
