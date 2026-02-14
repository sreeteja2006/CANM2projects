from Project2.Analysis.stability import StabilitySolver
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


F = lambda x,y,yp:-(yp ** 2) / ( (y + 1e-4))
Fy = lambda x,y,yp: (yp ** 2) / ((y + 1e-4) ** 2)
Fyp = lambda x,y,yp: -2*yp / (y + 1e-4)
N = 1000
domain = (1e-3, 1.0)

BC = np.array([
    [1, 0, -1e-3],   
    [1, 0, -1.0]
])



stability = StabilitySolver(F=F, Fy=Fy, Fyp=Fyp, N=N, domain=domain, BC=BC, w0=None)

w, iterations, cond = stability.solve(tol=1e-10, max_iter=50, verbose=True)

print("Final solution w:", w)
print(f"Converged in {iterations} iterations. Condition number: {cond:.3e}")
plt.plot(np.linspace(domain[0], domain[1], N + 1), w, label='Numerical Solution')
plt.savefig("plot.png", dpi=200, bbox_inches="tight")
