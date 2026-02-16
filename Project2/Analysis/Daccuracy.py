import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Ensure workspace root is on sys.path so we can import Project2 package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from Project2.Core.FDM_Solver import FDM_Solver

PLOTS_DIR = Path(__file__).parent.parent / "Plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def F(x, y, yp):
	# y'' = F(x,y,y') used in reference implementation
	return -(yp ** 2) / (y + 1e-4)



path = os.path.join(os.path.dirname(__file__), "y_ref.txt")
y_ref = np.loadtxt(path)
x_ref = np.linspace(0.0, 1.0, len(y_ref))


# Load reference solution
# y_ref, x_ref = load_reference()

	# Boundary conditions: y(0)=0, y(1)=1 encoded as a*y + b*y' + c = 0
BC = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, -1.0]])

Ns = [10, 100, 1000, 10000]
errors = {}
domain = (0.0, 1.0)
solver = FDM_Solver(F, None, None, 2, domain, BC)

for N in Ns:
	# Initialize solver with numerical partial derivatives (pass None)
	solver.set_N(N)
	w = solver.solver(tol=1e-8, max_iter=200)
	x = np.linspace(domain[0], domain[1], N + 1)
	y_ref_on_x = np.interp(x, x_ref, y_ref)
	err = np.abs(w - y_ref_on_x)
	errors[N] = (x, err)

	# Plot all error curves on one figure
plt.figure(figsize=(10, 6))
for N, (x, err) in errors.items():
	plt.plot(x, err, label=f"N={N}")
plt.xlabel("x")
plt.ylabel("Absolute error |$y_N$ - $y_ref$|")
plt.title("Absolute error of FDM solutions for various N values")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = PLOTS_DIR / 'Errors for various N values.png'
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to: {save_path}")
plt.show()

# Print RMS and Max error for each N
for N, (x, err) in errors.items():
	rms_error = np.sqrt(np.mean(err ** 2))
	max_error = np.max(err)
	print(f"N={N}: RMS error = {rms_error:.8e}, Max error = {max_error:.8e}")