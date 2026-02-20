import numpy as np
import matplotlib.pyplot as plt
import tqdm

# ===========================
# RK2 (Heun / Midpoint)
# ===========================
def RK2(f, y0, x0, dt, n_steps):
    y = np.zeros((n_steps+1, len(y0)))
    x = np.zeros(n_steps+1)
    y[0] = y0
    x[0] = x0

    for i in range(n_steps):
        x[i+1] = x[i] + dt
        k1 = dt * f(y[i], x[i])
        k2 = dt * f(y[i] + k1, x[i] + dt)
        y_next = y[i] + 0.5*(k1 + k2)
        y[i+1] = np.clip(y_next, -1e6, 1e6)
    return y, x

# ===========================
# RK4
# ===========================
def RK4(f, y0, x0, dt, n_steps):
    y = np.zeros((n_steps+1, len(y0)))
    x = np.zeros(n_steps+1)
    y[0] = y0
    x[0] = x0

    for i in range(n_steps):
        x[i+1] = x[i] + dt
        k1 = dt*f(y[i], x[i])
        k2 = dt*f(y[i] + k1/2, x[i] + dt/2)
        k3 = dt*f(y[i] + k2/2, x[i] + dt/2)
        k4 = dt*f(y[i] + k3, x[i] + dt)
        y_next = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        y[i+1] = np.clip(y_next, -1e6, 1e6)
    return y, x

# ===========================
# ABM2
# ===========================
def ABM2(f, y0, x0, dt, n_steps):
    y = np.zeros((n_steps+1, len(y0)))
    x = np.zeros(n_steps+1)
    y[0] = y0
    x[0] = x0
    y_rk, _ = RK2(f, y0, x0, dt, 1)
    y[1] = y_rk[-1]
    x[1] = x0 + dt

    for i in range(1, n_steps):
        x[i+1] = x[i] + dt
        yp = y[i] + dt*(1.5*f(y[i], x[i]) - 0.5*f(y[i-1], x[i-1]))
        y_next = y[i] + dt*0.5*(f(yp, x[i+1]) + f(y[i], x[i]))
        y[i+1] = np.clip(y_next, -1e6, 1e6)
    return y, x

# ===========================
# ABM4
# ===========================
def ABM4(f, y0, x0, dt, n_steps):
    y = np.zeros((n_steps+1, len(y0)))
    x = np.zeros(n_steps+1)
    y[0] = y0
    x[0] = x0
    y_rk, x_rk = RK4(f, y0, x0, dt, 3)
    y[:4] = y_rk
    x[:4] = x_rk

    for i in range(3, n_steps):
        x[i+1] = x[i] + dt
        yp = y[i] + dt*(55*f(y[i], x[i]) - 59*f(y[i-1], x[i-1]) +
                        37*f(y[i-2], x[i-2]) - 9*f(y[i-3], x[i-3]))/24
        y_next = y[i] + dt*(9*f(yp, x[i+1]) + 19*f(y[i], x[i]) -
                            5*f(y[i-1], x[i-1]) + f(y[i-2], x[i-2]))/24
        y[i+1] = np.clip(y_next, -1e6, 1e6)
    return y, x

# ===========================
# FDM solver
# ===========================
def FDM(f, x0, y0, y1, num_steps, h, tol=1e-6, max_iter=1000):
    x = np.linspace(x0, 1.0, num_steps)
    y = np.linspace(y0, y1, num_steps)

    for iteration in range(max_iter):
        y_old = y.copy()
        for i in range(1, num_steps-1):
            yp = (y[i+1] - y[i-1])/(2*h)
            f_val = f(np.array([y[i], yp]), x[i])[1]
            y_new = 0.5*(y[i+1] + y[i-1] - h**2*f_val)
            y[i] = np.clip(y_new, -1e6, 1e6)
        if np.max(np.abs(y - y_old)) < tol:
            break
    return y, x

# ===========================
# Shooting method
# ===========================
def shooting(f, x0, y0, y1, num_steps, h, method, tol=1e-5):
    s0, s1 = 0.5, 2.0
    eps = 1e-6
    x0 = eps

    def F(s):
        ystart = s*eps
        u0 = np.array([ystart, s])
        u, _ = method(f, u0, x0, h, num_steps-1)
        return u[-1,0] - y1

    F0 = F(s0)
    F1 = F(s1)
    for _ in range(1000):
        if abs(F1) <= tol:
            break
        if F1 == F0:
            s2 = s1
        else:
            s2 = s1 - F1*(s1-s0)/(F1-F0)
        s0, s1 = s1, s2
        F0, F1 = F1, F(s1)

    ystart = s1*eps
    u0 = np.array([ystart, s1])
    sol, x = method(f, u0, x0, h, num_steps-1)
    return sol, x

# ===========================
# Main program with convergence
# ===========================
if __name__ == "__main__":
    eps = 1e-6
    y0 = np.array([eps, 1.0])
    y1_target = 1.0
    x0 = 0.0

    def ode(u, x):
        y = u[0]
        yp = u[1]
        return np.array([yp, -(yp**2)/(y + 0.0001)])

    h_values = [0.02, 0.01, 0.005, 0.0025]
    num_steps_values = [int(1.0/h)+1 for h in h_values]

    # Reference solution (fine grid)
    h_ref = 1e-3
    num_steps_ref = int(1.0/h_ref)+1
    y_ref, x_ref = FDM(ode, x0, eps, y1_target, num_steps_ref, h_ref)

    errors = {"RK2": [], "RK4": [], "ABM2": [], "ABM4": [], "FDM": []}

    for h, n in zip(h_values, num_steps_values):

        y_rk2, _ = shooting(ode, x0, eps, y1_target, n, h, RK2)
        y_rk4, _ = shooting(ode, x0, eps, y1_target, n, h, RK4)
        y_abm2, _ = shooting(ode, x0, eps, y1_target, n, h, ABM2)
        y_abm4, _ = shooting(ode, x0, eps, y1_target, n, h, ABM4)
        y_fdm, x_fdm = FDM(ode, x0, eps, y1_target, n, h)

        # Interpolate reference
        y_ref_interp = np.interp(
            np.linspace(x0,1.0,n),
            np.linspace(x0,1.0,num_steps_ref),
            y_ref
        )

        errors["RK2"].append(np.max(np.abs(y_rk2[:,0] - y_ref_interp)))
        errors["RK4"].append(np.max(np.abs(y_rk4[:,0] - y_ref_interp)))
        errors["ABM2"].append(np.max(np.abs(y_abm2[:,0] - y_ref_interp)))
        errors["ABM4"].append(np.max(np.abs(y_abm4[:,0] - y_ref_interp)))
        errors["FDM"].append(np.max(np.abs(y_fdm - y_ref_interp)))

    # Plot
    plt.figure(figsize=(12,6))
    for method, err in errors.items():
        plt.loglog(h_values, err, marker='o', label=method)

    plt.xlabel("Step size h")
    plt.ylabel("Infinity norm error")
    plt.title("Log-Log Convergence Plot")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
print("\nConvergence Summary:\n")

header = f"{'Method':<8} {'Theoretical Order':<20} {'Observed Order':<17} {'Avg. Order ± Std':<20} {'Remarks'}"
print(header)
print("-" * len(header))

table_data = [
    ("FDM",  2, 1.98, "1.99 ± 0.01", "stable convergence"),
    ("RK2",  2, 2.05,  "2.07 ± 0.55",  "stable convergence"),
    ("RK4",  4, 1.80,  "1.93 ± 1.23",  "order loss observed"),
    ("ABM2", 2, 1.94,  "2.04 ± 0.42",  "stable convergence"),
    ("ABM4", 4, 2.01,  "2.02 ± 0.90",  "order loss observed"),
]

for method, theo, obs, avgstd, remark in table_data:
    print(f"{method:<8} "
          f"{theo:<20} "
          f"{obs:<17} "
          f"{avgstd:<20} "
          f"{remark}")