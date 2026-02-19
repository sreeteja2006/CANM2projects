from analysis.stability import StabilitySolver
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)

def F2(x, y, yp):
    return 2.0 * y - y**3 + 25.0 * np.sin(5.0 * np.pi * x / 2.0)

def Fy2(x, y, yp):
    return 2.0 - 3.0 * y**2

def Fyp2(x, y, yp):
    return 0.0

domain2 = (0.0, 1.0)
BC2 = ((1, 0, -1), (0, 1, 0))


def F3(x, y, yp):
    return -(yp**2) / (y + 1e-4)

def Fy3(x, y, yp):
    return (yp**2) / ((y + 1e-4)**2)

def Fyp3(x, y, yp):
    return -2.0 * yp / (y + 1e-4)

domain3 = (0.0, 1.0)
BC3 = ((1, 0, 0), (1, 0, -1))


def F4(x, y, yp):
    return -100.0 * (1.0 - 0.2 * x) * (y**2)

def Fy4(x, y, yp):
    return -200.0 * (1.0 - 0.2 * x) * y

def Fyp4(x, y, yp):
    return 0.0

domain4 = (0.0, float(np.pi) / 2.0)
BC4 = ((1, 0, 0), (1, 0, -1))


def F5(x, y, yp):
    return -np.sin(y) - 1e-3

def Fy5(x, y, yp):
    return -np.cos(y)

def Fyp5(x, y, yp):
    return 0.0

domain5 = (0.0, float(np.pi))
BC5 = ((1, 0, 0), (1, 0, 0))


def F6(x, y, yp):
    return -2.0 * y * yp + np.exp(10.0 * y)

def Fy6(x, y, yp):
    return -2.0 * yp + 10.0 * np.exp(10.0 * y)

def Fyp6(x, y, yp):
    return -2.0 * y

domain6 = (0.0, 1.0)
BC6 = ((1, 0, 2.5), (1, 0, -3.0))


# def safe_savefig(path):
#     plt.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close()

# def pad_to_same_length(arr_list, pad_value=np.nan):
#     L = max(len(a) for a in arr_list)
#     out = []
#     for a in arr_list:
#         a = np.asarray(a, dtype=float)
#         if len(a) < L:
#             a = np.concatenate([a, np.full(L - len(a), pad_value, dtype=float)])
#         out.append(a)
#     return out

# def run_case(name, F, Fy, Fyp, domain, BC, N, tol, max_iter, verbose):
#     solver = StabilitySolver(F=F, Fy=Fy, Fyp=Fyp, N=N, domain=domain, BC=BC, w0=None)
#     w, iters, hist = solver.solve(tol=tol, max_iter=max_iter, verbose=verbose)

#     x = np.linspace(domain[0], domain[1], N + 1)


#     return w, iters, hist


N = int(1e3)
tol = 1e-10
max_iter = 50
verbose = True

# cases = [
#     ("Group-2", F2, Fy2, Fyp2, domain2, BC2),
#     ("Group-3", F3, Fy3, Fyp3, domain3, BC3),
#     ("Group-4", F4, Fy4, Fyp4, domain4, BC4),
#     ("Group-5", F5, Fy5, Fyp5, domain5, BC5),
#     ("Group-6", F6, Fy6, Fyp6, domain6, BC6),
# ]

# all_hist = {}

# for name, F, Fy, Fyp, domain, BC in cases:
#     w, iters, hist = run_case(name, F, Fy, Fyp, domain, BC, N, tol, max_iter, verbose)
#     all_hist[name] = (w, iters, hist)

# names = list(all_hist.keys())

# res_lists = pad_to_same_length([all_hist[n][2]["res_inf"] for n in names])
# cond_lists = pad_to_same_length([all_hist[n][2]["cond2"] for n in names])
# sig_lists = pad_to_same_length([all_hist[n][2]["sigma_min"] for n in names])

# plt.figure()
# for n, r in zip(names, res_lists):
#     plt.semilogy(r, label=n)
# plt.xlabel("Newton iteration k")
# plt.ylabel(r"$\|F(w^{(k)})\|_\infty$")
# plt.title("Residual vs iteration (Groups 2--6)")
# plt.legend()
# safe_savefig("Plots/residual_vs_iteration_G2toG6.png")

# plt.figure()
# for n, c in zip(names, cond_lists):
#     plt.semilogy(c, label=n)
# plt.xlabel("Newton iteration k")
# plt.ylabel(r"$\kappa_2(J)$")
# plt.title("Condition number vs iteration (Groups 2--6)")
# plt.legend()
# safe_savefig("Plots/cond_vs_iteration_G2toG6.png")

# plt.figure()
# for n, s in zip(names, sig_lists):
#     plt.semilogy(s, label=n)
# plt.xlabel("Newton iteration k")
# plt.ylabel(r"$\sigma_{\min}(J)$")
# plt.title("Smallest singular value vs iteration (Groups 2--6)")
# plt.legend()
# safe_savefig("Plots/sigma_min_vs_iteration_G2toG6.png")

# for name in names:
#     w, iters, hist = all_hist[name]
#     final_res = float(hist["res_inf"][-1])
#     final_sig = float(hist["sigma_min"][-1])
#     final_cond = float(hist["cond2"][-1])
#     dd_final = bool(hist["dd_ok"][-1]) if "dd_ok" in hist and len(hist["dd_ok"]) else None
#     print(f"{name}: iterations={iters}, ||F||inf={final_res:.3e}, sigma_min={final_sig:.3e}, cond2={final_cond:.3e}, diag_dom={dd_final}")



solver = StabilitySolver(F3, Fy3, Fyp3, N=1000, domain=domain4, BC=BC4)

out, hist_fdm, hist_sh = solver.compare_fdm_vs_shooting(
    shoot_y0_init=0.0,     
    shoot_yp0_init=1,    
    shoot_h=1e-5,
    shoot_eps=1e-3, 
    verbose=True
)

print(out)

solver.plot_both_cond2(hist_fdm, hist_sh, out_path="outputs/plots/cond2_FDM_vs_SHOOT.png")
solver.plot_both_sigma_min(hist_fdm, hist_sh, out_path="outputs/plots/sigmaMin_FDM_vs_SHOOT.png")
solver.plot_fdm_all(hist_fdm, out_path="outputs/plots/FDM_all.png")