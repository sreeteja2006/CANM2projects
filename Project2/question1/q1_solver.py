import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from core.FDM_Solver import FDM_Solver
    from core.Jacobian import Jacobian
    from core.Boundary_Conditions import Boundary_Conditions
    from core.Function_Generator import Function_Generator
    from core.Residual import Residual
    from core.TDMA import TDMA
except ModuleNotFoundError:
    PROJECT2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT2_ROOT not in sys.path:
        sys.path.insert(0, PROJECT2_ROOT)
    from core.FDM_Solver import FDM_Solver
    from core.Jacobian import Jacobian
    from core.Boundary_Conditions import Boundary_Conditions
    from core.Function_Generator import Function_Generator
    from core.Residual import Residual
    from core.TDMA import TDMA


F = lambda x, y, yp: np.exp(20 * y) - yp / x
Fy = lambda x, y, yp: 20 * np.exp(20 * y)
Fyp = lambda x, y, yp: -1 / x
BC = np.array([[0, 1, 0], [1, 0, 0]])


def _full_tridiag(d, l, u):
    n = len(d)
    J = np.zeros((n, n), dtype=float)
    J[np.arange(n), np.arange(n)] = d
    J[np.arange(1, n), np.arange(n - 1)] = l
    J[np.arange(n - 1), np.arange(1, n)] = u
    return J


def _smin_cond2(J):
    s = np.linalg.svd(J, compute_uv=False)
    smin = float(np.min(s))
    smax = float(np.max(s))
    cond2 = float("inf") if smin == 0.0 else float(smax / smin)
    return smin, cond2

@staticmethod
def _is_diagonally_dominant(d, l, u):
    n = len(d)
    for i in range(n):
        left = abs(l[i-1]) if i > 0 else 0.0
        right = abs(u[i]) if i < n-1 else 0.0
        if abs(d[i]) < left + right:
            return False
    return True


def _plot_hist(hist, out_dir="outputs/plots", tag="Q1"):
    os.makedirs(out_dir, exist_ok=True)
    k = np.arange(len(hist["res_2"]), dtype=float)

    plt.figure(figsize=(10, 6))
    plt.semilogy(k, hist["res_2"], marker="o")
    plt.xlabel("Newton iteration")
    plt.ylabel(r"$\|F(w^{(k)})\|_2$")
    plt.title(f"Residual vs iteration ({tag})")
    plt.grid(True, which="both")
    plt.savefig(os.path.join(out_dir, f"res_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(k, hist["cond2"], marker="o")
    plt.xlabel("Newton iteration")
    plt.ylabel(r"$\kappa_2(J)$")
    plt.title(f"Condition number vs iteration ({tag})")
    plt.grid(True, which="both")
    plt.savefig(os.path.join(out_dir, f"cond2_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(k, hist["smin"], marker="o")
    plt.xlabel("Newton iteration")
    plt.ylabel(r"$\sigma_{\min}(J)$")
    plt.title(f"Smallest singular value vs iteration ({tag})")
    plt.grid(True, which="both")
    plt.savefig(os.path.join(out_dir, f"smin_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def question_1_solver(N, domain, F, BC, Fy=None, Fyp=None, tol=1e-10, max_iter=1000):
    xstart, xend = domain
    x = np.linspace(xstart, xend, N + 1)

    bc_handler = Boundary_Conditions(BC)
    F_residual_xN = bc_handler.build_right_bc_res(F, xend)
    bc_right_jac = bc_handler.build_right_bc_jac(Fy, Fyp, xend)

    h = (xend - xstart) / N
    bc_left_res = lambda w, h: -2 * w[1] + 2 * w[0] + np.exp(20 * w[0]) * h**2 / 2
    bc_left_jac = lambda w, h: (2 + 10 * h**2 * np.exp(20 * w[0]), -2)

    fg = Function_Generator(F, Fy=Fy, Fyp=Fyp)
    fu, fl, fd = fg.build_tridiagonal_terms()
    F_residual = fg.build_residual_function()

    jacobian = Jacobian(fu, fl, fd, bc_left_jac, bc_right_jac)
    residual = Residual(F_residual, N, bc_left_res, F_residual_xN)

    w = np.linspace(xstart, xend, N + 1, dtype=float)

    hist = {"res_2": [], "cond2": [], "sigma_min": [], "dd_ok": []}

    for k in range(int(max_iter)):
        u, l, d = jacobian.build(w, x)
        Fv = residual.build(w, x)

        r2 = float(np.linalg.norm(Fv, 2))
        hist["res_2"].append(r2)

        Jfull = _full_tridiag(d, l, u)
        smin, cond2 = _smin_cond2(Jfull)
        hist["sigma_min"].append(smin)
        hist["cond2"].append(cond2)

        is_diag_dom = _is_diagonally_dominant(d, l, u)
        hist["dd_ok"].append(is_diag_dom)


        delta = TDMA(u.copy(), l.copy(), d.copy(), -Fv)
        w = w + delta

        if float(np.linalg.norm(delta, np.inf)) < tol:
            return w, k + 1, hist

    return w, int(max_iter), hist


def main():
    os.makedirs("outputs/plots", exist_ok=True)

    xstart = 0.0
    xend = 1.0
    N = 32
    domain = (xstart, xend)

    solution, iters, hist = question_1_solver(N, domain, F, BC, Fy=Fy, Fyp=Fyp, tol=1e-10, max_iter=1000)
    x = np.linspace(xstart, xend, N + 1)
    print(f"Converged in {iters} iterations.")
    plt.figure(figsize=(10, 6))
    plt.plot(x, solution, label="Numerical Solution", marker="x", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical Solution of BVP")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/plots/solution_Q1.png", dpi=200, bbox_inches="tight")
    plt.close()

    _plot_hist(hist, out_dir="outputs/plots", tag="Q1")


if __name__ == "__main__":
    main()
