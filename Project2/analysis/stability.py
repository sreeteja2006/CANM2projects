import numpy as np
import matplotlib.pyplot as plt

from core.TDMA import TDMA
from core.Residual import Residual
from core.Jacobian import Jacobian
from core.Function_Generator import Function_Generator
from core.Boundary_Conditions import Boundary_Conditions
from core.FDM_Solver import FDM_Solver

class StabilitySolver:
    def __init__(self, F, Fy, Fyp, N, domain, BC, w0=None):
        self.F = F
        self.Fy = Fy
        self.Fyp = Fyp

        self.N = int(N)
        self.domain = (float(domain[0]), float(domain[1]))
        self.BC = BC

        a, b = self.domain
        self.x = np.linspace(a, b, self.N + 1)
        self.h = (b - a) / self.N

        fdmsolver = FDM_Solver(N,domain,F, Fy, Fyp, BC)
        bc = Boundary_Conditions(BC)

        if(Fy is None):
            Fy = fdmsolver._approx_Fy
        if(Fyp is None):
            Fyp = fdmsolver._approx_Fyp

        self.left_bc_jac = bc.build_left_bc_jac(Fy, Fyp, self.x[0])
        self.right_bc_jac = bc.build_right_bc_jac(Fy, Fyp, self.x[-1])
        self.left_bc_res = bc.build_left_bc_res(F, self.x[0])
        self.right_bc_res = bc.build_right_bc_res(F, self.x[-1])

        fg = Function_Generator(F, Fy, Fyp)
        self.fu, self.fl, self.fd = fg.build_tridiagonal_terms()
        self.residual_F = fg.build_residual_function()

        if w0 is None:
            self.w = np.linspace(a, b, self.N + 1, dtype=float)
        else:
            self.w = np.array(w0, dtype=float)



    @staticmethod
    def _full_tridiag(d, l, u):
        n = len(d)
        J = np.zeros((n, n), dtype=float)
        J[np.arange(n), np.arange(n)] = d
        J[np.arange(1, n), np.arange(n - 1)] = l
        J[np.arange(n - 1), np.arange(1, n)] = u
        return J

    @staticmethod
    def _svd_metrics(M):
        s = np.linalg.svd(M, compute_uv=False)
        smin = float(np.min(s))
        smax = float(np.max(s))
        cond2 = float("inf") if smin == 0 else float(smax / smin)
        return smin, cond2

    @staticmethod
    def _inv_amp_dense(J, trials=20, seed=0):
        rng = np.random.default_rng(seed)
        n = J.shape[0]
        vals = []
        for _ in range(trials):
            b = rng.standard_normal(n)
            nb = np.linalg.norm(b, 2)
            if nb == 0:
                continue
            b = b / nb
            try:
                x = np.linalg.solve(J, b)
            except np.linalg.LinAlgError:
                return float("inf"), float("inf")
            vals.append(float(np.linalg.norm(x, 2)))
        if not vals:
            return float("inf"), float("inf")
        vals = np.array(vals, dtype=float)
        return float(np.max(vals)), float(np.median(vals))

    @staticmethod
    def _inv_amp_tridiag(d, l, u, trials=20, seed=0):
        rng = np.random.default_rng(seed)
        n = len(d)
        vals = []
        for _ in range(trials):
            b = rng.standard_normal(n)
            nb = np.linalg.norm(b, 2)
            if nb == 0:
                continue
            b = b / nb
            x = TDMA(u.copy(), l.copy(), d.copy(), b)
            vals.append(float(np.linalg.norm(x, 2)))
        if not vals:
            return float("inf"), float("inf")
        vals = np.array(vals, dtype=float)
        return float(np.max(vals)), float(np.median(vals))
    
    @staticmethod
    def _is_diagonally_dominant(d, l, u):
        n = len(d)
        for i in range(n):
            left = abs(l[i-1]) if i > 0 else 0.0
            right = abs(u[i]) if i < n-1 else 0.0
            if abs(d[i]) < left + right:
                return False
        return True

    def solve_fdm(self, tol=1e-10, max_iter=50, verbose=False):
        residual = Residual(self.residual_F, self.N, self.left_bc_res, self.right_bc_res)
        jacobian = Jacobian(self.fu, self.fl, self.fd, self.left_bc_jac, self.right_bc_jac)

        w = self.w.copy()
        x = self.x

        hist = {
            "res_2": [],
            "sigma_min": [],
            "cond2": [],
            "inv_amp_max": [],
            "inv_amp_med": [],
            "dd_ok": [],
        }

        for k in range(int(max_iter)):
            u, l, d = jacobian.build(w, x)
            Fv = residual.build(w, x)

            r2 = float(np.linalg.norm(Fv, 2))
            hist["res_2"].append(r2)

            Jfull = self._full_tridiag(d, l, u)
            smin, c2 = self._svd_metrics(Jfull)
            hist["sigma_min"].append(smin)
            hist["cond2"].append(c2)

            iamx, iamed = self._inv_amp_tridiag(d, l, u, seed=k)
            hist["inv_amp_max"].append(iamx)
            hist["inv_amp_med"].append(iamed)
            dd_ok = self._is_diagonally_dominant(d, l, u)
            hist["dd_ok"].append(dd_ok)
            delta = TDMA(u.copy(), l.copy(), d.copy(), -Fv)
            step = float(np.linalg.norm(delta, np.inf))

            if verbose:
                print(
                    f"FDM iter {k}: ||F||2={r2:.3e}  "
                    f"sigma_min={smin:.3e}  cond2={c2:.3e}  inv_amp_max={iamx:.3e}"
                )

            w += delta
            if step < tol:
                return w, k + 1, hist

        return w, int(max_iter), hist


    def _rk4_step_uS(self, x, u, S, h):
        def f_u(xv, uv):
            yv, v = float(uv[0]), float(uv[1])
            return np.array([v, self.F(xv, yv, v)], dtype=float)

        def A(xv, uv):
            yv, v = float(uv[0]), float(uv[1])
            return np.array(
                [[0.0, 1.0],
                 [self.Fy(xv, yv, v), self.Fyp(xv, yv, v)]],
                dtype=float,
            )

        k1u = f_u(x, u)
        k1S = A(x, u) @ S

        u2 = u + 0.5 * h * k1u
        S2 = S + 0.5 * h * k1S
        k2u = f_u(x + 0.5 * h, u2)
        k2S = A(x + 0.5 * h, u2) @ S2

        u3 = u + 0.5 * h * k2u
        S3 = S + 0.5 * h * k2S
        k3u = f_u(x + 0.5 * h, u3)
        k3S = A(x + 0.5 * h, u3) @ S3

        u4 = u + h * k3u
        S4 = S + h * k3S
        k4u = f_u(x + h, u4)
        k4S = A(x + h, u4) @ S4

        un = u + (h / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        Sn = S + (h / 6.0) * (k1S + 2 * k2S + 2 * k3S + k4S)
        return un, Sn

    def shoot_integrate_uS(self, y0, yp0, h=5e-4, eps=1e-3):
        a, b = self.domain
        x0 = a + float(eps)
        x1 = b
        if x1 <= x0:
            return None, None, True

        n = int(np.ceil((x1 - x0) / h))
        n = max(n, 1)
        h = (x1 - x0) / n

        # eps start: approximate only y(eps) = y0 + yp0*eps
        u = np.array([float(y0 + yp0 * eps), float(yp0)], dtype=float)

        # sensitivity at eps:
        # y(eps)=y0+eps*yp0, yp(eps)=yp0
        S = np.array([[1.0, float(eps)],
                      [0.0, 1.0]], dtype=float)

        x = x0
        for _ in range(n):
            u, S = self._rk4_step_uS(x, u, S, h)
            x += h
            if not (np.isfinite(u).all() and np.isfinite(S).all()):
                return None, None, True

        return u, S, False

    def shoot_residual_and_jacobian(self, y0, yp0, h=5e-4, eps=1e-3):
        (aL, bL, cL), (aR, bR, cR) = self.BC

        r1 = float(aL * y0 + bL * yp0 + cL)

        u_end, S_end, blew = self.shoot_integrate_uS(y0, yp0, h=h, eps=eps)
        if blew or u_end is None or S_end is None:
            return None, None, True

        yb = float(u_end[0])
        ypb = float(u_end[1])

        r2 = float(aR * yb + bR * ypb + cR)

        J11 = float(aL)
        J12 = float(bL)

        S11 = float(S_end[0, 0])
        S12 = float(S_end[0, 1])
        S21 = float(S_end[1, 0])
        S22 = float(S_end[1, 1])

        J21 = float(aR * S11 + bR * S21)
        J22 = float(aR * S12 + bR * S22)

        r = np.array([r1, r2], dtype=float)
        J = np.array([[J11, J12], [J21, J22]], dtype=float)
        return r, J, False

    def solve_shooting_newton(
        self,
        y0_init,
        yp0_init,
        h=5e-4,
        eps=1e-3,
        tol=1e-10,
        max_iter=50,
        verbose=False,
    ):
        y0 = float(y0_init)
        yp0 = float(yp0_init)

        hist = {
            "res_2": [],
            "sigma_min": [],
            "cond2": [],
            "inv_amp_max": [],
            "inv_amp_med": [],
        }

        for k in range(int(max_iter)):
            r, J, blew = self.shoot_residual_and_jacobian(y0, yp0, h=h, eps=eps)
            if blew or r is None or J is None:
                return np.nan, np.nan, k, False, hist

            r2 = float(np.linalg.norm(r, 2))
            hist["res_2"].append(r2)

            smin, c2 = self._svd_metrics(J)
            hist["sigma_min"].append(smin)
            hist["cond2"].append(c2)

            iamx, iamed = self._inv_amp_dense(J, seed=k)
            hist["inv_amp_max"].append(iamx)
            hist["inv_amp_med"].append(iamed)

            if verbose:
                print(
                    f"SHOOT iter {k}: ||r||2={r2:.3e}  "
                    f"sigma_min={smin:.3e}  cond2={c2:.3e}  inv_amp_max={iamx:.3e}"
                )

            if r2 < tol:
                return y0, yp0, k + 1, True, hist

            try:
                delta = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                return np.nan, np.nan, k, False, hist

            y0 += float(delta[0])
            yp0 += float(delta[1])

            if not (np.isfinite(y0) and np.isfinite(yp0)):
                return np.nan, np.nan, k, False, hist

        return y0, yp0, int(max_iter), False, hist



    
    @staticmethod
    def plot_fdm_all(hist_fdm, out_path="outputs/plots/FDM_all_metrics.png"):
        k_res = np.arange(len(hist_fdm["res_2"]), dtype=float)
        k_cond = np.arange(len(hist_fdm["cond2"]), dtype=float)
        k_sig = np.arange(len(hist_fdm["sigma_min"]), dtype=float)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        # Residual
        if len(k_res):
            axs[0].semilogy(k_res, hist_fdm["res_2"], marker="o")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel(r"$\|F\|_2$")
        axs[0].set_title("FDM Residual vs Iteration")
        axs[0].grid(True)

        # Condition number
        if len(k_cond):
            axs[1].semilogy(k_cond, hist_fdm["cond2"], marker="o")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel(r"$\kappa_2(J)$")
        axs[1].set_title("FDM Condition Number vs Iteration")
        axs[1].grid(True)

        # Smallest singular value
        if len(k_sig):
            axs[2].semilogy(k_sig, hist_fdm["sigma_min"], marker="o")
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel(r"$\sigma_{\min}(J)$")
        axs[2].set_title("FDM Smallest Singular Value vs Iteration")
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.show()