import numpy as np
from Project2.Core.TDMA import TDMA
from Project2.Core.Residual import Residual
from Project2.Core.Jacobian import Jacobian
from Project2.Core.Function_Generator import Function_Generator
from Project2.Core.Boundary_Conditions import Boundary_Conditions


class StabilitySolver:
    def __init__(self, F, Fy, Fyp, N, domain, BC, w0=None):

        self.F = F
        self.Fy = Fy
        self.Fyp = Fyp
        self.N = N
        self.domain = domain
        self.BC = BC

        self.x = np.linspace(domain[0], domain[1], N + 1)
        self.h = (domain[1] - domain[0]) / N

        self.boundary_conditions = Boundary_Conditions(BC)

        self.left_bc_jac = self.boundary_conditions.build_left_bc_jac(Fy, Fyp, self.x[0])
        self.right_bc_jac = self.boundary_conditions.build_right_bc_jac(Fy, Fyp, self.x[-1])

        self.left_bc_res = self.boundary_conditions.build_left_bc_res(F, self.x[0])
        self.right_bc_res = self.boundary_conditions.build_right_bc_res(F, self.x[-1])

        self.function_generator = Function_Generator(F, Fy, Fyp)

        self.fu, self.fl, self.fd = self.function_generator.build_tridiagonal_terms()
        self.residual_F = self.function_generator.build_residual_function()

        if w0 is None:
            self.w = np.linspace(domain[0], domain[1], N + 1)
        else:
            self.w = np.array(w0)


    @staticmethod
    def build_full_tridiag(d, l, u):
        n = len(d)
        J = np.zeros((n, n))
        J[np.arange(n), np.arange(n)] = d
        J[np.arange(1, n), np.arange(n - 1)] = l
        J[np.arange(n - 1), np.arange(1, n)] = u
        return J

    @staticmethod
    def diag_dom_metrics(d, l, u):
        n = len(d)
        ok = True
        min_ratio = np.inf

        for i in range(n):
            diag = abs(d[i])
            off = 0.0
            if i > 0:
                off += abs(l[i - 1])
            if i < n - 1:
                off += abs(u[i])

            if diag < off:
                ok = False

            if off > 0:
                min_ratio = min(min_ratio, diag / off)

        if min_ratio == np.inf:
            min_ratio = float("inf")

        return {"dd_ok": ok, "dd_min_ratio": float(min_ratio)}


    @staticmethod
    def infnorm_tridiag(d, l, u):
        n = len(d)
        max_sum = 0.0
        for i in range(n):
            s = abs(d[i])
            if i > 0:
                s += abs(l[i - 1])
            if i < n - 1:
                s += abs(u[i])
            max_sum = max(max_sum, s)
        return float(max_sum)


    @staticmethod
    def inv_amplification_estimate(d, l, u, trials=10, seed=0):
        rng = np.random.default_rng(seed)
        n = len(d)
        amps = []

        for _ in range(trials):
            b = rng.standard_normal(n)
            b /= np.linalg.norm(b, 2)
            x = TDMA(u.copy(), l.copy(), d.copy(), b)
            amps.append(np.linalg.norm(x, 2))

        amps = np.array(amps)
        return {
            "inv_amp_max": float(np.max(amps)),
            "inv_amp_med": float(np.median(amps))
        }

    @staticmethod
    def sigma_min_and_cond2(J):
        s = np.linalg.svd(J, compute_uv=False)
        smin = float(np.min(s))
        smax = float(np.max(s))
        cond2 = float("inf") if smin == 0 else smax / smin
        return {"sigma_min": smin, "cond2": cond2}


    @staticmethod
    def newton_order_indicators(res_hist):
        r = np.array(res_hist)
        if len(r) < 3:
            return {"lin_last": np.nan, "quad_last": np.nan}
        lin = r[-1] / r[-2] if r[-2] != 0 else np.inf
        quad = r[-1] / (r[-2] ** 2) if r[-2] != 0 else np.inf
        return {"lin_last": float(lin), "quad_last": float(quad)}


    def solve(self, tol=1e-10, max_iter=50, verbose=True):

        residual = Residual(self.residual_F, self.N,
                            self.left_bc_res, self.right_bc_res)

        jacobian = Jacobian(self.fu, self.fl, self.fd,
                            self.left_bc_jac, self.right_bc_jac)

        w = self.w.copy()
        x = self.x

        res_hist = []

        for k in range(max_iter):

            u, l, d = jacobian.build(w, x)
            F = residual.build(w, x)

            res_norm = np.linalg.norm(F, 2)
            res_hist.append(res_norm)

            dd = self.diag_dom_metrics(d, l, u)
            inv_amp = self.inv_amplification_estimate(d, l, u)

            J_full = self.build_full_tridiag(d, l, u)
            svd_metrics = self.sigma_min_and_cond2(J_full)

            delta = TDMA(u.copy(), l.copy(), d.copy(), -F)

            step_norm = np.linalg.norm(delta, np.inf)

            if verbose:
                print(f"Iter {k}: ||F||={res_norm:.3e}")
                print(f"  Diag Dominant: {dd['dd_ok']}")
                print(f"  inv_amp_max: {inv_amp['inv_amp_max']:.3e}")
                print(f"  sigma_min: {svd_metrics['sigma_min']:.3e}")
                print(f"  cond2: {svd_metrics['cond2']:.3e}")

            w += delta

            if step_norm < tol:
                if verbose:
                    print(f"Converged in {k+1} iterations")
                    print(self.newton_order_indicators(res_hist))
                return w, k + 1

        print("Did not converge")
        return w, max_iter
