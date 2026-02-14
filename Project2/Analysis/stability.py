import numpy as np
from Project2.Core.TDMA import TDMA
from Project2.Core.Residual import Residual
from Project2.Core.Jacobian import Jacobian
from Project2.Core.Function_Generator import Function_Generator
from Project2.Core.Boundary_Conditions import Boundary_Conditions


class StabilitySolver:
    def check_diagonal_dominance(self, d, l, u):
        n = len(d)
        for i in range(n):
            diag = abs(d[i])
            off_diag = 0.0
            if i > 0:
                off_diag += abs(l[i - 1])
            if i < n - 1:
                off_diag += abs(u[i])
            if diag < off_diag:
                return False
        return True

    def spectral_radius(self, J):
        eigvals = np.linalg.eigvals(J)
        return np.max(np.abs(eigvals))
    def __init__(self, F, Fy, Fyp, N, domain, BC, w0=None):
        self.F = F
        self.Fy = Fy
        self.Fyp = Fyp
        self.N = N
        self.domain = domain
        self.BC = BC
        self.w0 = w0
        self.x = np.linspace(domain[0], domain[1], N + 1)
        self.h = (domain[1] - domain[0]) / N
        self.boundary_conditions = Boundary_Conditions(BC)
        self.left_bc_jac = self.boundary_conditions.build_left_bc_jac()
        self.right_bc_jac = self.boundary_conditions.build_right_bc_jac()
        self.left_bc_res = self.boundary_conditions.build_left_bc_res()
        self.right_bc_res = self.boundary_conditions.build_right_bc_res()
        self.function_generator = Function_Generator(F, Fy, Fyp)
        self.fu, self.fl, self.fd = self.function_generator.build_tridiagonal_terms()
        self.residual_F = self.function_generator.build_residual_function()
        if w0 is None:
            self.w = np.linspace(domain[0], domain[1], N + 1)
        else:
            self.w = np.array(w0)

    def norm_inf(self, vector):
        return float(np.max(np.abs(vector)))

    def cond_tridiag(self, d, l, u):
        def infnorm_tridiag(d, l, u):
            n = len(d)
            max_row_sum = 0.0
            for i in range(n):
                row_sum = abs(d[i])
                if i > 0:
                    row_sum += abs(l[i - 1])
                if i < n - 1:
                    row_sum += abs(u[i])
                max_row_sum = max(max_row_sum, row_sum)
            return max_row_sum
        def inverse_infnorm_tridiag(d, l, u):
            n = len(d)
            max_norm = 0.0
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                x = TDMA(u.copy(), l.copy(), d.copy(), e)
                max_norm = max(max_norm, np.linalg.norm(x, np.inf))
            return max_norm
        return infnorm_tridiag(d, l, u) * inverse_infnorm_tridiag(d, l, u)

    def solve(self, tol=1e-10, max_iter=50, verbose=True):
        residual = Residual(self.residual_F, self.N, self.left_bc_res, self.right_bc_res)
        jacobian = Jacobian(self.fu, self.fl, self.fd, self.left_bc_jac, self.right_bc_jac)
        w = self.w.copy()
        x = self.x
        cond = 0
        for k in range(max_iter):
            u, l, d = jacobian.build(w, x)
            F = residual.build(w, x)
            cond = self.cond_tridiag(d, l, u)
            J = np.diag(d) + np.diag(l, -1) + np.diag(u, 1)
            spectral_rad = self.spectral_radius(J)
            is_diag_dom = self.check_diagonal_dominance(d, l, u)
            if verbose:
                print(f"Eigenvalues of J at iteration {k}:", np.linalg.eigvals(J))
                print(f"Iter {k}: cond(J) = {cond:.3e}")
                print(f"Iter {k}: spectral radius = {spectral_rad:.3e}")
                print(f"Iter {k}: diagonally dominant: {is_diag_dom}")
            delta = TDMA(u.copy(), l.copy(), d.copy(), -F)
            w += delta
            if self.norm_inf(delta) < tol:
                if verbose:
                    print(f"Converged in {k+1} iterations")
                return w, k+1, cond
        if verbose:
            print("Did not converge")
        return w, max_iter, cond

