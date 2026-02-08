import numpy as np

class BoundaryConditions:
    def __init__(self, bc_start, bc_end, eps=1e-14):
        self.bc_start = bc_start
        self.bc_end = bc_end
        self.eps = eps
    def classify(self):
        a0, b0, c0 = self.bc_start
        a1, b1, c1 = self.bc_end
        start_is_dir = abs(a0) > self.eps and abs(b0) <= self.eps
        start_is_neu = abs(b0) > self.eps and abs(a0) <= self.eps
        end_is_dir = abs(a1) > self.eps and abs(b1) <= self.eps
        end_is_neu = abs(b1) > self.eps and abs(a1) <= self.eps
        if start_is_dir and end_is_dir:
            return -c0/a0, -c1/a1, "dd"
        if start_is_neu and end_is_neu:
            return -c0/b0, -c1/b1, "nn"
        if start_is_dir and end_is_neu:
            return -c0/a0, -c1/b1, "dn"
        if start_is_neu and end_is_dir:
            return -c0/b0, -c1/a1, "nd"
        raise ValueError("Unknown/unsupported boundary condition type")

class ShootingMethod:
    def __init__(self, solver, tol=1e-8, max_iter=100):
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
    def _build_initial_condition(self, s, y_start, bc_type, use_eps=False, eps_offset=1e-3):
        if bc_type in ("dd", "dn"):
            y0 = y_start + s * eps_offset if use_eps else y_start
            return np.array([y0, s])
        elif bc_type in ("nn", "nd"):
            return np.array([s, y_start])
        else:
            raise ValueError(f"Unknown BC type: {bc_type}")
    def _compute_residual(self, u_end, y_end, bc_type):
        if bc_type in ("dd", "nd"):
            return u_end[0] - y_end
        else:
            return u_end[1] - y_end
    def _evaluate(self, s, y_start, y_end, bc_type, use_eps=False, eps_offset=1e-3, true_xstart=None):
        original_xstart = true_xstart if true_xstart is not None else self.solver.xstart
        effective_xstart = original_xstart + eps_offset if use_eps else original_xstart
        u0 = self._build_initial_condition(s, y_start, bc_type, use_eps, eps_offset)
        self.solver.xstart = effective_xstart
        u, _, blew = self.solver.method(self.solver, u0)
        self.solver.xstart = original_xstart
        if blew or np.any(np.isnan(u[-1])) or np.any(np.isinf(u[-1])):
            return 1e15
        return self._compute_residual(u[-1], y_end, bc_type)
    def _secant_iteration(self, s0, s1, y_start, y_end, bc_type, use_eps=False, eps_offset=1e-3, true_xstart=None):
        F0 = self._evaluate(s0, y_start, y_end, bc_type, use_eps, eps_offset, true_xstart)
        F1 = self._evaluate(s1, y_start, y_end, bc_type, use_eps, eps_offset, true_xstart)
        for i in range(self.max_iter):
            if abs(F1) <= self.tol:
                print(f"Converged at iteration {i}, s = {s1:.8f}")
                return s1, True
            if F1 >= 1e15 or F0 >= 1e15 or np.isnan(F1) or np.isnan(F0) or abs(F1 - F0) < 1e-15:
                print(f"Numerical issue at iteration {i}")
                return s1, False
            s_new = s1 - F1 * (s1 - s0) / (F1 - F0)
            s0, s1 = s1, s_new
            F0, F1 = F1, self._evaluate(s1, y_start, y_end, bc_type, use_eps, eps_offset, true_xstart)
        return s1, abs(F1) <= self.tol
    def solve(self, guess, y_start, y_end, bc_type):
        true_xstart = self.solver._original_xstart
        s0, s1 = guess
        s, success = self._secant_iteration(s0, s1, y_start, y_end, bc_type, use_eps=False, true_xstart=true_xstart)
        if success:
            return s, False
        eps_offset = 1e-3
        print(f"Switching to eps method: starting from x = {true_xstart + eps_offset}")
        s, success = self._secant_iteration(s0, s1, y_start, y_end, bc_type, use_eps=True, eps_offset=eps_offset, true_xstart=true_xstart)
        if success:
            return s, True
        print("Eps method also failed. Using best guess.")
        return s, False

class odesolver:
    def __init__(self, order, method, bc, tol, max_iter, func, xstart, xend, h, blowup=False, guess=[69, 420]):
        self.order = order
        self.bc = bc
        self.tol = tol
        self.max_iter = max_iter
        self.func = func
        self.guess = guess
        self.xstart = xstart
        self._original_xstart = xstart
        self.xend = xend
        self.h = h
        self.method = method
        self.blowup = blowup
    def bcs(self, eps=1e-14):
        bc_handler = BoundaryConditions(self.bc[0], self.bc[1], eps)
        return bc_handler.classify()
    def F(self, s, y_start, y_end, bc_type, use_eps=False, backwards=False, eps_offset=1e-3, true_xstart=None):
        shooter = ShootingMethod(self, self.tol, self.max_iter)
        return shooter._evaluate(s, y_start, y_end, bc_type, use_eps, eps_offset, true_xstart)
    def shooting(self):
        y_start, y_end, bc_type = self.bcs()
        shooter = ShootingMethod(self, self.tol, self.max_iter)
        s, use_eps = shooter.solve(self.guess, y_start, y_end, bc_type)
        return s, bc_type, y_start, y_end, use_eps
    def solve(self):
        s, bc_type, start_val, end_val, use_eps = self.shooting()
        eps_offset = 1e-3
        shooter = ShootingMethod(self, self.tol, self.max_iter)
        if bc_type in ("dd", "dn"):
            if use_eps:
                u0 = np.array([start_val + s * eps_offset, s])
                self.xstart = self._original_xstart + eps_offset
            else:
                u0 = np.array([start_val, s])
                self.xstart = self._original_xstart
        else:
            u0 = np.array([s, start_val])
            self.xstart = self._original_xstart
        u, x, blew = self.method(self, u0)
        return u, x, s, blew, bc_type
    def get_order(self):
        return self.order
    def set_order(self, order):
        self.order = order
    def get_method(self):
        return self.method
    def set_method(self, method):
        if not callable(method):
            raise TypeError(f"Method must be callable, got {method} ({type(method)})")
        self.method = method
    def get_bc(self):
        return self.bc
    def set_bc(self, bc):
        self.bc = bc
    def get_tol(self):
        return self.tol
    def set_tol(self, tol):
        self.tol = tol
    def get_max_iter(self):
        return self.max_iter
    def set_max_iter(self, max_iter):
        self.max_iter = max_iter
    def get_func(self):
        return self.func
    def set_func(self, func):
        self.func = func
    def get_guess(self):
        return self.guess
    def set_guess(self, guess):
        self.guess = guess
    def get_xstart(self):
        return self.xstart
    def set_xstart(self, xstart):
        self.xstart = xstart
    def get_xend(self):
        return self.xend
    def set_xend(self, xend):
        self.xend = xend
    def get_h(self):
        return self.h
    def set_h(self, h):
        self.h = h
    def get_blowup(self):
        return self.blowup
    def set_blowup(self, blowup):
        self.blowup = blowup