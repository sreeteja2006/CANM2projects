import numpy as np

class odesolver:
    
    def __init__(self, order, method, bc, tol, max_iter, func, xstart, xend, h, blowup=False,guess = [69,420]):
        self.order = order
        self.bc = bc
        self.tol = tol
        self.max_iter = max_iter
        self.func = func
        self.guess = guess
        self.xstart = xstart
        self.xend = xend
        self.h = h
        self.method = method
        self.blowup = blowup

    def bcs(self, eps=1e-14):
        a0, b0, c0 = self.bc[0]
        a1, b1, c1 = self.bc[1]

        start_is_dir = abs(a0) > eps and abs(b0) <= eps
        start_is_neu = abs(b0) > eps and abs(a0) <= eps
        end_is_dir   = abs(a1) > eps and abs(b1) <= eps
        end_is_neu   = abs(b1) > eps and abs(a1) <= eps

        if start_is_dir and end_is_dir:
            return -c0/a0, -c1/a1, "dd"
        if start_is_neu and end_is_neu:
            return -c0/b0, -c1/b1, "nn"
        if start_is_dir and end_is_neu:
            return -c0/a0, -c1/b1, "dn"
        if start_is_neu and end_is_dir:
            return -c0/b0, -c1/a1, "nd"
        raise ValueError("Unknown/unsupported boundary condition type")

    def F(self, s, y_start, y_end, bc_type, use_eps=False,backwards=False, eps_offset=1e-3) -> float:
        if bc_type in ("dd", "dn"):
            u0 = np.array([y_start + s * eps_offset if use_eps else y_start, s])
        elif bc_type in ("nn", "nd"):
            u0 = np.array([s, y_start])
        else:
            raise ValueError("Unknown/unsupported boundary condition type")


        original_xstart = self.xstart
        if use_eps:
            self.xstart = original_xstart + eps_offset
        u, _, blew = self.method(self, u0)
        

        self.xstart = original_xstart
        
        if blew or np.any(np.isnan(u[-1])) or np.any(np.isinf(u[-1])):
            return 1e15 
        if bc_type in ("dd", "nd"):
            return u[-1, 0] - y_end
        else:  
            return u[-1, 1] - y_end

    def shooting(self) -> tuple:

        s0, s1 = 0.1, 1.0  
        ystart, yend, bc_type = self.bcs()
        use_eps = False
        blowup_detected = False
        
        F0 = self.F(s0, ystart, yend, bc_type, use_eps=False)
        F1 = self.F(s1, ystart, yend, bc_type, use_eps=False)
        
        for i in range(self.max_iter):
            if abs(F1) <= self.tol:
                print(f"Converged at iteration {i}, s = {s1:.8f}")
                break
            

            if F1 >= 1e15 or F0 >= 1e15 or np.isnan(F1) or np.isnan(F0) or abs(F1 - F0) < 1e-15:
                print(f"Blowup/numerical issue at iteration {i}. Switching to eps method...")
                blowup_detected = True
                break
            

            s_new = s1 - F1 * (s1 - s0) / (F1 - F0)
            s0, s1 = s1, s_new
            F0, F1 = F1, self.F(s1, ystart, yend, bc_type, use_eps=False)
        
    
        if blowup_detected:
            use_eps = True
            eps_offset = 1e-3
            print(f"Using eps method: starting from x = {self.xstart + eps_offset}")
            
            s0, s1 = self.guess[0], self.guess[1]
            F0 = self.F(s0, ystart, yend, bc_type, use_eps=True, eps_offset=eps_offset)
            F1 = self.F(s1, ystart, yend, bc_type, use_eps=True, eps_offset=eps_offset)
            
            for j in range(self.max_iter):
                if abs(F1) <= self.tol:
                    blowup_detected = False
                    print(f"Eps method converged at iteration {j}, s = {s1:.8f}")
                    break

                if F1 >= 1e15 or F0 >= 1e15 or np.isnan(F1) or np.isnan(F0) or abs(F1 - F0) < 1e-15:
                    print(f"Eps method also failed at iteration {j}. Using best guess.")
                    use_eps = False
                    break
                
                s_new = s1 - F1 * (s1 - s0) / (F1 - F0)
                s0, s1 = s1, s_new
                F0, F1 = F1, self.F(s1, ystart, yend, bc_type, use_eps=True, eps_offset=eps_offset)
        return s1, bc_type, ystart, yend, use_eps
    
    def solve(self):
        s, bc_type, start_val, end_val, use_eps = self.shooting()
        eps_offset = 1e-3

        if bc_type in ("dd", "dn"):
            if use_eps:
                u0 = np.array([start_val + s * eps_offset, s])
                self.xstart = self.xstart + eps_offset
            else:
                u0 = np.array([start_val, s])
        else:
            u0 = np.array([s, start_val])

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


    

    