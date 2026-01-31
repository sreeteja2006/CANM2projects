import numpy as np
import matplotlib.pyplot as plt
class ODESolver:
    def __init__(self,method,bc,degree,f0,x,h,tol,s):
        self.method = method
        self.degree = degree
        self.f0 =f0
        self.x0 =x[0]
        self.h =h
        self.bc = bc
        self.x1 =x[1]
        self.s =s
        self.tol =tol
    def function_constructor(self):
        def f(u,x):
            arr = np.zeros(len(u))
            for i in range(self.degree):
                arr[i] = u[i+1]
            arr[self.degree] = self.f0(x, u)
            return arr
        self.f = f
    def ABM2(self, f, u0, x0, h, n):
        u = np.zeros((n+1, len(u0)))
        x = np.zeros(n+1)

        u[0] = u0
        x[0] = x0

        def RK4_step(f, u, x, h):
            k1 = h*f(u, x)
            k2 = h*f(u + k1/2, x + h/2)
            k3 = h*f(u + k2/2, x + h/2)
            k4 = h*f(u + k3, x + h)
            return u + (k1 + 2*k2 + 2*k3 + k4)/6

        u[1] = RK4_step(f, u[0], x[0], h)
        x[1] = x[0] + h

        for i in range(1, n):
            x[i+1] = x[i] + h
            up = u[i] + h/2*(3*f(u[i], x[i]) - f(u[i-1], x[i-1]))
            u[i+1] = u[i] + h/2*(f(up, x[i+1]) + f(u[i], x[i]))

        return u, x

    def shootingmethod(self):
        self.function_constructor()

        num_steps = int((self.x1 - self.x0) / self.h)

        u0 = np.zeros(self.degree + 1)
        known_left = {}
        right_bcs = []

        for bc in self.bc:
            side, var, value = bc
            if side.lower() == "left":
                known_left[var] = value
            elif side.lower() == "right":
                right_bcs.append((var, value))

        for var, value in known_left.items():
            u0[var] = value

        unknown_idx = [i for i in range(self.degree + 1) if i not in known_left]

        if len(unknown_idx) != 1:
            raise ValueError("Shooting requires exactly one unknown initial condition")

        k = unknown_idx[0]
        if self.method.lower() == "abm2" or self.method.lower() == "abm_2":
            method_func = self.ABM2
        elif self.method.lower() == "rk4":
            method_func = self.ABM2
        else:
            method_func = self.method  
        
        def F(s):
            u0[k] = s
            sol, _ = method_func(self.f, u0, self.x0, self.h, num_steps)
            res = 0.0
            for var, value in right_bcs:
                res += sol[-1, var] - value
            return res
        
        # IMPLEMENT JACOBIAN

        s0, s1 = self.s
        F0, F1 = F(s0), F(s1)
        while abs(F1) > self.tol:
            s2 = s1 - F1 * (s1 - s0) / (F1 - F0)
            s0, s1 = s1, s2
            F0, F1 = F1, F(s1)

        u0[k] = s1
        sol, x = method_func(self.f, u0, self.x0, self.h, num_steps)
        return sol, x

    def exact_solution(self, x):
        return 10**-4*np.sqrt(100020000*x+1)-1
    

           
        