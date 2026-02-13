import numpy as np


class Jacobian:

    def __init__(self, fu, fl, fd, bc_left_jac, bc_right_jac):
        self.fu = fu
        self.fl = fl
        self.fd = fd
        self.bc_left_jac = bc_left_jac
        self.bc_right_jac = bc_right_jac

    def build(self, w, x):
        N = len(w) - 1
        h = x[1] - x[0]

        u = np.zeros(N)
        l = np.zeros(N)
        d = np.zeros(N + 1)

        # Left BC row
        d[0], u[0] = self.bc_left_jac(w, h)

        # Right BC row
        l[-1], d[-1] = self.bc_right_jac(w, h)

        # Interior rows
        for i in range(1, N):
            u[i] = self.fu(x, w, h, i)
            l[i-1] = self.fl(x, w, h, i)
            d[i] = self.fd(x, w, h, i)

        return u, l, d
