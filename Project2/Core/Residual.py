import numpy as np


class Residual:
    def __init__(self, F, N, left_bc_res, right_bc_res):
        self.F = F
        self.N = N
        self.left_bc_res = left_bc_res
        self.right_bc_res = right_bc_res

    def build(self, w, x):
        N = self.N
        h = x[1] - x[0]
        res = np.zeros(N + 1)
        #left BC
        res[0] = self.left_bc_res(w, h)
        #Right BC
        res[N] = self.right_bc_res(w, h)

        # Interior points
        for i in range(1, N):
            res[i] = self.F(i, x, w, h)

        return res
    
