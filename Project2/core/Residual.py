import numpy as np
from .constants import ZERO_TOLERANCE, GRID_UNIFORMITY_TOLERANCE


class Residual:
    def __init__(self, F, N, left_bc_res, right_bc_res):
        """
        Initialize Residual calculator.
        
        Parameters:
        F: Function for interior points residual calculation
        N: Number of grid intervals
        left_bc_res: Function for left boundary condition residual
        right_bc_res: Function for right boundary condition residual
        """
        self.F = F
        self.N = N
        self.left_bc_res = left_bc_res
        self.right_bc_res = right_bc_res

    def build(self, w, x):
        """
        Build residual vector.
        
        Parameters:
        w: Solution vector
        x: Grid points
        
        Returns:
        res: Residual vector
        """
        w = np.asarray(w, dtype=float)
        x = np.asarray(x, dtype=float)
        
        N = self.N
        h = x[1] - x[0]
        
        res = np.zeros(N + 1)
        
        # Left BC
        try:
            res[0] = self.left_bc_res(w, h)
        except Exception as e:
            raise RuntimeError(f"Error computing left boundary condition residual: {e}") from e
        
        # Right BC
        try:
            res[N] = self.right_bc_res(w, h)
        except Exception as e:
            raise RuntimeError(f"Error computing right boundary condition residual: {e}") from e

        # Interior points
        for i in range(1, N):
            res[i] = self.F(i, x, w, h)
        
        return res
    
