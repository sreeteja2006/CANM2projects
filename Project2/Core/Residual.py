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
        # Validate N
        if N is None:
            raise ValueError("N (number of grid intervals) must be provided")
        if not isinstance(N, (int, np.integer)):
            raise TypeError(f"N must be an integer, got {type(N).__name__}")
        if N < 1:
            raise ValueError(f"N must be positive, got {N}")
        
        # Validate functions
        if F is None:
            raise ValueError("F function must be provided")
        if not callable(F):
            raise TypeError("F must be callable")
        
        if left_bc_res is None:
            raise ValueError("left_bc_res function must be provided")
        if not callable(left_bc_res):
            raise TypeError("left_bc_res must be callable")
        
        if right_bc_res is None:
            raise ValueError("right_bc_res function must be provided")
        if not callable(right_bc_res):
            raise TypeError("right_bc_res must be callable")
        
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
        # Validate inputs
        if w is None or x is None:
            raise ValueError("Both w and x must be provided")
        
        # Convert to numpy arrays
        try:
            w = np.asarray(w, dtype=float)
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError("w and x must be convertible to numeric arrays") from e
        
        # Check dimensions
        if w.ndim != 1 or x.ndim != 1:
            raise ValueError("w and x must be 1-dimensional arrays")
        
        N = self.N
        
        # Check array lengths
        if len(w) != N + 1:
            raise ValueError(f"w must have length {N+1}, got {len(w)}")
        if len(x) != N + 1:
            raise ValueError(f"x must have length {N+1}, got {len(x)}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            raise ValueError("w contains NaN or infinite values")
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("x contains NaN or infinite values")
        
        # Check grid spacing
        if len(x) < 2:
            raise ValueError("x must have at least 2 points")
        
        h = x[1] - x[0]
        if abs(h) < ZERO_TOLERANCE:
            raise ValueError(f"Grid spacing h is too small or zero: {h}")
        
        # Check grid uniformity (optional warning)
        if len(x) > 2:
            diffs = np.diff(x)
            if not np.allclose(diffs, h, rtol=GRID_UNIFORMITY_TOLERANCE):
                import warnings
                warnings.warn("Grid spacing appears to be non-uniform")
        
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
            try:
                res[i] = self.F(i, x, w, h)
            except Exception as e:
                raise RuntimeError(f"Error computing residual at interior point {i}: {e}") from e
        
        # Validate result
        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            raise RuntimeError("Residual contains NaN or infinite values")

        return res
    
