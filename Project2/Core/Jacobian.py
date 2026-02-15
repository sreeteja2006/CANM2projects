import numpy as np
from .constants import ZERO_TOLERANCE

class Jacobian:
    """
    Jacobian matrix builder for tridiagonal systems.
    """

    def __init__(self, fu, fl, fd, bc_left_jac, bc_right_jac):
        """
        Initialize Jacobian calculator.
        
        Parameters:
        fu: Function for upper diagonal elements
        fl: Function for lower diagonal elements
        fd: Function for main diagonal elements
        bc_left_jac: Function for left boundary condition Jacobian
        bc_right_jac: Function for right boundary condition Jacobian
        """
        # Validate functions
        if fu is None:
            raise ValueError("fu function must be provided")
        if not callable(fu):
            raise TypeError("fu must be callable")
        
        if fl is None:
            raise ValueError("fl function must be provided")
        if not callable(fl):
            raise TypeError("fl must be callable")
        
        if fd is None:
            raise ValueError("fd function must be provided")
        if not callable(fd):
            raise TypeError("fd must be callable")
        
        if bc_left_jac is None:
            raise ValueError("bc_left_jac function must be provided")
        if not callable(bc_left_jac):
            raise TypeError("bc_left_jac must be callable")
        
        if bc_right_jac is None:
            raise ValueError("bc_right_jac function must be provided")
        if not callable(bc_right_jac):
            raise TypeError("bc_right_jac must be callable")
        
        self.fu = fu
        self.fl = fl
        self.fd = fd
        self.bc_left_jac = bc_left_jac
        self.bc_right_jac = bc_right_jac

    def build(self, w, x):
        """
        Build Jacobian matrix diagonals.
        
        Parameters:
        w: Solution vector
        x: Grid points
        
        Returns:
        u: Upper diagonal (length N)
        l: Lower diagonal (length N)
        d: Main diagonal (length N+1)
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
        
        # Check minimum size
        if len(w) < 2:
            raise ValueError(f"w must have at least 2 elements, got {len(w)}")
        
        N = len(w) - 1
        
        # Check array lengths match
        if len(x) != len(w):
            raise ValueError(f"Length mismatch: w has length {len(w)} but x has length {len(x)}")
        
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

        u = np.zeros(N)
        l = np.zeros(N)
        d = np.zeros(N + 1)

        # Left BC row
        try:
            d[0], u[0] = self.bc_left_jac(w, h)
        except Exception as e:
            raise RuntimeError(f"Error computing left boundary Jacobian: {e}") from e
        
        # Validate left BC outputs
        if np.isnan(d[0]) or np.isinf(d[0]) or np.isnan(u[0]) or np.isinf(u[0]):
            raise ValueError("Left boundary Jacobian returned NaN or infinite values")

        # Right BC row
        try:
            l[-1], d[-1] = self.bc_right_jac(w, h)
        except Exception as e:
            raise RuntimeError(f"Error computing right boundary Jacobian: {e}") from e
        
        # Validate right BC outputs
        if np.isnan(l[-1]) or np.isinf(l[-1]) or np.isnan(d[-1]) or np.isinf(d[-1]):
            raise ValueError("Right boundary Jacobian returned NaN or infinite values")

        # Interior rows
        for i in range(1, N):
            try:
                u[i] = self.fu(x, w, h, i)
                l[i-1] = self.fl(x, w, h, i)
                d[i] = self.fd(x, w, h, i)
            except Exception as e:
                raise RuntimeError(f"Error computing Jacobian at interior point {i}: {e}") from e
            
            # Check for invalid values
            if np.isnan(u[i]) or np.isinf(u[i]):
                raise ValueError(f"Upper diagonal element u[{i}] is NaN or infinite")
            if np.isnan(l[i-1]) or np.isinf(l[i-1]):
                raise ValueError(f"Lower diagonal element l[{i-1}] is NaN or infinite")
            if np.isnan(d[i]) or np.isinf(d[i]):
                raise ValueError(f"Main diagonal element d[{i}] is NaN or infinite")

        return u, l, d
