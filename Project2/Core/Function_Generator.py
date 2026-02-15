import numpy as np
import warnings
from .constants import ZERO_TOLERANCE

class Function_Generator:
    def __init__(self, F, Fy, Fyp):
        """
        Initialize Function Generator.
        
        Parameters:
        F: Function for y'' = F(x, y, y')
        Fy: Partial derivative of F with respect to y (can be None)
        Fyp: Partial derivative of F with respect to y' (can be None)
        """
        # Validate F
        if F is None:
            raise ValueError("F function must be provided")
        if not callable(F):
            raise TypeError("F must be callable")
        
        # Validate Fy (optional but must be callable if provided)
        if Fy is not None and not callable(Fy):
            raise TypeError("Fy must be callable or None")
        
        # Validate Fyp (optional but must be callable if provided)
        if Fyp is not None and not callable(Fyp):
            raise TypeError("Fyp must be callable or None")
        
        self.F = F
        self.Fy = Fy
        self.Fyp = Fyp

    def build_tridiagonal_terms(self):
        """
        Build tridiagonal matrix terms for the Jacobian.
        
        Returns:
        fu: Upper diagonal function
        fl: Lower diagonal function
        fd: Main diagonal function
        """
        if self.Fy is None:
            raise ValueError("Fy must be provided to build tridiagonal terms")
        if self.Fyp is None:
            raise ValueError("Fyp must be provided to build tridiagonal terms")

        def fu(x, w, h, i):
            # Validate inputs
            if i < 0 or i >= len(x):
                raise IndexError(f"Index i={i} out of bounds for array of length {len(x)}")
            if i+1 >= len(w) or i-1 < 0:
                raise IndexError(f"Cannot compute derivative at i={i}: need w[{i-1}] to w[{i+1}]")
            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(xi) or np.isinf(xi):
                raise ValueError(f"x[{i}] is NaN or infinite")
            if np.isnan(yi) or np.isinf(yi):
                raise ValueError(f"w[{i}] is NaN or infinite")
            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed ypi is NaN or infinite at i={i}")

            try:
                Fyp_val = self.Fyp(xi, yi, ypi)
            except Exception as e:
                raise RuntimeError(f"Error evaluating Fyp at interior point {i}: {e}") from e
            
            if np.isnan(Fyp_val) or np.isinf(Fyp_val):
                raise ValueError(f"Fyp returned NaN or infinite at i={i}")

            result = -1 + (h/2) * Fyp_val
            
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"Upper diagonal term is NaN or infinite at i={i}")
            
            return result

        def fl(x, w, h, i):
            # Validate inputs
            if i < 0 or i >= len(x):
                raise IndexError(f"Index i={i} out of bounds for array of length {len(x)}")
            if i+1 >= len(w) or i-1 < 0:
                raise IndexError(f"Cannot compute derivative at i={i}: need w[{i-1}] to w[{i+1}]")
            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(xi) or np.isinf(xi):
                raise ValueError(f"x[{i}] is NaN or infinite")
            if np.isnan(yi) or np.isinf(yi):
                raise ValueError(f"w[{i}] is NaN or infinite")
            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed ypi is NaN or infinite at i={i}")

            try:
                Fyp_val = self.Fyp(xi, yi, ypi)
            except Exception as e:
                raise RuntimeError(f"Error evaluating Fyp at interior point {i}: {e}") from e
            
            if np.isnan(Fyp_val) or np.isinf(Fyp_val):
                raise ValueError(f"Fyp returned NaN or infinite at i={i}")

            result = -1 - (h/2) * Fyp_val
            
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"Lower diagonal term is NaN or infinite at i={i}")
            
            return result

        def fd(x, w, h, i):
            # Validate inputs
            if i < 0 or i >= len(x):
                raise IndexError(f"Index i={i} out of bounds for array of length {len(x)}")
            if i+1 >= len(w) or i-1 < 0:
                raise IndexError(f"Cannot compute derivative at i={i}: need w[{i-1}] to w[{i+1}]")
            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(xi) or np.isinf(xi):
                raise ValueError(f"x[{i}] is NaN or infinite")
            if np.isnan(yi) or np.isinf(yi):
                raise ValueError(f"w[{i}] is NaN or infinite")
            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed ypi is NaN or infinite at i={i}")

            try:
                Fy_val = self.Fy(xi, yi, ypi)
            except Exception as e:
                raise RuntimeError(f"Error evaluating Fy at interior point {i}: {e}") from e
            
            if np.isnan(Fy_val) or np.isinf(Fy_val):
                raise ValueError(f"Fy returned NaN or infinite at i={i}")

            result = 2 + h**2 * Fy_val
            
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"Main diagonal term is NaN or infinite at i={i}")
            
            return result

        return fu, fl, fd

    def build_residual_function(self):
        """
        Build residual function for interior points.
        
        Returns:
        Function that computes residual at given point
        """
        if self.F is None:
            raise ValueError("F must be provided to build residual function")

        def residual(i, x, w, h):
            # Validate inputs
            if i < 0 or i >= len(x):
                raise IndexError(f"Index i={i} out of bounds for array of length {len(x)}")
            if i+1 >= len(w) or i-1 < 0:
                raise IndexError(f"Cannot compute residual at i={i}: need w[{i-1}] to w[{i+1}]")
            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            sip = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(xi) or np.isinf(xi):
                raise ValueError(f"x[{i}] is NaN or infinite")
            if np.isnan(yi) or np.isinf(yi):
                raise ValueError(f"w[{i}] is NaN or infinite")
            if np.isnan(w[i-1]) or np.isinf(w[i-1]):
                raise ValueError(f"w[{i-1}] is NaN or infinite")
            if np.isnan(w[i+1]) or np.isinf(w[i+1]):
                raise ValueError(f"w[{i+1}] is NaN or infinite")
            if np.isnan(sip) or np.isinf(sip):
                raise ValueError(f"Computed derivative is NaN or infinite at i={i}")

            try:
                F_val = self.F(xi, yi, sip)
            except Exception as e:
                raise RuntimeError(f"Error evaluating F at interior point {i}: {e}") from e
            
            if np.isnan(F_val) or np.isinf(F_val):
                raise ValueError(f"F returned NaN or infinite at i={i}")

            result = (
                -w[i-1]
                + 2*w[i]
                - w[i+1]
                + h**2 * F_val
            )
            
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"Residual is NaN or infinite at i={i}")

            return result

        return residual