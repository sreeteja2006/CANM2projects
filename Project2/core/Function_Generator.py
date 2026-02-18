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

        def fu(x, w, h, i):
            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed ypi is NaN or infinite at i={i}")

            try:
                Fyp_val = self.Fyp(xi, yi, ypi)
            except Exception as e:
                raise RuntimeError(f"Error evaluating Fyp at interior point {i}: {e}") from e
            
            if np.isnan(Fyp_val) or np.isinf(Fyp_val):
                raise ValueError(f"Fyp returned NaN or infinite at i={i}")

            result = -1 + (h/2) * Fyp_val
            
            
            return result

        def fl(x, w, h, i):
            
            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed ypi is NaN or infinite at i={i}")

            try:
                Fyp_val = self.Fyp(xi, yi, ypi)
            except Exception as e:
                raise RuntimeError(f"Error evaluating Fyp at interior point {i}: {e}") from e
            
            if np.isnan(Fyp_val) or np.isinf(Fyp_val):
                raise ValueError(f"Fyp returned NaN or infinite at i={i}")

            result = -1 - (h/2) * Fyp_val
            
            return result

        def fd(x, w, h, i):

            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)
            
            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed ypi is NaN or infinite at i={i}")

            try:
                Fy_val = self.Fy(xi, yi, ypi)
            except Exception as e:
                raise RuntimeError(f"Error evaluating Fy at interior point {i}: {e}") from e
            
            if np.isnan(Fy_val) or np.isinf(Fy_val):
                raise ValueError(f"Fy returned NaN or infinite at i={i}")

            result = 2 + h**2 * Fy_val
            
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

            if abs(h) < ZERO_TOLERANCE:
                raise ValueError(f"Grid spacing h is too small or zero: {h}")
            
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            if np.isnan(ypi) or np.isinf(ypi):
                raise ValueError(f"Computed derivative is NaN or infinite at i={i}")

            try:
                F_val = self.F(xi, yi, ypi)
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

            return result

        return residual