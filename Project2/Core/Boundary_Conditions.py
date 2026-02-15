import numpy as np
from enum import Enum
import warnings
from .constants import ZERO_TOLERANCE

class BCType(Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    ROBIN = "Robin"

class Boundary_Conditions:
    def __init__(self, BC: np.ndarray):
        """
        Initialize boundary conditions.
        
        Parameters:
        BC: 2x3 array of boundary condition coefficients
            Format: a*y + b*y' + c = 0 for each boundary
            [[a_left, b_left, c_left],
             [a_right, b_right, c_right]]
        """
        # Validate BC input
        if BC is None:
            raise ValueError("BC (boundary conditions) must be provided")
        
        # Convert to numpy array
        try:
            BC = np.asarray(BC, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError("BC must be convertible to numeric array") from e
        
        # Validate shape
        if BC.shape != (2, 3):
            raise ValueError(f"BC must have shape (2, 3), got {BC.shape}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(BC)) or np.any(np.isinf(BC)):
            raise ValueError("BC contains NaN or infinite values")
        
        # Validate that at least one coefficient (a or b) is non-zero for each boundary
        for i, side in enumerate(["left", "right"]):
            a, b, c = BC[i]
            if abs(a) < ZERO_TOLERANCE and abs(b) < ZERO_TOLERANCE:
                raise ValueError(f"Both a and b are zero for {side} BC. At least one must be non-zero.")
        
        self.BC = BC
        self.bc_types = self.get_bc_type(BC)


    @staticmethod
    def get_bc_type(BC: np.ndarray) -> list[BCType]:
        """
        Determine boundary condition type.
        
        BC format:
        For each boundary:
            a*y + b*y' + c= 0
        BC is shape (2, 3):
            [[a0, b0, c0],
            [a1, b1, c1]]
        
        Returns:
        List of BCType enums for left and right boundaries
        """
        bc_types = []

        for i in range(2):
            a, b, _ = BC[i]

            if abs(b) < ZERO_TOLERANCE:
                bc_types.append(BCType.DIRICHLET)
            elif abs(a) < ZERO_TOLERANCE:
                bc_types.append(BCType.NEUMANN)
            else:
                bc_types.append(BCType.ROBIN)

        return bc_types

    def build_left_bc_jac(self, Fy, Fyp, x0):
        """
        Build left boundary condition Jacobian.
        
        Parameters:
        Fy: Partial derivative of F with respect to y
        Fyp: Partial derivative of F with respect to y'
        x0: Left boundary point
        
        Returns:
        Function that computes left BC Jacobian entries
        """
        # Validate inputs
        if Fy is None or Fyp is None:
            raise ValueError("Fy and Fyp must be provided for Jacobian computation")
        if not callable(Fy):
            raise TypeError("Fy must be callable")
        if not callable(Fyp):
            raise TypeError("Fyp must be callable")
        if x0 is None:
            raise ValueError("x0 (left boundary point) must be provided")
        
        try:
            x0 = float(x0)
        except (ValueError, TypeError) as e:
            raise TypeError("x0 must be numeric") from e
        
        if np.isnan(x0) or np.isinf(x0):
            raise ValueError("x0 is NaN or infinite")

        a, b, c = self.BC[0]

        if self.bc_types[0] == BCType.DIRICHLET:
            return lambda w, h: (a, 0.0)

        else:
            # For Neumann or Robin BC, need to ensure b != 0
            if abs(b) < ZERO_TOLERANCE:
                raise ValueError("Division by zero: b coefficient is zero for non-Dirichlet left BC")

            def jac(w, h):
                # Validate inputs
                if w is None or h is None:
                    raise ValueError("w and h must be provided")
                if len(w) < 1:
                    raise ValueError("w must have at least 1 element")
                if abs(h) < ZERO_TOLERANCE:
                    raise ValueError(f"Grid spacing h is too small or zero: {h}")

                y0 = w[0]
                
                if np.isnan(y0) or np.isinf(y0):
                    raise ValueError("y0 is NaN or infinite")
                
                Yp = (-c - a*y0) / b
                
                if np.isnan(Yp) or np.isinf(Yp):
                    raise ValueError("Computed Yp is NaN or infinite")

                try:
                    Fy_val = Fy(x0, y0, Yp)
                    Fyp_val = Fyp(x0, y0, Yp)
                except Exception as e:
                    raise RuntimeError(f"Error evaluating Fy or Fyp at left boundary: {e}") from e
                
                if np.isnan(Fy_val) or np.isinf(Fy_val):
                    raise ValueError("Fy returned NaN or infinite at left boundary")
                if np.isnan(Fyp_val) or np.isinf(Fyp_val):
                    raise ValueError("Fyp returned NaN or infinite at left boundary")

                d0 = (
                    2*(1 - h*a/b)
                    + h**2 * Fy_val
                    - h**2 * (a/b) * Fyp_val
                )

                u0 = -2.0
                
                if np.isnan(d0) or np.isinf(d0):
                    raise ValueError("Computed d0 is NaN or infinite")

                return d0, u0

            return jac


    def build_right_bc_jac(self, Fy, Fyp, xN):
        """
        Build right boundary condition Jacobian.
        
        Parameters:
        Fy: Partial derivative of F with respect to y
        Fyp: Partial derivative of F with respect to y'
        xN: Right boundary point
        
        Returns:
        Function that computes right BC Jacobian entries
        """
        # Validate inputs
        if Fy is None or Fyp is None:
            raise ValueError("Fy and Fyp must be provided for Jacobian computation")
        if not callable(Fy):
            raise TypeError("Fy must be callable")
        if not callable(Fyp):
            raise TypeError("Fyp must be callable")
        if xN is None:
            raise ValueError("xN (right boundary point) must be provided")
        
        try:
            xN = float(xN)
        except (ValueError, TypeError) as e:
            raise TypeError("xN must be numeric") from e
        
        if np.isnan(xN) or np.isinf(xN):
            raise ValueError("xN is NaN or infinite")

        a, b, c = self.BC[1]

        if self.bc_types[1] == BCType.DIRICHLET:
            return lambda w, h: (0.0, a)
        else:
            # For Neumann or Robin BC, need to ensure b != 0
            if abs(b) < ZERO_TOLERANCE:
                raise ValueError("Division by zero: b coefficient is zero for non-Dirichlet right BC")

            def jac(w, h):
                # Validate inputs
                if w is None or h is None:
                    raise ValueError("w and h must be provided")
                if len(w) < 1:
                    raise ValueError("w must have at least 1 element")
                if abs(h) < ZERO_TOLERANCE:
                    raise ValueError(f"Grid spacing h is too small or zero: {h}")

                yNp1 = w[-1]
                
                if np.isnan(yNp1) or np.isinf(yNp1):
                    raise ValueError("yNp1 is NaN or infinite")
                
                yp = (-c - a*yNp1) / b
                
                if np.isnan(yp) or np.isinf(yp):
                    raise ValueError("Computed yp is NaN or infinite")

                try:
                    Fy_val = Fy(xN, yNp1, yp)
                    Fyp_val = Fyp(xN, yNp1, yp)
                except Exception as e:
                    raise RuntimeError(f"Error evaluating Fy or Fyp at right boundary: {e}") from e
                
                if np.isnan(Fy_val) or np.isinf(Fy_val):
                    raise ValueError("Fy returned NaN or infinite at right boundary")
                if np.isnan(Fyp_val) or np.isinf(Fyp_val):
                    raise ValueError("Fyp returned NaN or infinite at right boundary")

                dNp1 = (
                    2*(1 + h*a/b)
                    + h**2 * Fy_val
                    - h**2 * (a/b) * Fyp_val
                )

                lNp1 = -2.0
                
                if np.isnan(dNp1) or np.isinf(dNp1):
                    raise ValueError("Computed dNp1 is NaN or infinite")

                return lNp1, dNp1

            return jac

    def build_left_bc_res(self, F, x0):
        """
        Build left boundary condition residual.
        
        Parameters:
        F: Function F(x, y, y')
        x0: Left boundary point
        
        Returns:
        Function that computes left BC residual
        """
        # Validate inputs
        if F is None:
            raise ValueError("F must be provided")
        if not callable(F):
            raise TypeError("F must be callable")
        if x0 is None:
            raise ValueError("x0 (left boundary point) must be provided")
        
        try:
            x0 = float(x0)
        except (ValueError, TypeError) as e:
            raise TypeError("x0 must be numeric") from e
        
        if np.isnan(x0) or np.isinf(x0):
            raise ValueError("x0 is NaN or infinite")

        a, b, c = self.BC[0]

        if self.bc_types[0] == BCType.DIRICHLET:
            return lambda w, h: a*w[0] + c

        else:
            # For Neumann or Robin BC, need to ensure b != 0
            if abs(b) < ZERO_TOLERANCE:
                raise ValueError("Division by zero: b coefficient is zero for non-Dirichlet left BC")

            def res(w, h):
                # Validate inputs
                if w is None or h is None:
                    raise ValueError("w and h must be provided")
                if len(w) < 2:
                    raise ValueError("w must have at least 2 elements for non-Dirichlet BC")
                if abs(h) < ZERO_TOLERANCE:
                    raise ValueError(f"Grid spacing h is too small or zero: {h}")

                y0 = w[0]
                y1 = w[1]
                
                if np.isnan(y0) or np.isinf(y0) or np.isnan(y1) or np.isinf(y1):
                    raise ValueError("y0 or y1 is NaN or infinite")

                yp = (-c - a*y0) / b
                
                if np.isnan(yp) or np.isinf(yp):
                    raise ValueError("Computed yp is NaN or infinite")

                try:
                    F_val = F(x0, y0, yp)
                except Exception as e:
                    raise RuntimeError(f"Error evaluating F at left boundary: {e}") from e
                
                if np.isnan(F_val) or np.isinf(F_val):
                    raise ValueError("F returned NaN or infinite at left boundary")

                result = (
                    2*(1 - h*a/b)*y0
                    - 2*y1
                    + h**2 * F_val
                    - 2*h *c/b
                )
                
                if np.isnan(result) or np.isinf(result):
                    raise ValueError("Computed residual is NaN or infinite")

                return result

            return res


    def build_right_bc_res(self, F, xN):
        """
        Build right boundary condition residual.
        
        Parameters:
        F: Function F(x, y, y')
        xN: Right boundary point
        
        Returns:
        Function that computes right BC residual
        """
        # Validate inputs
        if F is None:
            raise ValueError("F must be provided")
        if not callable(F):
            raise TypeError("F must be callable")
        if xN is None:
            raise ValueError("xN (right boundary point) must be provided")
        
        try:
            xN = float(xN)
        except (ValueError, TypeError) as e:
            raise TypeError("xN must be numeric") from e
        
        if np.isnan(xN) or np.isinf(xN):
            raise ValueError("xN is NaN or infinite")
    
        a, b, c = self.BC[1]

        if self.bc_types[1] == BCType.DIRICHLET:
            c = self.BC[1][2]
            return lambda w, h: a*w[-1] + c
        else:
            # For Neumann or Robin BC, need to ensure b != 0
            if abs(b) < ZERO_TOLERANCE:
                raise ValueError("Division by zero: b coefficient is zero for non-Dirichlet right BC")
            
            def res(w, h):
                # Validate inputs
                if w is None or h is None:
                    raise ValueError("w and h must be provided")
                if len(w) < 2:
                    raise ValueError("w must have at least 2 elements for non-Dirichlet BC")
                if abs(h) < ZERO_TOLERANCE:
                    raise ValueError(f"Grid spacing h is too small or zero: {h}")

                yNp1 = w[-1]
                yN = w[-2]
                
                if np.isnan(yNp1) or np.isinf(yNp1) or np.isnan(yN) or np.isinf(yN):
                    raise ValueError("yNp1 or yN is NaN or infinite")

                yp = (-c - a*yNp1) / b
                
                if np.isnan(yp) or np.isinf(yp):
                    raise ValueError("Computed yp is NaN or infinite")

                try:
                    F_val = F(xN, yNp1, yp)
                except Exception as e:
                    raise RuntimeError(f"Error evaluating F at right boundary: {e}") from e
                
                if np.isnan(F_val) or np.isinf(F_val):
                    raise ValueError("F returned NaN or infinite at right boundary")

                result = (
                    2*(1 + h*a/b)*yNp1
                    - 2*yN
                    + h**2 * F_val
                    + 2*h *c/b
                )
                
                if np.isnan(result) or np.isinf(result):
                    raise ValueError("Computed residual is NaN or infinite")

                return result

            return res