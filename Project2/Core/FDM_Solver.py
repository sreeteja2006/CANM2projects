import numpy as np
from typing import Callable
import warnings

from .constants import (ZERO_TOLERANCE, DERIVATIVE_EPSILON, MIN_DERIVATIVE_EPSILON,
                        MAX_GRID_SIZE, MAX_ITERATIONS, DERIVATIVE_VALIDATION_REL_TOL,
                        DERIVATIVE_VALIDATION_ABS_TOL)
from .validation import (validate_not_none, validate_callable, validate_positive_integer,
                          validate_array_conversion, validate_1d_array, validate_nan_inf,
                          validate_array_length, validate_tolerance, validate_finite_value,
                          validate_bc_coefficients, validate_in_range)
from .Jacobian import Jacobian
from .Boundary_Conditions import Boundary_Conditions
from .TDMA import TDMA
from .Function_Generator import Function_Generator
from .Residual import Residual

class FDM_Solver:
    def __init__(self, F, Fy, Fyp, N, domain, BC, w0=None):
        """
        Initialize Finite Difference Method Solver.

        Args:
            F (callable): y'' = F(x,y,y') in FDM
            Fy (callable): partial derivative of F wrt y (None for auto-approximation)
            Fyp (callable): partial derivative of F wrt y' (None for auto-approximation)
            N (int): Number of grid intervals
            domain (Tuple): Domain (start, end)
            BC (Array): Array of BC in a 2*3 matrix of the form ay + by' + c = 0
            w0 (Vector, optional): Initial guess vector. Defaults to None.
        """
        # Validate F
        F = validate_not_none(F, "F")
        F = validate_callable(F, "F")
        
        # Validate Fy and Fyp (can be None or callable)
        if Fy is not None:
            Fy = validate_callable(Fy, "Fy")
        if Fyp is not None:
            Fyp = validate_callable(Fyp, "Fyp")
        
        # Validate N
        N = validate_positive_integer(N, "N", min_val=2, max_val=MAX_GRID_SIZE)
        
        # Validate domain
        domain = validate_not_none(domain, "domain")
        xstart, xend = validate_finite_value(domain[0], "domain_start"), validate_finite_value(domain[1], "domain_end")
        
        if xstart >= xend:
            raise ValueError(f"domain start must be less than end: [{xstart}, {xend}]")
        
        # Validate BC
        BC = validate_bc_coefficients(BC)
        
        # Store validated parameters
        self.F = F
        self.N = N
        self.xstart = xstart
        self.xend = xend
        self.h = (self.xend - self.xstart) / N
        self.BC = BC
        
        # Validate and set partial derivatives with validation
        self._user_provided_Fy = Fy
        self._user_provided_Fyp = Fyp
        
        if Fy is None:
            print("Note: Fy not provided, using numerical approximation.")
            self.Fy = self._approx_Fy
        else:
            # Validate user-provided Fy
            is_valid, message = self._validate_partial_derivative(Fy, 'Fy')
            if is_valid:
                self.Fy = Fy
            else:
                warnings.warn(f"User-provided Fy validation failed: {message}\\nSwitching to numerical approximation.")
                print(f"WARNING: User-provided Fy appears incorrect: {message}")
                print("Switching to numerical approximation for Fy.")
                self.Fy = self._approx_Fy

        if Fyp is None:
            print("Note: Fyp not provided, using numerical approximation.")
            self.Fyp = self._approx_Fyp
        else:
            # Validate user-provided Fyp
            is_valid, message = self._validate_partial_derivative(Fyp, 'Fyp')
            if is_valid:
                self.Fyp = Fyp
            else:
                warnings.warn(f"User-provided Fyp validation failed: {message}\\nSwitching to numerical approximation.")
                print(f"WARNING: User-provided Fyp appears incorrect: {message}")
                print("Switching to numerical approximation for Fyp.")
                self.Fyp = self._approx_Fyp
        
        # Initialize boundary conditions
        self.Boundary_Conditions = Boundary_Conditions(BC)
        self.left_bc_jac = self.Boundary_Conditions.build_left_bc_jac(self.Fy, self.Fyp, self.xstart)
        self.right_bc_jac = self.Boundary_Conditions.build_right_bc_jac(self.Fy, self.Fyp, self.xend)
        self.left_bc_res = self.Boundary_Conditions.build_left_bc_res(self.F, self.xstart)
        self.right_bc_res = self.Boundary_Conditions.build_right_bc_res(self.F, self.xend)
        
        # Initialize function generator
        self.Function_Generator = Function_Generator(F, self.Fy, self.Fyp)
        self.fu, self.fl, self.fd = self.Function_Generator.build_tridiagonal_terms()
        self.Residual_F = self.Function_Generator.build_residual_function()

        # Set initial guess
        if w0 is None:
            self.w0 = np.linspace(self.xstart, self.xend, self.N + 1)
        else:
            w0 = validate_array_conversion(w0, "w0")
            w0 = validate_1d_array(w0, "w0")
            w0 = validate_array_length(w0, self.N + 1, "w0")
            w0 = validate_nan_inf(w0, "w0")
            self.w0 = w0
    
    def _validate_partial_derivative(self, user_func, func_name):
        """
        Validate user-provided partial derivative by comparing with numerical approximation.
        
        Parameters:
        user_func: User-provided partial derivative function
        func_name: Name of the function ('Fy' or 'Fyp')
        
        Returns:
        (is_valid, message): Tuple of boolean and string message
        """
        # Test at multiple points
        test_points = [
            (0.5, 1.0, 0.5),
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 1.0),
            (-0.5, -1.0, -0.5),
            (0.1, 0.9, 0.1)
        ]
        
        max_rel_error = 0.0
        max_abs_error = 0.0
        
        try:
            for x, y, yp in test_points:
                # Get user-provided value
                try:
                    user_val = user_func(x, y, yp)
                except Exception as e:
                    return False, f"Function raised exception at test point ({x}, {y}, {yp}): {e}"
                
                # Check for invalid values
                if np.isnan(user_val) or np.isinf(user_val):
                    return False, f"Function returned NaN or Inf at test point ({x}, {y}, {yp})"
                
                # Get numerical approximation
                if func_name == 'Fy':
                    numerical_val = self._approx_Fy(x, y, yp)
                else:  # Fyp
                    numerical_val = self._approx_Fyp(x, y, yp)
                
                # Compute error
                abs_error = abs(user_val - numerical_val)
                
                # Compute relative error (avoiding division by zero)
                if abs(numerical_val) > MIN_DERIVATIVE_EPSILON:
                    rel_error = abs_error / abs(numerical_val)
                else:
                    rel_error = abs_error
                
                max_rel_error = max(max_rel_error, rel_error)
                max_abs_error = max(max_abs_error, abs_error)
            
            # Validation criteria
            # Accept if relative error < threshold OR absolute error < threshold (for cases near zero)
            if max_rel_error < DERIVATIVE_VALIDATION_REL_TOL or max_abs_error < DERIVATIVE_VALIDATION_ABS_TOL:
                return True, "Validation passed"
            else:
                return False, f"Max relative error: {max_rel_error:.2e}, Max absolute error: {max_abs_error:.2e}"
        
        except Exception as e:
            return False, f"Validation error: {e}"

    def _approx_Fy(self, x, y, yp):
        """
        Numerical approximation of partial derivative of F with respect to y.
        
        Parameters:
        x, y, yp: Point at which to evaluate the derivative
        
        Returns:
        Approximate value of ∂F/∂y
        """
        eps = DERIVATIVE_EPSILON * max(1.0, abs(y))
        if eps < MIN_DERIVATIVE_EPSILON:
            eps = DERIVATIVE_EPSILON
        
        try:
            f_plus = self.F(x, y + eps, yp)
            f_minus = self.F(x, y - eps, yp)
        except Exception as e:
            raise RuntimeError(f"Error computing Fy approximation: {e}") from e
        
        result = (f_plus - f_minus) / (2 * eps)
        
        if np.isnan(result) or np.isinf(result):
            raise ValueError(f"Fy approximation returned NaN or Inf at ({x}, {y}, {yp})")
        
        return result

    def _approx_Fyp(self, x, y, yp):
        """
        Numerical approximation of partial derivative of F with respect to y'.
        
        Parameters:
        x, y, yp: Point at which to evaluate the derivative
        
        Returns:
        Approximate value of ∂F/∂y'
        """
        eps = DERIVATIVE_EPSILON * max(1.0, abs(yp))
        if eps < MIN_DERIVATIVE_EPSILON:
            eps = DERIVATIVE_EPSILON
        
        try:
            f_plus = self.F(x, y, yp + eps)
            f_minus = self.F(x, y, yp - eps)
        except Exception as e:
            raise RuntimeError(f"Error computing Fyp approximation: {e}") from e
        
        result = (f_plus - f_minus) / (2 * eps)
        
        if np.isnan(result) or np.isinf(result):
            raise ValueError(f"Fyp approximation returned NaN or Inf at ({x}, {y}, {yp})")
        
        return result


    def Norm_inf(self, vector) -> float:
        """
        Compute infinity norm (maximum absolute value) of a vector.
        
        Parameters:
        vector: Input vector
        
        Returns:
        Infinity norm as float
        """
        vector = validate_not_none(vector, "vector")
        vector = validate_array_conversion(vector, "vector")
        vector = validate_1d_array(vector, "vector")
        vector = validate_nan_inf(vector, "vector")
        
        if len(vector) == 0:
            raise ValueError("vector must not be empty")
        
        return float(np.max(np.abs(vector)))

    def solver(self, tol=1e-6, max_iter=100):
        """
        Solve the nonlinear BVP using Newton's method with FDM.
        
        Parameters:
        tol: Convergence tolerance (default: 1e-6)
        max_iter: Maximum number of iterations (default: 100)
        
        Returns:
        w: Solution vector
        """
        # Validate parameters
        tol = validate_tolerance(tol, "tol")
        max_iter = validate_positive_integer(max_iter, "max_iter", max_val=MAX_ITERATIONS)
        
        # Initialize
        x = np.linspace(self.xstart, self.xend, self.N + 1)
        w = self.w0.copy()

        residual = Residual(
            self.Residual_F,
            self.N,
            self.left_bc_res,
            self.right_bc_res
        )

        jacobian = Jacobian(
            fu=self.fu,
            fl=self.fl,
            fd=self.fd,
            bc_left_jac=self.left_bc_jac,
            bc_right_jac=self.right_bc_jac
        )

        converged = False
        
        for k in range(max_iter):
            try:
                u, l, d = jacobian.build(w, x)
            except Exception as e:
                raise RuntimeError(f"Error building Jacobian at iteration {k+1}: {e}") from e
            
            try:
                b = residual.build(w, x)
            except Exception as e:
                raise RuntimeError(f"Error building residual at iteration {k+1}: {e}") from e

            try:
                delta = TDMA(u, l, d, -b)
            except Exception as e:
                raise RuntimeError(f"Error solving linear system at iteration {k+1}: {e}") from e
            
            # Validate delta
            delta = validate_nan_inf(delta, f"delta_iter_{k+1}")
            
            w += delta
            
            # Validate w
            w = validate_nan_inf(w, f"w_iter_{k+1}")

            try:
                norm_delta = self.Norm_inf(delta)
            except Exception as e:
                raise RuntimeError(f"Error computing norm at iteration {k+1}: {e}") from e
            
            if np.isinf(norm_delta):
                raise RuntimeError(f"Solution diverged at iteration {k+1} (infinite norm)")
            
            if norm_delta < tol:
                print(f"Converged in {k+1} iterations (residual norm: {norm_delta:.2e})")
                converged = True
                break
        
        if not converged:
            warnings.warn(f"Maximum iterations ({max_iter}) reached without convergence. "
                        f"Final residual norm: {norm_delta:.2e}")
            print(f"WARNING: Did not converge in {max_iter} iterations. "
                  f"Final residual norm: {norm_delta:.2e}")

        return w

    
