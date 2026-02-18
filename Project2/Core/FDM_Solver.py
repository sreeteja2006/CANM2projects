import numpy as np
from typing import Callable
import warnings

from .constants import *
from .validation import *
from .Jacobian import Jacobian
from .Boundary_Conditions import Boundary_Conditions
from .TDMA import TDMA
from .Function_Generator import Function_Generator
from .Residual import Residual

class FDM_Solver:
    def __init__(self, N, domain, F=None, Fy=None, Fyp=None, BC=None, w0=None, jacobian=None, residual=None):
        """
        Initialize Finite Difference Method Solver.

        Args:
            N (int): Number of grid intervals (required)
            domain (Tuple): Domain (start, end) (required)
            F (callable, optional): y'' = F(x,y,y') in FDM. Required if jacobian or residual not provided.
            Fy (callable, optional): partial derivative of F wrt y (None for auto-approximation)
            Fyp (callable, optional): partial derivative of F wrt y' (None for auto-approximation)
            BC (Array, optional): Array of BC in a 2*3 matrix of the form ay + by' + c = 0
            w0 (Vector, optional): Initial guess vector. Defaults to None.
            jacobian (Jacobian, optional): Custom Jacobian object. Defaults to None.
            residual (Residual, optional): Custom Residual object. Defaults to None.
        """
        # Validate required parameter combinations
        if jacobian is not None and residual is not None:
            # Custom jacobian and residual provided - F is optional
            if F is None:
                print("Note: Using custom Jacobian and Residual. F not required.\n")
        else:
            # Standard mode - F is required
            if F is None:
                raise ValueError("F must be provided unless both jacobian and residual are specified.")
        
        # Validate F if provided
        if F is not None:
            F = validate_callable(F, "F")
        
        # Validate Fy and Fyp (can be None or callable)
        if Fy is not None:
            Fy = validate_callable(Fy, "Fy")
        if Fyp is not None:
            Fyp = validate_callable(Fyp, "Fyp")
        
        # Validate N
        N = validate_positive_integer(N, "N", min_val=2, max_val=MAX_GRID_SIZE)
        
        # Validate domain
        xstart, xend = validate_finite_value(domain[0], "domain_start"), validate_finite_value(domain[1], "domain_end")
        
        if xstart >= xend:
            raise ValueError(f"domain start must be less than end: [{xstart}, {xend}]")
        
        # Validate BC if provided 
        if BC is not None:
            BC = validate_bc_coefficients(BC)
        
        # Store validated parameters
        self.F = F
        self.N = N
        self.xstart = xstart
        self.xend = xend
        self.h = (self.xend - self.xstart) / N
        self.BC = BC
        
        if F is not None:
            # Standard mode: handle partial derivatives
            if Fy is None:
                print("Note: Fy not provided, using numerical approximation.\n")
                self.Fy = self._approx_Fy
            else:
                # Validate user-provided Fy
                is_valid, message = self._validate_partial_derivative(Fy, 'Fy')
                if is_valid:
                    self.Fy = Fy
                else:
                    warnings.warn(f"User-provided Fy validation failed: {message}\\nSwitching to numerical approximation.\n")
                    self.Fy = self._approx_Fy

            if Fyp is None:
                print("Note: Fyp not provided, using numerical approximation.\n")
                self.Fyp = self._approx_Fyp
            else:
                # Validate user-provided Fyp
                is_valid, message = self._validate_partial_derivative(Fyp, 'Fyp')
                if is_valid:
                    self.Fyp = Fyp
                else:
                    warnings.warn(f"User-provided Fyp validation failed: {message}\\nSwitching to numerical approximation.")
                    self.Fyp = self._approx_Fyp
            
            # Initialize boundary conditions
            if BC is None:
                raise ValueError("BC must be provided when F is specified.")
            self.Boundary_Conditions = Boundary_Conditions(BC)
            self.left_bc_jac = self.Boundary_Conditions.build_left_bc_jac(self.Fy, self.Fyp, self.xstart)
            self.right_bc_jac = self.Boundary_Conditions.build_right_bc_jac(self.Fy, self.Fyp, self.xend)
            self.left_bc_res = self.Boundary_Conditions.build_left_bc_res(self.F, self.xstart)
            self.right_bc_res = self.Boundary_Conditions.build_right_bc_res(self.F, self.xend)
            
            # Initialize function generator
            self.Function_Generator = Function_Generator(F, self.Fy, self.Fyp)
            self.fu, self.fl, self.fd = self.Function_Generator.build_tridiagonal_terms()
            self.Residual_F = self.Function_Generator.build_residual_function()
        else:
            #F not provided, set placeholders
            self.Fy = None
            self.Fyp = None
            self.Boundary_Conditions = None
            self.left_bc_jac = None
            self.right_bc_jac = None
            self.left_bc_res = None
            self.right_bc_res = None
            self.Function_Generator = None
            self.fu = None
            self.fl = None
            self.fd = None
            self.Residual_F = None

        # Set initial guess
        if w0 is None:
            self.w0 = np.linspace(self.xstart, self.xend, self.N + 1)
        else:
            w0 = validate_array_conversion(w0, "w0")
            w0 = validate_1d_array(w0, "w0")
            w0 = validate_array_length(w0, self.N + 1, "w0")
            w0 = validate_nan_inf(w0, "w0")
            self.w0 = w0
        
        # Store custom jacobian and residual or initialize default ones
        if jacobian is not None:
            self.jacobian = jacobian
        else:
            if F is None:
                raise ValueError("Jacobian must be provided when F is not specified.")
            self.jacobian = Jacobian(
                fu=self.fu,
                fl=self.fl,
                fd=self.fd,
                bc_left_jac=self.left_bc_jac,
                bc_right_jac=self.right_bc_jac
            )
        
        if residual is not None:
            self.residual = residual
        else:
            if F is None:
                raise ValueError("Residual must be provided when F is not specified.")
            self.residual = Residual(
                self.Residual_F,
                self.N,
                self.left_bc_res,
                self.right_bc_res
            )
    
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
        if self.F is None:
            raise RuntimeError("Cannot compute Fy approximation: F is not defined. "
                             "Provide F or use custom Jacobian.")
        
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
        if self.F is None:
            raise RuntimeError("Cannot compute Fyp approximation: F is not defined. "
                             "Provide F or use custom Jacobian.")
        
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

        converged = False
        
        for k in range(max_iter):
            try:
                u, l, d = self.jacobian.build(w, x)
            except Exception as e:
                raise RuntimeError(f"Error building Jacobian at iteration {k+1}: {e}") from e
            
            try:
                b = self.residual.build(w, x)
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
                print(f"Converged in {k+1} iterations (residual norm: {norm_delta:.2e})\n")
                converged = True
                break
        
        if not converged:
            warnings.warn(f"Maximum iterations ({max_iter}) reached without convergence. "
                        f"Final residual norm: {norm_delta:.2e}")
            print(f"WARNING: Did not converge in {max_iter} iterations. "
                  f"Final residual norm: {norm_delta:.2e}\n")

        return w
    

    def set_N(self, new_N):
        """
        Update the number of grid intervals and recompute related parameters.
        
        Parameters:
        new_N: New number of grid intervals
        """
        new_N = validate_positive_integer(new_N, "new_N", min_val=2, max_val=MAX_GRID_SIZE)
        self.N = new_N
        self.h = (self.xend - self.xstart) / self.N
        
        # Update initial guess to match new grid size
        self.w0 = np.linspace(self.xstart, self.xend, self.N + 1)
    
