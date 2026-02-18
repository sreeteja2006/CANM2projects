"""
Constants for the FDM solver.

This module defines numerical tolerance values used throughout the solver
to ensure consistency and maintainability.
"""

# Tolerance for checking if a value is effectively zero
# Used for division by zero protection, singular matrix detection, etc.
ZERO_TOLERANCE = 1e-14

# Tolerance for numerical derivative approximation
# Used as the base epsilon for finite difference approximations
DERIVATIVE_EPSILON = 1e-8

# Minimum epsilon for derivative approximation
# Used when the adaptive epsilon would be too small
MIN_DERIVATIVE_EPSILON = 1e-10

# Tolerance for grid uniformity checking
# Used to validate that grid spacing is uniform
GRID_UNIFORMITY_TOLERANCE = 1e-10

# Maximum allowed values for parameters
MAX_GRID_SIZE = 1000000
MAX_ITERATIONS = 1000000

# Validation tolerance for partial derivative checking
# Used to validate user-provided derivatives against numerical approximations
DERIVATIVE_VALIDATION_REL_TOL = 0.1  # 10% relative error
DERIVATIVE_VALIDATION_ABS_TOL = 1e-6  # Absolute error for near-zero values
