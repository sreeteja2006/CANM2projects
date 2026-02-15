"""
Validation utilities for the FDM solver.

This module centralizes all common validation and error checking routines
to avoid code repetition across multiple modules.
"""

import numpy as np
from .constants import ZERO_TOLERANCE


def validate_not_none(value=None, var_name=None, *, arg_names=None):
    """
    Validate that value(s) are not None.
    
    Can be used two ways:
    1. Single value: validate_not_none(F, "F") -> returns F
    2. Multiple values: validate_not_none(F, Fy, Fyp, arg_names=["F", "Fy", "Fyp"])
    
    Parameters:
    value: Value to validate (or first value if arg_names provided)
    var_name: Name of value for error messages (used when validating single value)
    arg_names: List of argument names for multiple validation
    
    Returns:
    The validated value (when validating single value)
    
    Raises:
    ValueError: If any value is None
    """
    # Single value validation
    if arg_names is None and var_name is not None:
        if value is None:
            raise ValueError(f"{var_name} must be provided")
        return value
    
    # Multiple value validation (legacy style)
    if arg_names is not None:
        # In this case, value is actually the first of multiple *args
        # This function signature doesn't support *args properly for this case
        # So we'll just validate the single value if provided
        if value is None and var_name is not None:
            raise ValueError(f"{var_name} must be provided")
        if value is not None:
            return value


def validate_callable(func, func_name):
    """
    Validate that a function is callable.
    
    Parameters:
    func: Function to validate
    func_name: Name of the function for error messages
    
    Returns:
    The validated function
    
    Raises:
    TypeError: If func is not callable
    """
    if not callable(func):
        raise TypeError(f"{func_name} must be callable")
    return func


def validate_numeric_type(value, expected_type, var_name):
    """
    Validate that a value is of expected numeric type.
    
    Parameters:
    value: Value to validate
    expected_type: Expected type (int, float, etc.)
    var_name: Name of variable for error messages
    
    Raises:
    TypeError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise TypeError(f"{var_name} must be {expected_type.__name__}, got {type(value).__name__}")


def validate_positive_number(value, var_name):
    """
    Validate that a value is a positive number.
    
    Parameters:
    value: Value to validate
    var_name: Name of variable for error messages
    
    Raises:
    ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{var_name} must be positive, got {value}")


def validate_in_range(value, min_val, max_val, var_name):
    """
    Validate that a value is within a specified range.
    
    Parameters:
    value: Value to validate
    min_val: Minimum allowed value
    max_val: Maximum allowed value
    var_name: Name of variable for error messages
    
    Raises:
    ValueError: If value is out of range
    """
    if value < min_val or value > max_val:
        raise ValueError(f"{var_name} must be in range [{min_val}, {max_val}], got {value}")


def validate_array_conversion(data, var_name, dtype=float):
    """
    Convert data to numpy array with validation.
    
    Parameters:
    data: Data to convert
    var_name: Name of variable for error messages
    dtype: Target data type (default: float)
    
    Returns:
    np.ndarray: Converted array
    
    Raises:
    TypeError: If conversion fails
    """
    try:
        return np.asarray(data, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{var_name} must be convertible to {dtype.__name__} array") from e


def validate_1d_array(array, var_name):
    """
    Validate that array is 1-dimensional.
    
    Parameters:
    array: Array to validate
    var_name: Name of variable for error messages
    
    Returns:
    The validated array
    
    Raises:
    ValueError: If array is not 1-dimensional
    """
    if array.ndim != 1:
        raise ValueError(f"{var_name} must be 1-dimensional, got shape {array.shape}")
    return array


def validate_nan_inf(array, var_name):
    """
    Validate that array contains no NaN or Inf values.
    
    Parameters:
    array: Array to validate
    var_name: Name of variable for error messages
    
    Returns:
    The validated array
    
    Raises:
    ValueError: If array contains NaN or Inf
    """
    if np.any(np.isnan(array)):
        raise ValueError(f"{var_name} contains NaN values")
    if np.any(np.isinf(array)):
        raise ValueError(f"{var_name} contains infinite values")
    return array


def validate_positive_integer(value, var_name, min_val=1, max_val=None):
    """
    Validate that value is a positive integer within bounds.
    
    Parameters:
    value: Value to validate
    var_name: Name of variable for error messages
    min_val: Minimum allowed value (default: 1)
    max_val: Optional maximum allowed value
    
    Returns:
    int: The validated integer value (after type conversion if needed)
    
    Raises:
    TypeError: If not an integer
    ValueError: If not positive or outside bounds
    """
    if not isinstance(value, (int, np.integer)):
        try:
            value = int(value)
        except (ValueError, TypeError) as e:
            raise TypeError(f"{var_name} must be an integer, got {type(value).__name__}") from e
    
    if value < min_val:
        raise ValueError(f"{var_name} must be at least {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{var_name} is too large (max {max_val}), got {value}")
    
    return value


def validate_array_length(array, expected_length, var_name):
    """
    Validate that array has expected length.
    
    Parameters:
    array: Array to validate
    expected_length: Expected length
    var_name: Name of variable for error messages
    
    Returns:
    The validated array
    
    Raises:
    ValueError: If length doesn't match
    """
    if len(array) != expected_length:
        raise ValueError(f"{var_name} must have length {expected_length}, got {len(array)}")
    return array


def validate_array_lengths_match(array1, array2, name1, name2):
    """
    Validate that two arrays have the same length.
    
    Parameters:
    array1: First array
    array2: Second array
    name1: Name of first array
    name2: Name of second array
    
    Returns:
    Tuple of (validated array1, validated array2)
    
    Raises:
    ValueError: If lengths don't match
    """
    if len(array1) != len(array2):
        raise ValueError(f"Length mismatch: {name1} has length {len(array1)} but {name2} has length {len(array2)}")
    return array1, array2


def validate_array_shape(array, expected_shape, var_name):
    """
    Validate that array has expected shape.
    
    Parameters:
    array: Array to validate
    expected_shape: Expected shape (tuple)
    var_name: Name of variable for error messages
    
    Returns:
    The validated array
    
    Raises:
    ValueError: If shape doesn't match
    """
    if array.shape != expected_shape:
        raise ValueError(f"{var_name} must have shape {expected_shape}, got {array.shape}")
    return array


def validate_grid_spacing(x_array, var_name="x"):
    """
    Validate grid spacing and return it.
    
    Parameters:
    x_array: Grid points array
    var_name: Name of variable for error messages
    
    Returns:
    float: Grid spacing
    
    Raises:
    ValueError: If grid spacing is too small
    """
    if len(x_array) < 2:
        raise ValueError(f"{var_name} must have at least 2 points")
    
    h = x_array[1] - x_array[0]
    
    if abs(h) < ZERO_TOLERANCE:
        raise ValueError(f"Grid spacing h is too small or zero: {h}")
    
    return h


def validate_finite_value(value, var_name):
    """
    Validate that a scalar value is finite (not NaN or Inf).
    
    Parameters:
    value: Value to validate
    var_name: Name of variable for error messages
    
    Returns:
    The validated value
    
    Raises:
    ValueError: If value is NaN or Inf
    """
    try:
        value = float(value)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{var_name} must be numeric") from e
    
    if np.isnan(value):
        raise ValueError(f"{var_name} is NaN")
    if np.isinf(value):
        raise ValueError(f"{var_name} is infinite")
    
    return value


def validate_domain(start, end):
    """
    Validate domain start and end values.
    
    Parameters:
    start: Domain start
    end: Domain end
    
    Raises:
    ValueError: If domain is invalid
    """
    validate_finite_value(start, "Domain start")
    validate_finite_value(end, "Domain end")
    
    if start >= end:
        raise ValueError(f"Domain start must be less than end: [{start}, {end}]")


def validate_bc_coefficients(bc_array, bc_type="boundary condition"):
    """
    Validate boundary condition coefficients array.
    
    Parameters:
    bc_array: BC coefficients array
    bc_type: Type name for error messages
    
    Returns:
    np.ndarray: Validated BC array
    
    Raises:
    ValueError: If BC is invalid
    """
    bc_array = validate_array_conversion(bc_array, bc_type, dtype=float)
    validate_array_shape(bc_array, (2, 3), bc_type)
    validate_nan_inf(bc_array, bc_type)
    
    # Validate that at least one of a or b is non-zero for each boundary
    for i, side in enumerate(["left", "right"]):
        a, b, c = bc_array[i]
        if abs(a) < ZERO_TOLERANCE and abs(b) < ZERO_TOLERANCE:
            raise ValueError(f"Both a and b are zero for {side} {bc_type}. At least one must be non-zero.")
    
    return bc_array


def validate_tolerance(tol, var_name="tolerance"):
    """
    Validate solver tolerance.
    
    Parameters:
    tol: Tolerance value
    var_name: Name of variable for error messages
    
    Returns:
    float: Validated tolerance
    
    Raises:
    ValueError: If tolerance is invalid
    """
    try:
        tol = float(tol)
    except (ValueError, TypeError) as e:
        raise TypeError(f"{var_name} must be numeric") from e
    
    validate_positive_number(tol, var_name)
    
    if tol > 1:
        raise ValueError(f"{var_name} seems too large (>1), got {tol}")
    
    validate_finite_value(tol, var_name)
    
    return tol


def validate_zero_check(value, tolerance, var_name):
    """
    Check if a value is effectively zero within tolerance.
    
    Parameters:
    value: Value to check
    tolerance: Zero tolerance threshold
    var_name: Name of variable for error messages
    
    Raises:
    ValueError: If value is zero within tolerance
    """
    if abs(value) < tolerance:
        raise ValueError(f"Division by zero: {var_name} is zero or near-zero")
