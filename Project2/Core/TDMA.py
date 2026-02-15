import numpy as np
from .constants import ZERO_TOLERANCE

def TDMA(u, l, d, b):
    """
    Tridiagonal Matrix Algorithm (Thomas Algorithm) solver.
    
    Parameters:
    u: upper diagonal (length n-1)
    l: lower diagonal (length n-1)
    d: main diagonal (length n)
    b: right-hand side vector (length n)
    
    Returns:
    x: solution vector (length n)
    """
    # Input validation
    if u is None or l is None or d is None or b is None:
        raise ValueError("All input parameters (u, l, d, b) must be provided")
    
    # Convert to numpy arrays if needed
    try:
        u = np.asarray(u, dtype=float)
        l = np.asarray(l, dtype=float)
        d = np.asarray(d, dtype=float)
        b = np.asarray(b, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError("All inputs must be convertible to numeric arrays") from e
    
    # Check dimensions
    if d.ndim != 1 or b.ndim != 1:
        raise ValueError("Diagonal d and vector b must be 1-dimensional arrays")
    
    n = len(d)
    
    # Check for minimum size
    if n < 2:
        raise ValueError("Matrix size must be at least 2x2")
    
    # Check array lengths
    if len(b) != n:
        raise ValueError(f"Length mismatch: d has length {n} but b has length {len(b)}")
    
    if len(u) != n - 1:
        raise ValueError(f"Upper diagonal u must have length {n-1}, got {len(u)}")
    
    if len(l) != n - 1:
        raise ValueError(f"Lower diagonal l must have length {n-1}, got {len(l)}")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(u)) or np.any(np.isnan(l)) or np.any(np.isnan(d)) or np.any(np.isnan(b)):
        raise ValueError("Input arrays contain NaN values")
    
    if np.any(np.isinf(u)) or np.any(np.isinf(l)) or np.any(np.isinf(d)) or np.any(np.isinf(b)):
        raise ValueError("Input arrays contain infinite values")
    
    # Initialize arrays
    Q = np.zeros(n)
    P = np.zeros(n - 1)
    
    # Check for zero diagonal at first position
    if abs(d[0]) < ZERO_TOLERANCE:
        raise ValueError("Division by zero: diagonal element d[0] is zero or near-zero")
    
    # Forward elimination
    Q[0] = b[0] / d[0]
    P[0] = u[0] / d[0]
    
    for i in range(1, n - 1):
        denominator = d[i] - l[i - 1] * P[i - 1]
        
        # Check for zero or near-zero denominator (singular matrix)
        if abs(denominator) < ZERO_TOLERANCE:
            raise ValueError(f"Matrix is singular or near-singular at row {i}. "
                           f"Denominator: {denominator}")
        
        P[i] = u[i] / denominator
        Q[i] = (b[i] - l[i - 1] * Q[i - 1]) / denominator
    
    # Last row
    denominator = d[n - 1] - l[n - 2] * P[n - 2]
    if abs(denominator) < ZERO_TOLERANCE:
        raise ValueError(f"Matrix is singular or near-singular at last row. "
                       f"Denominator: {denominator}")
    
    Q[n - 1] = (b[n - 1] - l[n - 2] * Q[n - 2]) / denominator
    
    # Back substitution
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    
    for i in range(n - 2, -1, -1):
        x[i] = Q[i] - P[i] * x[i + 1]
    
    # Check for NaN or Inf in solution
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise RuntimeError("Solution contains NaN or infinite values. Matrix may be ill-conditioned")
    
    return x