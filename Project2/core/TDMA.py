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
    # Convert to numpy arrays
    u = np.asarray(u, dtype=float)
    l = np.asarray(l, dtype=float)
    d = np.asarray(d, dtype=float)
    b = np.asarray(b, dtype=float)
    
    n = len(d)
    Q = np.zeros(n)
    P = np.zeros(n - 1)
    
    # Forward elimination
    if abs(d[0]) < ZERO_TOLERANCE:
        raise ValueError(f"Matrix is singular: d[0]={d[0]}")
    
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
    
    return x