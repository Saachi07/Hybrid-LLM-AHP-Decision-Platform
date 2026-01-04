import numpy as np
from fractions import Fraction

# Saaty's Random Index
RI_DICT = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

def parse_fraction(val):
    """Parses strings like '1/3' or '5' into floats."""
    try:
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if '/' in val:
            num, den = val.split('/')
            return float(num) / float(den)
        return float(val)
    except:
        return 1.0  # Fallback

def calculate_ahp(matrix):
    """
    Computes Weights, Lambda_max, CI, and CR using the Eigenvalue method.
    """
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(np.real(eigvals))
    lambda_max = np.real(eigvals[max_index])
    
    # Eigenvector (Weights)
    principal_eigvec = np.real(eigvecs[:, max_index])
    weights = principal_eigvec / np.sum(principal_eigvec)
    
    # CI and CR
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_DICT.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    
    return weights, lambda_max, CI, CR

def compress_scale_1_to_5(val_str):
    """
    Implements the logic from 'scale_change(1-5).py'.
    Maps Saaty 1-9 scale to a compressed 1-5 scale to improve consistency.
    """
    val_str = str(val_str).strip()
    
    # Handle Fractions (Reciprocals)
    if '/' in val_str:
        num, den = val_str.split('/')
        den = int(den.strip())
        if den in [2, 3]: return 1/2
        elif den in [4, 5]: return 1/3
        elif den in [6, 7]: return 1/4
        elif den in [8, 9]: return 1/5
        else: return float(val_str) # Fallback
        
    # Handle Integers
    else:
        try:
            num = int(float(val_str))
            if num == 1: return 1.0
            elif num in [2, 3]: return 2.0
            elif num in [4, 5]: return 3.0
            elif num in [6, 7]: return 4.0
            elif num in [8, 9]: return 5.0
            else: return float(num)
        except:
            return 1.0

def apply_scale_compression(matrix):
    """Applies the 1-5 compression to a full matrix."""
    n = matrix.shape[0]
    new_matrix = np.ones((n, n))
    
    # We reconstruct based on the upper triangle to maintain reciprocity
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i, j]
            # Convert float back to rough integer representation for the mapping logic
            # (In a real scenario, we'd map the raw input string, but here we approximate)
            if val >= 1:
                mapped_val = compress_scale_1_to_5(str(int(round(val))))
            else:
                denom = int(round(1/val))
                mapped_val = compress_scale_1_to_5(f"1/{denom}")
            
            new_matrix[i, j] = mapped_val
            new_matrix[j, i] = 1.0 / mapped_val
            
    return new_matrix