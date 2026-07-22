import numpy as np
from .utils import compute_empirical_bit_prob

class RPKClustOptimizer:
    """
    Handles the Two-Stage Bayesian Inference logic for Keyword selection.
    """
    def __init__(self):
        pass

    def compute_stage1_prior(self, values):
        """
        Calculates Stage 1 Prior Probability (p_f).
        Paper Note: Replaces missing Netplier constraints with a distributional heuristic.
        Keywords usually have a bounded distinct value count.
        """
        valid_vals = [v for v in values if v is not None]
        if not valid_vals:
            return 1e-5
            
        unique_count = len(set(valid_vals))
        total_count = len(valid_vals)
        ratio = unique_count / total_count
        
        # Heuristic: True keywords are not constants (ratio > 0) 
        # and not completely random payloads (ratio < 0.5)
        if unique_count < 2:
            return 0.01  # Too constant to be a clustering keyword
        elif ratio > 0.5:
            return 0.05  # Too random/high-entropy
        else:
            # Good candidate range (e.g., OpCodes, Message Types)
            return 0.85

    def compute_p_bit(self, values):
        """
        Calculates Bit-use Likelihood (p_bit) based on Eq 7, 8, 10.
        """
        valid_vals = [int.from_bytes(v, 'big') for v in values if v is not None]
        if not valid_vals: return 1e-5
        
        max_val = max(valid_vals)
        if max_val == 0: return 1e-5
        
        # MSB is the position of the highest set bit (0-indexed).
        # Use bit_length() — np.log2 fails on arbitrary-precision Python ints
        # from int.from_bytes (e.g. long NFOR TLV payloads).
        msb = max_val.bit_length() - 1
        
        # Compute empirical Q(k)
        q_k = compute_empirical_bit_prob(values, msb)
        
        # Compute theoretical P(k) = 1 - 1 / 2^(MSB+1-k)
        p_k = np.zeros(msb + 1)
        for k in range(msb + 1):
            p_k[k] = 1 - (1.0 / (2**(msb + 1 - k)))
            
        # Euclidean distance Eq 8
        d = np.sqrt(np.sum((q_k - p_k)**2))
        
        # D_max normalization Eq 10 (Max possible distance in MSB+1 dimensional hypercube)
        d_max = np.sqrt(msb + 1)
        
        p_bit = 1 - (d / (d_max + 1e-9))
        return max(p_bit, 1e-5) # Ensure strictly positive

    def compute_p_offset(self, cand_type, offset):
        """
        Calculates Position Likelihood (p_offset) based on Eq 11.
        """
        if cand_type == 'FOR':
            return max(0.95 - 0.01 * offset, 0.7)
        else:
            return 0.6  # NFOR constant

    def bayesian_update(self, p_bit, p_offset, p_f):
        """
        Computes final probability P(K=1) via Bayesian Update (Eq 15).
        """
        M = p_bit * p_offset * p_f
        N = (1 - p_bit) * (1 - p_offset) * (1 - p_f)
        
        # Add epsilon for numerical stability
        return M / (M + N + 1e-9)