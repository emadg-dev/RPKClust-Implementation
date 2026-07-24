"""
RPKClust Bayesian Inference Engine (optimizer.py)
Fully compliant with RPKClust two-stage Bayesian probability formulations.
"""

import numpy as np


def _val_to_int(v):
    """Safely converts candidate field values to integer representation."""
    if isinstance(v, int):
        return v
    if isinstance(v, (bytes, bytearray)):
        return int.from_bytes(v, 'big')
    if isinstance(v, tuple):
        b_list = []
        for item in v:
            if isinstance(item, (bytes, bytearray)):
                b_list.append(item)
            elif isinstance(item, int):
                b_list.append(bytes([item]))
        return int.from_bytes(b''.join(b_list), 'big')
    if isinstance(v, str):
        return int.from_bytes(v.encode('utf-8'), 'big')
    return int(v)


class RPKClustOptimizer:
    """
    Two-Stage Bayesian Inference Model for Protocol Keyword Identification.
    """

    def compute_stage1_prior(self, values, total_msgs=None):
        """
        Calculates Stage-1 Prior P(f) = N_f / N.
        
        N_f: Number of messages containing candidate field f.
        N: Total number of messages in trace X.
        """
        if total_msgs is None:
            total_msgs = len(values)

        if total_msgs == 0:
            return 1e-6

        valid_vals = [v for v in values if v is not None]
        n_f = len(valid_vals)

        p_f = n_f / total_msgs
        return float(np.clip(p_f, 1e-6, 1.0 - 1e-6))

    def compute_p_bit(self, values):
        """
        Calculates Bit-Use Constraint Probability P_bit.
        Uses Euclidean distance between empirical bit probabilities Q(k)
        and uniform expectation baseline P(k) = 0.5.
        
        Note: If cardinality <= 1 (constant field), P_bit = 1e-6 as constant
        fields carry zero information entropy and are header constants, not keywords.
        """
        valid_vals = [_val_to_int(v) for v in values if v is not None]
        if not valid_vals:
            return 1e-6

        # Constant field filter: single unique value holds zero keyword information
        if len(set(valid_vals)) <= 1:
            return 1e-6

        max_val = max(valid_vals)
        num_bits = max_val.bit_length() if max_val > 0 else 8

        # Calculate empirical bit probability Q(k) across all valid values
        q_k = np.zeros(num_bits)
        for val in valid_vals:
            for k in range(num_bits):
                if (val >> k) & 1:
                    q_k[k] += 1
        q_k /= len(valid_vals)

        # Baseline expectation P(k) = 0.5 (maximum entropy baseline)
        p_k = np.full(num_bits, 0.5)

        # Euclidean distance D = sqrt(sum((Q(k) - P(k))^2))
        euclidean_dist = np.sqrt(np.sum((q_k - p_k) ** 2))
        max_dist = np.sqrt(num_bits * (0.5 ** 2))

        if max_dist == 0:
            p_bit = 1e-6
        else:
            p_bit = euclidean_dist / max_dist

        return float(np.clip(p_bit, 1e-6, 1.0 - 1e-6))

    def compute_p_offset(self, candidate_type, offset=0, boundary_B=0):
        """
        Calculates Position Prior P_offset using monotonic distance decay.
        """
        if candidate_type == 'FOR':
            rel_offset = max(0, offset)
        else:
            rel_offset = max(0, offset - boundary_B)

        p_offset = 1.0 / (1.0 + rel_offset)
        return float(np.clip(p_offset, 1e-6, 1.0 - 1e-6))

    def bayesian_update(self, p_bit, p_offset, p_f):
        """
        Computes Bayesian Posterior Probability P(K=1 | D).
        
        Likelihood P(D | K=1) = P_bit * P_offset
        Prior P(K=1) = P_f
        """
        likelihood = p_bit * p_offset
        prior = p_f

        numerator = likelihood * prior
        denominator = numerator + ((1.0 - likelihood) * (1.0 - prior))

        if denominator <= 0:
            return 1e-6
        return float(np.clip(numerator / denominator, 1e-6, 1.0 - 1e-6))