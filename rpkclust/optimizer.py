"""
RPKClust Bayesian Inference Engine (optimizer.py)
Fully compliant with RPKClust two-stage Bayesian probability formulations.
"""

import numpy as np
from .constraints import ClusteringConstraints

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

    def compute_stage1_probability(self, candidate, X):

        values = candidate["values"]

        labels = self.cluster_by_candidate(values)


        c1 = ClusteringConstraints.message_similarity(
            labels,
            X
        )

        c2 = ClusteringConstraints.remote_coupling(
            labels,
            X
        )

        c3 = ClusteringConstraints.structural_consistency(
            labels,
            candidate
        )

        c4 = ClusteringConstraints.dimensional_constraint(
            labels
        )


        return self.constraint_bayesian_update(
            c1,
            c2,
            c3,
            c4
        )

    def cluster_by_candidate(self, values):

        mapping = {}
        labels = []

        for v in values:

            if isinstance(v, bytes):
                key = v
            else:
                key = str(v)

            if key not in mapping:
                mapping[key] = len(mapping)

            labels.append(mapping[key])

        return np.array(labels)

    def constraint_bayesian_update(
        self,
        c1,
        c2,
        c3,
        c4
    ):
        constraints = np.array(
            [
                c1,
                c2,
                c3,
                c4
            ],
            dtype=float
        )


        # Remove invalid values
        constraints = np.clip(
            constraints,
            1e-6,
            1 - 1e-6
        )


        # Prior probability of a random candidate being keyword
        #
        # Usually keyword fields are rare among all candidates.
        #
        # A small prior avoids every candidate becoming keyword.
        prior = 0.1


        # Likelihood if candidate is a real keyword
        likelihood_keyword = np.prod(
            constraints
        )


        # Likelihood if candidate is not keyword
        likelihood_not_keyword = np.prod(
            1 - constraints
        )


        numerator = (
            likelihood_keyword *
            prior
        )


        denominator = (
            numerator +
            likelihood_not_keyword *
            (1 - prior)
        )


        if denominator == 0:
            return 1e-6


        posterior = (
            numerator /
            denominator
        )


        return float(
            np.clip(
                posterior,
                1e-6,
                1 - 1e-6
            )
        )

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

    def compute_p_offset(self, candidate_type, offset=0, boundary_B=0, alpha=0.15):
        """
        Calculates Position Prior P_offset using monotonic distance decay.
        """
        if candidate_type == 'FOR':
            rel_offset = max(0, offset)
        else:
            rel_offset = max(0, offset - boundary_B)

        p_offset = np.exp(-alpha * rel_offset)
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


