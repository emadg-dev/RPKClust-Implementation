import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .constraints import ClusteringConstraints

_MISSING = object()  # sentinel for absent NFOR TLV values


def _val_to_int(v) -> int:
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


# ------------------------------------------------------------------
#  Two-Stage Bayesian Inference Model
# ------------------------------------------------------------------

class RPKClustOptimizer:

    def compute_stage1_probability(
        self,
        candidate: Dict[str, Any],
        X: List[bytes],
        interaction_metadata: Optional[List[Dict[str, Any]]] = None,
        prior: float = 0.1,
    ) -> Tuple[float, Dict[str, float]]:
        
        if X is None:
            raise ValueError("X is not valid")
        values = candidate.get("values")
        if values is None or len(values) != len(X):
            raise ValueError("candidate values not valid")
        if interaction_metadata is not None and len(interaction_metadata) != len(X):
            raise ValueError("interaction_metadata not valid")
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("prior not valid")
        labels = self.cluster_by_candidate(values)

        c1 = ClusteringConstraints.message_similarity(labels, X)
        try:
            c2 = ClusteringConstraints.remote_coupling(
                labels, X, interaction_metadata
            )
        except TypeError:
            c2 = ClusteringConstraints.remote_coupling(labels, X)
        c3 = ClusteringConstraints.structural_consistency(
            labels, candidate, X=X
        )
        c4 = ClusteringConstraints.dimensional_constraint(labels)

        constraint_values = {
            "message_similarity": float(c1),
            "remote_coupling": float(c2),
            "structural_consistency": float(c3),
            "dimensional_constraint": float(c4),
        }

        p_f = self._constraint_bayesian_update(
            c1, c2, c3, c4, prior=prior,
        )

        return p_f, constraint_values

    def rank_stage1_candidates(
        self,
        candidates: List[Dict[str, Any]],
        X: List[bytes],
        interaction_metadata: Optional[List[Dict[str, Any]]] = None,
        prior: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:

        scored: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []

        for candidate in candidates:
            p_f, constraints = self.compute_stage1_probability(
                candidate, X, interaction_metadata=interaction_metadata,
                prior=prior,
            )
            scored.append((candidate, p_f, constraints))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def cluster_by_candidate(self, values: List[Any]) -> np.ndarray:
     
        mapping: Dict[Any, int] = {}
        labels: List[int] = []

        for v in values:
            if v is None:
                key = _MISSING
            elif isinstance(v, (bytes, bytearray)):
                key = bytes(v)
            else:
                key = v

            if key not in mapping:
                mapping[key] = len(mapping)
            labels.append(mapping[key])

        return np.array(labels)

    def _constraint_bayesian_update(
        self,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
        prior: float = 0.1,
    ) -> float:
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("prior not valid")
        constraints = np.array([c1, c2, c3, c4], dtype=float)
        constraints = np.clip(constraints, 1e-6, 1 - 1e-6)

        likelihood_keyword = float(np.prod(constraints))
        likelihood_not_keyword = float(np.prod(1.0 - constraints))

        numerator = likelihood_keyword * prior
        denominator = (
            numerator
            + likelihood_not_keyword * (1.0 - prior)
        )

        if denominator <= 0:
            return 1e-6

        posterior = numerator / denominator
        return float(np.clip(posterior, 1e-6, 1.0 - 1e-6))

    def compute_p_bit(self, values: List[Any]) -> float:

        valid_vals = [_val_to_int(v) for v in values if v is not None]
        if not valid_vals:
            return 1e-6

        if len(set(valid_vals)) <= 1:
            return 1e-6

        max_val = max(valid_vals)
        if max_val == 0:
            return 1e-6

        msb = max_val.bit_length() - 1

        if msb > 64:
            return 1e-6

        # Calculate individual MSBs
        individual_msbs = np.array([
            0 if val == 0 else val.bit_length() - 1 for val in valid_vals
        ])

        k_indices = np.arange(msb + 1)  # [0, 1, ..., msb]

        # Paper: "Q(k) = proportion of values where MSB >= k" (Vectorized)
        q_k = np.mean(individual_msbs[:, None] >= k_indices[None, :], axis=0)

        exponents = (msb + 1) - k_indices
        p_k = np.ones(msb + 1)
        safe_mask = exponents <= 1000
        p_k[safe_mask] = 1.0 - 1.0 / (2.0 ** exponents[safe_mask])

        # Paper Eq. (8): D = sqrt(sum((Q(k) - P(k))^2))
        euclidean_dist = float(np.sqrt(np.sum((q_k - p_k) ** 2)))

        # ---------------------------------------------------------------------
        # FIX 3: VECTORIZED D_MAX (O(MSB) instead of O(MSB^2))
        # Paper Eq. (9): Dmax across concentration points m
        # ---------------------------------------------------------------------
        term_low = (1.0 - p_k) ** 2
        term_high = p_k ** 2

        cumsum_low = np.cumsum(term_low)
        cumsum_high = np.cumsum(term_high[::-1])[::-1]
        cumsum_high_shifted = np.append(cumsum_high[1:], 0.0)

        d_candidates = np.sqrt(cumsum_low + cumsum_high_shifted)
        d_max = float(np.max(d_candidates)) if len(d_candidates) > 0 else 0.0

        # Paper Eq. (10): p_bit = 1 - D / Dmax
        if d_max <= 0:
            return 1e-6

        p_bit = 1.0 - euclidean_dist / d_max
        return float(np.clip(p_bit, 1e-6, 1.0 - 1e-6))
        
    def compute_p_offset(
        self,
        candidate_type: str,
        offset: int = 0,
        boundary_B: int = 0,
    ) -> float:
        """
        Position Constraint Probability p_offset (Eq. 11).

        Paper:
            p_offset = max(0.95 - 0.01 * cand_offset, 0.7)  if cand in FOR
            p_offset = 0.60                                  if cand in NFOR

        Parameters
        ----------
        candidate_type : str
            "FOR" or "NFOR".
        offset : int
            For FOR candidates: the byte offset within the message.
            For NFOR candidates: not used (fixed 0.60).
        boundary_B : int
            Not used directly; included for interface consistency.
        """
        if candidate_type == "FOR":
            # Paper Eq. (11): max(0.95 - 0.01 * cand_offset, 0.7)
            return max(0.95 - 0.01 * offset, 0.7)
        if candidate_type == "NFOR":
            # Paper Eq. (11): 0.60 for NFOR
            return 0.60
        raise ValueError("candidate_type must be 'FOR' or 'NFOR'")

    def bayesian_update(self, p_bit: float, p_offset: float, p_f: float) -> float:
        """
        Final posterior probability P(K=1 | p_bit, p_offset) (Eq. 12–15).

        Paper:
            f_bit(K=1) = p_bit,     f_bit(K=0) = 1 - p_bit
            f_offset(K=1) = p_offset, f_offset(K=0) = 1 - p_offset
            P(K, p_bit, p_offset) ∝ f_bit(K) * f_offset(K) * P(K)

            M = p_bit * p_offset * p_f
            N = (1 - p_bit) * (1 - p_offset) * (1 - p_f)
            P(K=1 | p_bit, p_offset) = M / (M + N)
        """
        M = p_bit * p_offset * p_f
        N = (1.0 - p_bit) * (1.0 - p_offset) * (1.0 - p_f)

        denominator = M + N
        if denominator <= 0:
            return 1e-6

        posterior = M / denominator
        return float(np.clip(posterior, 1e-6, 1.0 - 1e-6))

    # ==============================================================
    #  Full Two-Stage Pipeline
    # ==============================================================

    def infer_keyword(
        self,
        candidates: List[Dict[str, Any]],
        X: List[bytes],
        interaction_metadata: Optional[List[Dict[str, Any]]] = None,
        stage1_prior: float = 0.1,
        top_k: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]]]:
        """
        Full two-stage keyword inference pipeline.

        Paper: "Select the field with the highest probability result as
        the keyword field."

        Parameters
        ----------
        candidates : List[dict]
            Merged FOR + NFOR candidate list.
        X : List[bytes]
            Full message set.
        interaction_metadata : optional
            Per-message metadata for remote coupling constraint.
        stage1_prior : float
            Prior for Stage 1 naive Bayes.
        top_k : optional int
            If specified, only the top-k Stage 1 candidates proceed to
            Stage 2. Paper says "fields ranked high in the results of the
            first stage inference."

        Returns
        -------
        best_candidate : dict or None
            The candidate with highest final probability.
        best_probability : float
            Final posterior probability.
        all_scored : list of (candidate, probability)
            All candidates scored in Stage 2, sorted by probability.
        """
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")

        # --- Stage 1: ranking ---
        stage1_ranked = self.rank_stage1_candidates(
            candidates, X,
            interaction_metadata=interaction_metadata,
            prior=stage1_prior,
        )

        # Paper: "we conduct a new round of inference on the fields ranked
        # high in the results of the first stage inference"
        if top_k is not None:
            stage1_ranked = stage1_ranked[:top_k]

        # --- Stage 2: self-constraint inference ---
        all_scored: List[Tuple[Dict[str, Any], float]] = []

        for candidate, p_f, _ in stage1_ranked:
            values = candidate["values"]

            p_bit = self.compute_p_bit(values)

            cand_type = candidate.get("type", "FOR")
            cand_offset = candidate.get("offset", 0)
            boundary_B = candidate.get("boundary_B", 0)
            p_offset = self.compute_p_offset(cand_type, cand_offset, boundary_B)

            # Paper Eq. (15): P(K=1 | p_bit, p_offset) = M / (M + N)
            final_prob = self.bayesian_update(p_bit, p_offset, p_f)

            all_scored.append((candidate, final_prob))

        all_scored.sort(key=lambda t: t[1], reverse=True)

        if not all_scored:
            return None, 0.0, []

        best_candidate, best_prob = all_scored[0]
        return best_candidate, best_prob, all_scored