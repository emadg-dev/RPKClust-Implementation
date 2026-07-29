"""
RPKClust Section 3.6: Keyword Inference (keyword_inference.py)

Two-stage probability inference for protocol keyword identification.

First Stage (Section 3.6, "The First Stage"):
    - Cluster messages by each candidate's values
    - Evaluate four Netplier constraints: message similarity, remote coupling,
      structural consistency, dimensional
    - Calculate posterior probability p_f for each candidate
    - Rank candidates by p_f

Second Stage (Section 3.6, "The Second Stage"):
    - Bit-use constraint p_bit (Eq. 7–10)
    - Position constraint p_offset (Eq. 11)
    - Final posterior P(K=1 | p_bit, p_offset) = M / (M + N) (Eq. 12–15)
    - Select field with highest probability as the keyword

Paper: "We take the probability of cluster constraint inference as the prior
probability, that is, P(K=1) = p_f and P(K=0) = 1 - p_f."
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .constraints import ClusteringConstraints


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

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
    """
    Two-Stage Bayesian Inference Model for Protocol Keyword Identification.

    Stage 1 produces p_f (prior probability from clustering constraints).
    Stage 2 combines p_f with p_bit and p_offset to produce the final
    posterior probability P(K=1 | p_bit, p_offset).
    """

    # ==============================================================
    #  First Stage: Clustering Constraint Inference
    # ==============================================================

    def compute_stage1_probability(
        self,
        candidate: Dict[str, Any],
        X: List[bytes],
        interaction_metadata: Optional[List[Dict[str, Any]]] = None,
        prior: float = 0.1,
    ) -> Tuple[float, Dict[str, float]]:
        """
        First Stage: compute p_f (prior probability) from four Netplier
        clustering constraints.

        Paper: "We traverse this list and cluster the messages according to
        the values of each candidate. Then we observe whether the clustering
        results meet the clustering constraints."

        Parameters
        ----------
        candidate : dict
            Candidate field with 'values' key (list of bytes/None per message).
        X : List[bytes]
            Full message set.
        interaction_metadata : optional
            Per-message metadata (source/dest IP, ports, timestamps) needed
            for remote coupling constraint. If None, remote coupling may
            be degraded.
        prior : float
            Prior probability P(K=1) for a random candidate being a keyword.
            The paper does not specify this value for Stage 1. Default 0.1
            reflects that keyword fields are rare among all candidates.

        Returns
        -------
        p_f : float
            Posterior probability from constraint inference (used as prior
            in Stage 2).
        constraint_values : dict
            Individual constraint scores for debugging/ranking.
        """
        if X is None:
            raise ValueError("X must be a sequence of messages, not None")
        values = candidate.get("values")
        if values is None or len(values) != len(X):
            raise ValueError("candidate values must contain one entry per message")
        if interaction_metadata is not None and len(interaction_metadata) != len(X):
            raise ValueError("interaction_metadata must contain one entry per message")
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("prior must be a finite probability strictly between 0 and 1")
        labels = self.cluster_by_candidate(values)

        c1 = ClusteringConstraints.message_similarity(labels, X)
        # remote_coupling may need interaction metadata; fall back to
        # the original 2-arg signature for compatibility.
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
        """
        First Stage ranking: traverse all candidates, compute p_f for each,
        and rank by probability.

        Paper: "Based on the probability inference of the above constraints,
        we initially rank the keyword fields according to their probabilities."

        Returns
        -------
        List of (candidate, p_f, constraint_values) sorted by p_f descending.
        """
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
        """
        Cluster messages by candidate field values.
        Messages with the same value are assigned the same cluster label.
        Uses _MISSING sentinel for None values so absent NFOR TLVs
        do not collide with real string values.
        """
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
        """
        Naive Bayes posterior for Stage 1 constraint inference.

        NOTE: The paper does not specify the exact formula for Stage 1
        probability. It says the inference is "similar to the factor graph
        used in Netplier." This implementation uses a naive Bayes approach
        (independent constraint factors), which is a reasonable approximation.

        Paper: "We calculate the posterior probability of each field being
        a keyword field."
        """
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("prior must be a finite probability strictly between 0 and 1")
        constraints = np.array([c1, c2, c3, c4], dtype=float)
        constraints = np.clip(constraints, 1e-6, 1 - 1e-6)

        # Likelihood P(D | K=1) = product of constraint satisfactions.
        likelihood_keyword = float(np.prod(constraints))

        # Likelihood P(D | K=0) = product of constraint violations.
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

    # ==============================================================
    #  Second Stage: Self-Constraint Inference
    # ==============================================================

    def compute_p_bit(self, values: List[Any]) -> float:
        """
        Bit-Use Constraint Probability p_bit (Eq. 7–10).

        Paper:
            MSB = highest bit position used by any value in the field.
            Q(k) = proportion of values where MSB >= k.
            P(k) = 1 - 1 / (2^(MSB+1-k)).
            D = sqrt(sum((Q(k) - P(k))^2)).
            Dmax = max over concentration points m of:
                sqrt(sum_{k=0}^{m}(1-P(k))^2 + sum_{k=m+1}^{MSB}(P(k))^2).
            p_bit = 1 - D / Dmax.
        """
        valid_vals = [_val_to_int(v) for v in values if v is not None]
        if not valid_vals:
            return 1e-6

        # Constant field filter: single unique value carries no keyword info.
        if len(set(valid_vals)) <= 1:
            return 1e-6

        # Paper: "traverse all values and determine the position of the
        # Most Significant Bit (MSB)."
        # MSB = highest bit position used by any value.
        max_val = max(valid_vals)
        if max_val == 0:
            return 1e-6

        msb = max_val.bit_length() - 1  # 0-based MSB position

        # Paper: "Q(k) = proportion of values where MSB >= k"
        # For each value, compute its individual MSB, then count how many
        # values have individual MSB >= k.
        individual_msbs = []
        for val in valid_vals:
            if val == 0:
                individual_msbs.append(0)
            else:
                individual_msbs.append(val.bit_length() - 1)

        n = len(valid_vals)
        q_k = np.zeros(msb + 1)
        for k in range(msb + 1):
            count = sum(1 for ind_msb in individual_msbs if ind_msb >= k)
            q_k[k] = count / n

        # Paper Eq. (7): P(k) = 1 - 1 / (2^(MSB+1-k))
        p_k = np.zeros(msb + 1)
        for k in range(msb + 1):
            p_k[k] = 1.0 - 1.0 / (2.0 ** (msb + 1 - k))

        # Paper Eq. (8): D = sqrt(sum((Q(k) - P(k))^2))
        euclidean_dist = float(np.sqrt(np.sum((q_k - p_k) ** 2)))

        # Paper Eq. (9): Dmax = max over concentration points m
        # Dmax = sqrt(sum_{k=0}^{m}(1-P(k))^2 + sum_{k=m+1}^{MSB}(P(k))^2)
        # The concentration point m represents the extreme case where all
        # values concentrate at a single bit position.
        d_max = 0.0
        for m in range(msb + 1):
            # Q concentrated at bit m: Q(k)=1 for k<=m, Q(k)=0 for k>m
            sum_low = sum((1.0 - p_k[k]) ** 2 for k in range(0, m + 1))
            sum_high = sum((p_k[k]) ** 2 for k in range(m + 1, msb + 1))
            d_candidate = np.sqrt(sum_low + sum_high)
            if d_candidate > d_max:
                d_max = d_candidate

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