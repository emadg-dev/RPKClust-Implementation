"""
RPKClust Protocol Field Discovery and Message Clustering Pipeline.

Paper Sections 3.1–3.6:
    1. FOR-NFOR Boundary Identification (Algorithm 1)
    2. Region-Partitioned Candidate Generation (Algorithms 2 & 3)
    3. Two-Stage Bayesian Inference (Section 3.6)
    4. Semantic Clustering Assignment

Integration notes:
    - Uses SemanticRules.identify_boundary + collect_semantic_regions
      (shared _scan_semantic_regions to avoid drift).
    - Passes semantic_regions to extract_for_candidates (Algorithm 2).
    - Uses the revised RPKClustOptimizer with paper-accurate Eq. 7–15.
    - Stage 1 ranks candidates by p_f before Stage 2 (paper: "fields
      ranked high in the results of the first stage inference").
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable

from .optimizer import RPKClustOptimizer
from .semantic_rules import SemanticRules
from .utils import extract_for_candidates, extract_nfor_tlv_candidates


class RPKClust:
    """
    RPKClust Protocol Field Discovery and Message Clustering Pipeline.
    """

    def __init__(self):
        self.optimizer = RPKClustOptimizer()
        self.boundary_B = 0
        self.semantic_regions: List[Dict[str, Any]] = []
        self.candidates: List[Dict[str, Any]] = []
        self.best_candidate: Optional[Dict[str, Any]] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(
        self,
        X: List[bytes],
        interaction_metadata: Optional[List[Dict[str, Any]]] = None,
        capture_start: Optional[float] = None,
        capture_end: Optional[float] = None,
        timezone_offset: float = 0.0,
        direction_labels: Optional[List[Any]] = None,
        t_len: int = 1,
        l_len: int = 1,
        validate_tlv: Optional[Callable[..., bool]] = None,
        stage1_prior: float = 0.1,
        top_k: Optional[int] = None,
    ) -> "RPKClust":
        """
        Executes the full RPKClust pipeline on binary message trace X.

        Parameters
        ----------
        X : List[bytes]
            Application-layer binary messages.
        interaction_metadata : optional
            Per-message metadata (source/dest IP, ports, timestamps) for
            the remote coupling constraint in Stage 1.
        capture_start, capture_end : optional float
            Capture time window for timestamp rule in boundary detection.
        timezone_offset : float
            Timezone offset in seconds for timestamp rule.
        direction_labels : optional
            Per-message direction labels for address rule.
        t_len, l_len : int
            TLV type and length field widths for NFOR candidate generation.
        validate_tlv : optional callable
            Semantic TLV validator for NFOR candidate generation.
        stage1_prior : float
            Prior probability for Stage 1 naive Bayes.
        top_k : optional int
            If set, only the top-k Stage 1 candidates proceed to Stage 2.
            Paper: "we conduct a new round of inference on the fields ranked
            high in the results of the first stage inference."
        """
        if X is None:
            raise ValueError("X must be a sequence of messages, not None")
        if not X:
            self.boundary_B = 0
            self.semantic_regions = []
            self.candidates = []
            self.best_candidate = None
            self.labels_ = np.array([], dtype=int)
            return self
        if not 0.0 < float(stage1_prior) < 1.0:
            raise ValueError("stage1_prior must be a finite probability strictly between 0 and 1")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        if interaction_metadata is not None and len(interaction_metadata) != len(X):
            raise ValueError("interaction_metadata must contain one entry per message")
        if direction_labels is not None and len(direction_labels) != len(X):
            raise ValueError("direction_labels must contain one entry per message")

        # ---- Step 1: Boundary Identification + Semantic Region Collection
        print("Identifying Boundaries...")
        self.boundary_B, _ = SemanticRules.identify_boundary(
            X,
            capture_start=capture_start,
            capture_end=capture_end,
            timezone_offset=timezone_offset,
            direction_labels=direction_labels,
        )
        self.semantic_regions = SemanticRules.collect_semantic_regions(
            X,
            capture_start=capture_start,
            capture_end=capture_end,
            timezone_offset=timezone_offset,
            direction_labels=direction_labels,
        )
        print(f"  Boundary B = {self.boundary_B}")
        print(f"  Semantic regions: {len(self.semantic_regions)}")

        # ---- Step 2: Region-Partitioned Candidate Generation
        print("Extracting FOR Candidates...")
        for_cands = extract_for_candidates(
            X,
            self.boundary_B,
            semantic_regions=self.semantic_regions,
        )
        print(f"  FOR candidates: {len(for_cands)}")

        print("Extracting NFOR Candidates...")
        nfor_cands = extract_nfor_tlv_candidates(
            X,
            self.boundary_B,
            t_len=t_len,
            l_len=l_len,
            validate_tlv=validate_tlv,
        )
        print(f"  NFOR candidates: {len(nfor_cands)}")

        self.candidates = for_cands + nfor_cands
        print(f"  Total candidates: {len(self.candidates)}")

        # ---- Step 3: Two-Stage Bayesian Inference
        print("Two Stage Bayesian Inference...")

        # Stage 1: Rank all candidates by p_f (clustering constraints).
        stage1_ranked = self.optimizer.rank_stage1_candidates(
            self.candidates,
            X,
            interaction_metadata=interaction_metadata,
            prior=stage1_prior,
        )

        # Paper: "fields ranked high in the results of the first stage
        # inference" proceed to Stage 2.
        if top_k is not None:
            stage1_ranked = stage1_ranked[:top_k]

        # Stage 2: Self-constraint inference (p_bit, p_offset, final posterior).
        for candidate, p_f, constraint_values in stage1_ranked:
            candidate["stage1_prob"] = p_f
            candidate["constraint_values"] = constraint_values

            p_bit = self.optimizer.compute_p_bit(candidate["values"])
            candidate["p_bit"] = p_bit

            cand_type = candidate.get("type", "FOR")
            cand_offset = candidate.get("offset", 0)
            p_offset = self.optimizer.compute_p_offset(
                cand_type, cand_offset, self.boundary_B
            )
            candidate["p_offset"] = p_offset

            final_prob = self.optimizer.bayesian_update(
                p_bit, p_offset, p_f
            )
            candidate["prob"] = final_prob

        # Update self.candidates to reflect scored results.
        scored_candidates = [c for c, _, _ in stage1_ranked]
        self.candidates = scored_candidates

        # ---- Step 4: Keyword Selection
        print("Keyword Selection...")
        if self.candidates:
            self.candidates.sort(key=lambda c: c["prob"], reverse=True)
            self.best_candidate = self.candidates[0]
            print(f"  Best: {self.best_candidate.get('tag', '?')} "
                  f"prob={self.best_candidate['prob']:.4f}")
        else:
            self.best_candidate = None

        # ---- Step 5: Semantic Clustering Assignment
        print("Assigning Semantic Clustering...")
        self.labels_ = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X: List[bytes]) -> np.ndarray:
        """
        Assigns cluster labels based on the selected keyword field's values.
        Reuses optimizer.cluster_by_candidate for consistency with Stage 1.
        """
        if not self.best_candidate:
            return np.zeros(len(X), dtype=int)

        vals = self.best_candidate["values"]
        return self.optimizer.cluster_by_candidate(vals)

    def fit_predict(self, X: List[bytes], **fit_kwargs) -> np.ndarray:
        """Fit and return cluster labels. Accepts same kwargs as fit()."""
        self.fit(X, **fit_kwargs)
        if self.labels_ is None:
            raise RuntimeError("fit completed without producing cluster labels")
        return self.labels_