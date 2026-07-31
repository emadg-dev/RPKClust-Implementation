import numpy as np
from typing import List, Dict, Any, Optional, Callable
from .optimizer import RPKClustOptimizer
from .semantic_rules import SemanticRules
from .utils import extract_for_candidates, extract_nfor_tlv_candidates

class RPKClust:

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
      
        if X is None:
            raise ValueError("X not valid")
        if not X:
            self.boundary_B = 0
            self.semantic_regions = []
            self.candidates = []
            self.best_candidate = None
            self.labels_ = np.array([], dtype=int)
            return self
        if not 0.0 < float(stage1_prior) < 1.0:
            raise ValueError("stage1_prior not valid")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k not valid")
        if interaction_metadata is not None and len(interaction_metadata) != len(X):
            raise ValueError("interaction_metadata not valid")
        if direction_labels is not None and len(direction_labels) != len(X):
            raise ValueError("direction_labels not valid")

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

        print("-" * 20)

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

        print("Two Stage Bayesian Inference...")

        stage1_ranked = self.optimizer.rank_stage1_candidates(
            self.candidates,
            X,
            interaction_metadata=interaction_metadata,
            prior=stage1_prior,
        )

        if top_k is not None:
            stage1_ranked = stage1_ranked[:top_k]

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

        scored_candidates = [c for c, _, _ in stage1_ranked]
        self.candidates = scored_candidates

        print("Keyword Selection...")
        if self.candidates:
            self.candidates.sort(key=lambda c: c["prob"], reverse=True)
            self.best_candidate = self.candidates[0]
            print(f"  Best: {self.best_candidate.get('tag', '?')} "
                  f"prob={self.best_candidate['prob']:.4f}")
        else:
            self.best_candidate = None

        print("Assigning Semantic Clusters...")
        self.labels_ = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X: List[bytes]) -> np.ndarray:
        if not self.best_candidate:
            return np.zeros(len(X), dtype=int)

        vals = self.best_candidate["values"]
        return self.optimizer.cluster_by_candidate(vals)

    def fit_predict(self, X: List[bytes], **fit_kwargs) -> np.ndarray:
        self.fit(X, **fit_kwargs)
        if self.labels_ is None:
            raise RuntimeError("fit completed without producing cluster labels")
        return self.labels_