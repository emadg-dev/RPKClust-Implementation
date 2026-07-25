"""
RPKClust Core Pipeline (rpkclust.py)
End-to-End Region-Partitioned Keyword Identification and Message Clustering.
"""

import numpy as np
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
        self.candidates = []
        self.best_candidate = None
        self.labels_ = None

    def fit(self, X):
        """
        Executes RPKClust pipeline on binary message trace X.
        """
        if X is None or len(X) == 0:
            return self

        # Step 1: Boundary Identification
        self.boundary_B, _ = SemanticRules.identify_boundary(X)

        # Step 2: Region-Partitioned Candidate Generation
        for_cands = extract_for_candidates(X, self.boundary_B)
        nfor_cands = extract_nfor_tlv_candidates(X, self.boundary_B)

        self.candidates = for_cands + nfor_cands

        # Step 3: Two-Stage Bayesian Inference
        total_msgs = len(X)
        for cand in self.candidates:
            cand['stage1_prob'] = (
                self.optimizer.compute_stage1_probability(
                    cand,
                    X
                )
            )
            cand['p_bit'] = self.optimizer.compute_p_bit(
                cand['values']
            )
            cand['p_offset'] = self.optimizer.compute_p_offset(
                cand['type'],
                cand.get('offset', 0),
                self.boundary_B
            )
            cand['prob'] = self.optimizer.bayesian_update(
                cand['p_bit'],
                cand['p_offset'],
                cand['stage1_prob']
            )

        # Step 4: Keyword Selection
        if self.candidates:
            self.candidates.sort(key=lambda c: c['prob'], reverse=True)
            self.best_candidate = self.candidates[0]
        else:
            self.best_candidate = None

        # Step 5: Semantic Clustering Assignment
        self.labels_ = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X):
        """Assigns cluster labels based on selected keyword field values."""
        if not self.best_candidate:
            return np.zeros(len(X), dtype=int)

        vals = self.best_candidate['values']
        unique_map = {}
        labels = []

        for v in vals:
            if v is None:
                key = "MISSING"
            elif isinstance(v, (bytes, bytearray)):
                key = bytes(v)
            else:
                key = str(v)

            if key not in unique_map:
                unique_map[key] = len(unique_map)
            labels.append(unique_map[key])

        return np.array(labels, dtype=int)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_