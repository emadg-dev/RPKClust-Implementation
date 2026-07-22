import numpy as np
from collections import defaultdict
from .semantic_rules import SemanticRules
from .optimizer import RPKClustOptimizer
from .utils import extract_for_candidate, extract_nfor_candidate

class RPKClust:
    """
    Main implementation of the RPKClust algorithm.
    """
    def __init__(self, t_cap=None, L=1, t_len=1, l_len=1):
        """
        :param t_cap: Base timestamp for rule 3.
        :param L: Candidate length (default 1 byte).
        :param t_len: TLV Tag length.
        :param l_len: TLV Length field length.
        """
        self.L = L
        self.t_len = t_len
        self.l_len = l_len
        self.semantic_engine = SemanticRules(t_cap=t_cap)
        self.optimizer = RPKClustOptimizer()
        
        self.boundary_B = 0
        self.best_candidate = None
        self.candidates = []
        self.labels_ = None
        self.cluster_mapping = {}

    def _identify_boundary(self, X):
        """Algorithm 1: Boundary Identification"""
        min_len = min(len(m) for m in X)
        hit_offsets = set()
        
        for offset in range(min_len):
            for rule_len, rule_func in self.semantic_engine.rules:
                if offset + rule_len <= min_len:
                    fragments = [m[offset:offset+rule_len] for m in X]
                    
                    # Rule 6 requires full messages
                    if rule_func.__name__ == 'check_checksum':
                        is_match = rule_func(fragments, X, offset)
                    else:
                        is_match = rule_func(fragments)
                        
                    if is_match:
                        hit_offsets.add(offset + rule_len - 1)
                        # Mark semantic hit and break (First-match precedence)
                        break 
                        
        self.boundary_B = max(hit_offsets) + 1 if hit_offsets else 0
        return self.boundary_B, hit_offsets

    def _generate_for_candidates(self, X, hit_offsets):
        """Generates candidates from the Fixed-Offset Region."""
        for_candidates = []
        # Filter out explicit semantic bytes
        valid_offsets = [i for i in range(self.boundary_B) if i not in hit_offsets]
        
        # Apply sliding window of length L
        for offset in valid_offsets:
            if offset % self.L == 0:
                vals = extract_for_candidate(X, offset, self.L)
                for_candidates.append({
                    'type': 'FOR',
                    'offset': offset,
                    'values': vals
                })
        return for_candidates

    def _generate_nfor_candidates(self, X):
        """Algorithm 3: Generates candidates from NFOR using TLV parsing."""
        nfor_candidates = []
        found_tags = set()
        
        for m in X:
            if len(m) <= self.boundary_B: continue
            
            i = self.boundary_B
            while i < len(m) - (self.t_len + self.l_len):
                tag = int.from_bytes(m[i:i+self.t_len], 'big')
                if tag not in found_tags:
                    vals = extract_nfor_candidate(X, tag, self.t_len, self.l_len)
                    # DetectRepeatedTLV heuristic: Ensure tag appears in > 1 message
                    valid_vals = [v for v in vals if v is not None]
                    if len(valid_vals) > 1:
                        nfor_candidates.append({
                            'type': 'NFOR',
                            'tag': tag,
                            'values': vals
                        })
                        found_tags.add(tag)
                # Naive jump (assuming 1 byte Tag, 1 byte Len) for discovery
                # A robust parser would jump by TLV length, but without knowing 
                # if the current byte is truly a Tag, we slide by 1.
                i += 1 
                
        return nfor_candidates

    def fit(self, X):
        """
        Fits the RPKClust model to the message dataset M.
        X: list of byte arrays representing network payloads.
        """
        if not X or not isinstance(X, list):
            raise ValueError("Input X must be a list of byte strings/arrays.")
            
        # 1. Boundary Identification
        _, hit_offsets = self._identify_boundary(X)
        
        # 2. Candidate Generation
        c_for = self._generate_for_candidates(X, hit_offsets)
        c_nfor = self._generate_nfor_candidates(X)
        self.candidates = c_for + c_nfor
        
        # 3. Inference & Probability Calculation
        best_prob = -1
        
        for cand in self.candidates:
            # Stage 1
            p_f = self.optimizer.compute_stage1_prior(cand['values'])
            
            # Stage 2
            p_bit = self.optimizer.compute_p_bit(cand['values'])
            offset_val = cand['offset'] if cand['type'] == 'FOR' else 0
            p_offset = self.optimizer.compute_p_offset(cand['type'], offset_val)
            
            # Bayesian Update
            prob = self.optimizer.bayesian_update(p_bit, p_offset, p_f)
            cand['prob'] = prob
            
            if prob > best_prob:
                best_prob = prob
                self.best_candidate = cand

        # 4. Final Clustering based on best candidate
        if self.best_candidate:
            self._assign_clusters(X, self.best_candidate['values'])
        else:
            # Fallback if no candidates found
            self.labels_ = np.zeros(len(X))
            
        return self

    def _assign_clusters(self, X, values):
        """Maps specific keyword values to integer cluster labels."""
        self.labels_ = np.zeros(len(X), dtype=int)
        current_cluster_id = 0
        
        for i, val in enumerate(values):
            if val is None:
                self.labels_[i] = -1 # Noise / Missing
                continue
                
            val_int = int.from_bytes(val, 'big')
            if val_int not in self.cluster_mapping:
                self.cluster_mapping[val_int] = current_cluster_id
                current_cluster_id += 1
                
            self.labels_[i] = self.cluster_mapping[val_int]

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def get_candidate_summary(self):
        """
        Exports the internal Bayesian evaluations for presentation purposes (How we approached it).
        Returns a pandas DataFrame sorted by final probability.
        """
        import pandas as pd
        if not self.candidates:
            return pd.DataFrame()
            
        records = []
        for c in self.candidates:
            records.append({
                "Type": c['type'],
                "Offset/Tag": c.get('offset', c.get('tag')),
                "P_bit (Bit-use)": round(self.optimizer.compute_p_bit(c['values']), 4),
                "P_offset (Position)": round(self.optimizer.compute_p_offset(c['type'], c.get('offset', 0)), 4),
                "P_f (Prior)": round(self.optimizer.compute_stage1_prior(c['values']), 4),
                "Final Prob P(K=1)": round(c.get('prob', 0), 4)
            })
            
        df = pd.DataFrame(records)
        return df.sort_values(by="Final Prob P(K=1)", ascending=False).head(10)