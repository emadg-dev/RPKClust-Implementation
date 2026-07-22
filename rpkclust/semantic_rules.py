import numpy as np
import time

class SemanticRules:
    """
    Implements the 6 semantic detection rules for identifying the FOR/NFOR boundary.
    """
    def __init__(self, t_cap=None):
        # Default T_cap to current time if not provided
        self.t_cap = t_cap if t_cap else int(time.time())
        
        self.rules = [
            (1, self.check_constant),
            (1, self.check_sequence),
            (4, self.check_timestamp),
            (1, self.check_sparse),
            (2, self.check_address),
            (1, self.check_checksum)
        ]

    def check_constant(self, fragments, **kwargs):
        """Rule 1: Constant Field (Entropy H(s) == 0)"""
        return len(set(fragments)) == 1

    def check_sequence(self, fragments, **kwargs):
        """Rule 2: Sequence ID (Incrementing counters)"""
        vals = [int.from_bytes(f, 'big') for f in fragments]
        if len(vals) < 2: return False
        
        deltas = np.diff(vals)
        # Check if all differences are exactly the same (delta v is constant)
        return len(set(deltas)) == 1 and deltas[0] != 0

    def check_timestamp(self, fragments, **kwargs):
        """Rule 3: Timestamp within T_cap +/- 86400s"""
        if len(fragments[0]) != 4: return False
        try:
            vals = [int.from_bytes(f, 'big') for f in fragments]
            # Check if all values fall within the 1-day window
            return all(abs(v - self.t_cap) <= 86400 for v in vals)
        except:
            return False

    def check_sparse(self, fragments, **kwargs):
        """Rule 4: Sparse Value (Unique ratio <= 0.02 and 0 not in V_unique)"""
        vals = [int.from_bytes(f, 'big') for f in fragments]
        unique_vals = set(vals)
        ratio = len(unique_vals) / len(vals)
        return ratio <= 0.02 and 0 not in unique_vals

    def check_address(self, fragments, **kwargs):
        """Rule 5: Address (Field alternation with cross-correlation rho <= -0.8)"""
        if len(fragments) < 2: return False
        vals = [int.from_bytes(f, 'big') for f in fragments]
        
        # Simulate cross-correlation of alternating sequences
        v1 = vals[:-1]
        v2 = vals[1:]
        if np.std(v1) == 0 or np.std(v2) == 0:
            return False
            
        rho = np.corrcoef(v1, v2)[0, 1]
        return not np.isnan(rho) and rho <= -0.8

    def check_checksum(self, fragments, full_messages, offset):
        """Rule 6: Checksum (Simplified XOR-8 test)"""
        # Exclude the current offset from the XOR calculation of the full message
        for frag, msg in zip(fragments, full_messages):
            chk_byte = frag[0]
            msg_xor = 0
            for i, b in enumerate(msg):
                if i != offset:
                    msg_xor ^= b
            if msg_xor != chk_byte:
                return False
        return True