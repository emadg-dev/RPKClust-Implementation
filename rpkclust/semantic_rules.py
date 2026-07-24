"""
RPKClust Boundary Identification & Semantic Rules (semantic_rules.py)
Generic evaluation of structural semantics across packet byte traces.
"""

import numpy as np


class SemanticRules:
    """
    Evaluates semantic rules for boundary detection and field profiling.
    """

    @staticmethod
    def is_constant(fragments):
        """Rule 1: Constant Field (Zero Variance across non-None values)."""
        valid = [f for f in fragments if f is not None]
        if not valid:
            return False
        return len(set(valid)) == 1

    @staticmethod
    def is_sequence(fragments):
        """Rule 2: Monotonic Sequence Number (v_{i+1} = v_i + 1)."""
        valid = [f for f in fragments if f is not None]
        if len(valid) < 3:
            return False

        nums = []
        for v in valid:
            if isinstance(v, (bytes, bytearray)):
                nums.append(int.from_bytes(v, 'big'))
            elif isinstance(v, int):
                nums.append(v)
            else:
                return False

        diffs = np.diff(nums)
        return bool(np.all(diffs == 1))

    @staticmethod
    def is_timestamp(fragments):
        """Rule 3: Monotonic Timestamp (v_{i+1} >= v_i)."""
        valid = [f for f in fragments if f is not None]
        if len(valid) < 3:
            return False

        nums = []
        for v in valid:
            if isinstance(v, (bytes, bytearray)):
                nums.append(int.from_bytes(v, 'big'))
            elif isinstance(v, int):
                nums.append(v)
            else:
                return False

        diffs = np.diff(nums)
        return bool(np.all(diffs >= 0) and np.max(nums) > np.min(nums))

    @classmethod
    def identify_boundary(cls, X):
        """
        Identifies FOR-NFOR Boundary B.
        Finds the end offset of the contiguous fixed header region starting from offset 0.
        """
        if not X:
            return 0, set()

        min_len = min(len(msg) for msg in X)
        semantic_hits = set()

        offset = 0
        while offset < min_len:
            matched_width = 0

            # Evaluate candidate structural widths (4, 2, 1) at current offset
            for width in (4, 2, 1):
                if offset + width <= min_len:
                    fragments = [msg[offset:offset + width] for msg in X]

                    # Multi-byte sequence/timestamp anchors
                    if width >= 2 and (cls.is_timestamp(fragments) or cls.is_sequence(fragments)):
                        matched_width = width
                        break
                    # Leading fixed-header constants (e.g. Magic / Version bytes)
                    elif cls.is_constant(fragments) and offset <= 2:
                        matched_width = width
                        break

            if matched_width > 0:
                for p in range(offset, offset + matched_width):
                    semantic_hits.add(p)
                offset += matched_width
            else:
                # If offset == 0 and no anchor matched (e.g., leading 1-byte OpCode before Timestamp),
                # check if a valid structural anchor starts within the next few bytes
                if offset == 0:
                    found_anchor = False
                    for skip in range(1, min(4, min_len)):
                        for w in (4, 2):
                            if skip + w <= min_len:
                                frags = [msg[skip:skip + w] for msg in X]
                                if cls.is_timestamp(frags) or cls.is_sequence(frags) or cls.is_constant(frags):
                                    offset = skip
                                    found_anchor = True
                                    break
                        if found_anchor:
                            break
                    if not found_anchor:
                        break
                else:
                    break

        boundary_B = max(offset, max(semantic_hits) + 1 if semantic_hits else 0)
        return boundary_B, semantic_hits