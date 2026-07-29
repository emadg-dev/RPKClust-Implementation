
# RPKClust Boundary Identification & Semantic Rules (semantic_rules.py)
# Generic evaluation of structural semantics across packet byte traces.


import numpy as np
import math
import zlib
import struct
from typing import List, Set, Tuple, Any, Optional, Dict, Callable


class SemanticRules:
    """
    Evaluates semantic rules for boundary detection and field profiling.
    All rules follow the formal definitions in RPKClust Section 3.3.
    """

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_int_list(fragments: List[Any], width: int) -> Optional[List[int]]:
        """
        Converts a list of byte/int fragments into a uniform list of integers.
        Returns None if any fragment is invalid, None, or has an incorrect byte length.
        Validates every value is in [0, 2^(8*width) - 1].
        """
        max_value = (1 << (8 * width)) - 1
        nums = []
        for f in fragments:
            if f is None:
                return None
            if isinstance(f, (bytes, bytearray)):
                if len(f) != width:
                    return None
                nums.append(int.from_bytes(f, 'big'))
            elif isinstance(f, int):
                if f < 0 or f > max_value:          # range-check every element
                    return None
                nums.append(f)
            else:
                return None
        return nums

    @staticmethod
    def _extract_fragments(X: List[bytes], offset: int, width: int) -> Optional[List[bytes]]:
        """
        Extracts a slice [offset : offset + width] from each message in cluster X.
        Returns None if any message is too short for the slice.
        """
        fragments = []
        for msg in X:
            if offset + width > len(msg):
                return None
            fragments.append(msg[offset:offset + width])
        return fragments

    @staticmethod
    def _calc_internet_checksum(data: bytes) -> int:
        """RFC 1071 Internet Checksum (16-bit ones' complement sum)."""
        if len(data) % 2 == 1:
            data += b'\x00'
        words = struct.unpack(f">{len(data) // 2}H", data)
        checksum = sum(words)
        while checksum >> 16:
            checksum = (checksum & 0xFFFF) + (checksum >> 16)
        return (~checksum) & 0xFFFF

    @staticmethod
    def _calc_crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
        """CRC-16/CCITT-FALSE checksum."""
        crc = init
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    @staticmethod
    def _calc_xor8(data: bytes) -> int:
        """XOR-8 checksum: XOR of all bytes in the data range."""
        result = 0
        for b in data:
            result ^= b
        return result & 0xFF

    # ------------------------------------------------------------------
    #  Rule 1 – Constant Field
    # ------------------------------------------------------------------

    @classmethod
    def is_constant(cls, fragments: List[Any], width: int = 1) -> bool:
        """
        Paper Eq. (1): H(s) = 0 AND for all i, j: s_i ≡ s_j.
        Zero Shannon entropy is equivalent to all values being identical.
        Input: byte slices of any length.
        """
        if len(fragments) < 2:
            return False

        first = fragments[0]
        if first is None:
            return False

        if isinstance(first, (bytes, bytearray)) and len(first) != width:
            return False

        for f in fragments[1:]:
            if f is None or f != first:
                return False
        return True

    # ------------------------------------------------------------------
    #  Rule 2 – Sequence ID
    # ------------------------------------------------------------------

    @classmethod
    def is_sequence(cls, fragments: List[Any], width: int = 2) -> bool:
        """
        Paper Eq. (2):
            for all i in [1, n-1]:  delta = v_{i+1} - v_i = delta
            AND  v_i in [0, 2^(8k) - 1]   (no wrap-around)
        k = byte length (1-4).  delta is a positive constant step.
        """
        nums = cls._to_int_list(fragments, width)
        if nums is None or len(nums) < 3:
            return False

        delta = nums[1] - nums[0]
        if delta <= 0:                         # must be an increment
            return False

        for i in range(1, len(nums) - 1):
            if nums[i + 1] - nums[i] != delta:
                return False

        return True

    # ------------------------------------------------------------------
    #  Rule 3 – Timestamp
    # ------------------------------------------------------------------

    @classmethod
    def is_timestamp(
        cls,
        fragments: List[Any],
        width: int = 4,
        capture_start: Optional[float] = None,
        capture_end: Optional[float] = None,
        timezone_offset: float = 0.0,
    ) -> bool:
        """
        Paper Eq. (3):  t_candidate in T_cap +/- Delta
            T_cap = [t_start, t_end]   (capture window)
            Delta  = 86400 s           (one day tolerance)
        Input: 4/8-byte slices plus capture time range.
        Supports timezone offsets.

        The capture window is REQUIRED per the paper. If it is not
        provided the rule returns False (conservative – no semantic
        evidence of a timestamp).
        """
        nums = cls._to_int_list(fragments, width)
        if nums is None or len(nums) < 3:
            return False

        # Capture window is required by the paper.
        if capture_start is None or capture_end is None:
            return False

        delta = 86400.0  # paper: +/- 86400 seconds

        lo = capture_start - delta
        hi = capture_end + delta

        # Apply timezone offset: the raw integer is interpreted as
        # (timestamp + timezone_offset) so we compare in that frame.
        for v in nums:
            adjusted = float(v) + timezone_offset
            if not (lo <= adjusted <= hi):
                return False

        # Paper requires only that candidates fall inside T_cap ± Δ.
        return True

    # ------------------------------------------------------------------
    #  Rule 4 – Sparse Value
    # ------------------------------------------------------------------

    @classmethod
    def is_sparse(cls, fragments: List[Any], width: int = 1) -> bool:
        """
        Paper Eq. (4):
            |V_unique| / 2^(8k) <= 0.02   AND   0 not in V_unique
            k in {1, 2}
        Identifies underutilized non-zero fields (e.g., protocol flags).
        """
        if width not in (1, 2):
            return False

        nums = cls._to_int_list(fragments, width)
        if nums is None or not nums:
            return False

        unique_vals = set(nums)

        # 0 must not appear in the unique value set.
        if 0 in unique_vals:
            return False

        unique_count = len(unique_vals)
        max_possible = 2 ** (8 * width)
        ratio = unique_count / float(max_possible)

        return ratio <= 0.02

    # ------------------------------------------------------------------
    #  Rule 5 – Address (paired fields)
    # ------------------------------------------------------------------

    @classmethod
    def is_address(
        cls,
        fragments_f1: List[Any],
        fragments_f2: List[Any],
        width: int = 4,
        direction_labels: Optional[List[Any]] = None,
    ) -> bool:
        """
        Paper Eq. (5):
            For all c in C:  s1^c ≡ s2^c   (direction equivalence)
            AND  rho(s1, s2) <= -0.8      (Pearson correlation)
        Input: adjacent 2-4 byte slices.

        When direction_labels is provided (one label per message, e.g.
        'client'/'server' or 0/1), the exact paper condition is evaluated:
        within each direction class, field-1 values must all be identical,
        field-2 values must all be identical, and the two classes must use
        swapped address pairs.

        When direction_labels is None, a reciprocal-pair heuristic is used
        as a self-contained fallback: every (a, b) pair must have its
        mirror (b, a) present, and the mapping must be bijective.
        """
        nums1 = cls._to_int_list(fragments_f1, width)
        nums2 = cls._to_int_list(fragments_f2, width)

        if nums1 is None or nums2 is None or len(nums1) < 3:
            return False
        if len(nums1) != len(nums2):
            return False

        # Non-zero variance to avoid division by zero in correlation.
        std1 = np.std(nums1)
        std2 = np.std(nums2)
        if std1 == 0 or std2 == 0:
            return False

        # --- Condition 1: direction-based equivalence -----------------
        if direction_labels is not None:
            # Exact paper implementation using class labels.
            if len(direction_labels) != len(nums1):
                return False

            # Group messages by direction class.
            classes: Dict[Any, Tuple[List[int], List[int]]] = {}
            for label, v1, v2 in zip(direction_labels, nums1, nums2):
                if label not in classes:
                    classes[label] = ([], [])
                classes[label][0].append(v1)
                classes[label][1].append(v2)

            if len(classes) < 2:
                return False

            # Within each class: field-1 constant, field-2 constant.
            class_pairs: Dict[Any, Tuple[int, int]] = {}
            for label, (f1_vals, f2_vals) in classes.items():
                if len(set(f1_vals)) != 1 or len(set(f2_vals)) != 1:
                    return False
                a = f1_vals[0]
                b = f2_vals[0]
                if a == b:
                    return False
                class_pairs[label] = (a, b)

            # Direction equivalence: the two classes must have swapped
            # address pairs.  i.e. class1 = (A, B) and class2 = (B, A).
            pair_list = list(class_pairs.values())
            for i in range(len(pair_list)):
                for j in range(i + 1, len(pair_list)):
                    a_i, b_i = pair_list[i]
                    a_j, b_j = pair_list[j]
                    if a_i != b_j or b_i != a_j:
                        return False

        else:
            # Heuristic fallback: reciprocal-pair check.
            pairs = list(zip(nums1, nums2))
            pair_set = set(pairs)

            if len(pair_set) < 2:
                return False

            map12: Dict[int, int] = {}
            map21: Dict[int, int] = {}

            for a, b in pair_set:
                if a == b:
                    return False
                if a in map12 and map12[a] != b:
                    return False
                if b in map21 and map21[b] != a:
                    return False
                map12[a] = b
                map21[b] = a

            for a, b in pair_set:
                if (b, a) not in pair_set:
                    return False

        # --- Condition 2: Pearson correlation <= -0.8 -----------------
        corr_matrix = np.corrcoef(nums1, nums2)
        rho = corr_matrix[0, 1]

        return bool(rho <= -0.8)

    # ------------------------------------------------------------------
    #  Rule 6 – Checksum
    # ------------------------------------------------------------------

    @classmethod
    def is_checksum(
        cls,
        fragments: List[Any],
        width: int = 2,
        messages: Optional[List[bytes]] = None,
        offset: int = 0,
        strict: bool = True,
    ) -> bool:
        """
        Paper Eq. (6):  exists A in A_set,  A(D) ≡ s
        Paper's named algorithms A_set = {CRC-16, XOR-8}.
        Input: 1/2/4-byte slices plus data range (full message + offset).

        When strict=True (default) only the paper's named algorithms are
        evaluated: XOR-8 for width 1, CRC-16 for width 2.
        When strict=False, practical extensions are also tried: RFC 1071
        Internet Checksum and CRC-32 for width 4.
        """
        if not fragments or messages is None or len(messages) != len(fragments):
            return False

        if width not in (1, 2, 4):
            return False

        raw_frags = []
        for f in fragments:
            if not isinstance(f, (bytes, bytearray)) or len(f) != width:
                return False
            raw_frags.append(bytes(f))

        # ---- 1-byte: XOR-8 -----------------------------------------
        if width == 1:
            xor8_exclude = True
            xor8_suffix = True
            for msg, frag in zip(messages, raw_frags):
                val = frag[0]
                payload_ex = msg[:offset] + msg[offset + width:]
                payload_suf = msg[offset + width:]
                if val != cls._calc_xor8(payload_ex):
                    xor8_exclude = False
                if val != cls._calc_xor8(payload_suf):
                    xor8_suffix = False
            return xor8_exclude or xor8_suffix

        # ---- 2-byte: CRC-16 (paper) + RFC 1071 (extension) ------------
        if width == 2:
            crc16_match = True
            rfc1071_match = True
            for msg, frag in zip(messages, raw_frags):
                val_big = int.from_bytes(frag, 'big')

                # CRC-16/CCITT over payload excluding the checksum field.
                payload_ex = msg[:offset] + msg[offset + width:]
                calc_crc16 = cls._calc_crc16_ccitt(payload_ex)
                if val_big != calc_crc16:
                    crc16_match = False

                if not strict:
                    # RFC 1071: zero out the checksum field then compute.
                    zeroed_msg = msg[:offset] + b'\x00\x00' + msg[offset + width:]
                    calc_rfc = cls._calc_internet_checksum(zeroed_msg)
                    if val_big != calc_rfc:
                        rfc1071_match = False

            if strict:
                return crc16_match
            return crc16_match or rfc1071_match

        # ---- 4-byte: CRC-32 (extension only) ------------------------
        if width == 4:
            if strict:
                return False   # paper has no 4-byte algorithm

            crc32_match_exclude = True
            crc32_match_suffix = True
            for msg, frag in zip(messages, raw_frags):
                val_big = int.from_bytes(frag, 'big')
                val_little = int.from_bytes(frag, 'little')

                payload_ex = msg[:offset] + msg[offset + width:]
                payload_suf = msg[offset + width:]

                crc_ex = zlib.crc32(payload_ex) & 0xFFFFFFFF
                crc_suf = zlib.crc32(payload_suf) & 0xFFFFFFFF

                if val_big != crc_ex and val_little != crc_ex:
                    crc32_match_exclude = False
                if val_big != crc_suf and val_little != crc_suf:
                    crc32_match_suffix = False

            return crc32_match_exclude or crc32_match_suffix

        return False

    # ------------------------------------------------------------------
    #  Rule registry
    # ------------------------------------------------------------------

    RULE_REGISTRY: List[Dict[str, Any]] = [

        {
            "name": "constant",
            "widths": [8, 4, 2, 1],
            "type": "single",
            "func": lambda frags, w, X, offset:
                SemanticRules.is_constant(frags, width=w)
        },

        {
            "name": "sequence",
            "widths": [1, 2, 3, 4],
            "type": "single",
            "func": lambda frags, w, X, offset:
                SemanticRules.is_sequence(frags, width=w)
        },

        {
            "name": "timestamp",
            "widths": [4, 8],
            "type": "single",
            # NOTE: identify_boundary special-cases this rule by name to
            # pass capture_start/capture_end/timezone_offset. The lambda
            # below is used only for standalone calls without metadata.
            "func": lambda frags, w, X, offset:
                SemanticRules.is_timestamp(frags, width=w)
        },

        {
            "name": "sparse",
            "widths": [1, 2],
            "type": "single",
            "func": lambda frags, w, X, offset:
                SemanticRules.is_sparse(frags, width=w)
        },

        {
            "name": "address",
            "widths": [2, 3, 4],
            "type": "pair",
            "func": lambda f1, f2, w, X, offset, direction_labels=None:
                SemanticRules.is_address(f1, f2, width=w, direction_labels=direction_labels)
        },

        {
            "name": "checksum",
            "widths": [1, 2],
            "type": "single",
            "func": lambda frags, w, X, offset:
                SemanticRules.is_checksum(
                    fragments=frags, width=w, messages=X, offset=offset,
                    strict=True
                )
        },
    ]

    # ------------------------------------------------------------------
    #  Shared Semantic Region Scanner
    # ------------------------------------------------------------------

    @classmethod
    def _scan_semantic_regions(
        cls,
        X: List[bytes],
        capture_start: Optional[float] = None,
        capture_end: Optional[float] = None,
        timezone_offset: float = 0.0,
        direction_labels: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Shared scanner used by both identify_boundary and
        collect_semantic_regions to avoid logic drift.

        Scans ALL offsets 0..min_len-1. At each offset, tries every rule;
        on the first rule that matches, records a semantic region dict
        and stops trying further rules at that offset (first-match precedence).

        Returns a list of regions:
            {"name": str, "offset": int, "width": int}
        where width is the FULL span of the matched field (2*width for
        pair rules like address).
        """
        if not X:
            return []

        min_len = min(len(m) for m in X)
        regions: List[Dict[str, Any]] = []

        for offset in range(min_len):

            matched_at_offset = False

            for rule in cls.RULE_REGISTRY:

                if matched_at_offset:
                    break

                for width in rule["widths"]:

                    # ---- Single-field rules ----
                    if rule["type"] == "single":

                        if offset + width > min_len:
                            continue

                        fragments = cls._extract_fragments(X, offset, width)
                        if fragments is None:
                            continue

                        if rule["name"] == "timestamp":
                            matched = cls.is_timestamp(
                                fragments,
                                width=width,
                                capture_start=capture_start,
                                capture_end=capture_end,
                                timezone_offset=timezone_offset,
                            )
                        else:
                            matched = rule["func"](fragments, width, X, offset)

                        if matched:
                            regions.append({
                                "name": rule["name"],
                                "offset": offset,
                                "width": width,
                            })
                            matched_at_offset = True
                            break

                    # ---- Pair-field rule (Address) ----
                    elif rule["type"] == "pair":

                        if offset + 2 * width > min_len:
                            continue

                        fragments_1 = cls._extract_fragments(X, offset, width)
                        fragments_2 = cls._extract_fragments(
                            X, offset + width, width
                        )

                        if fragments_1 is None or fragments_2 is None:
                            continue

                        matched = rule["func"](
                            fragments_1, fragments_2, width, X, offset,
                            direction_labels=direction_labels,
                        )

                        if matched:
                            regions.append({
                                "name": rule["name"],
                                "offset": offset,
                                "width": 2 * width,
                            })
                            matched_at_offset = True
                            break

        return regions

    # ------------------------------------------------------------------
    #  Algorithm 1 – FOR-NFOR Boundary Detection
    # ------------------------------------------------------------------

    @classmethod
    def identify_boundary(
        cls,
        X: List[bytes],
        capture_start: Optional[float] = None,
        capture_end: Optional[float] = None,
        timezone_offset: float = 0.0,
        direction_labels: Optional[List[Any]] = None,
    ) -> Tuple[int, Set[int]]:
        """
        RPKClust Algorithm 1: FOR-NFOR Boundary Detection.

        Scans ALL offsets 0..min_len-1. At each offset, tries every rule;
        on the first rule that matches, records the right-boundary endpoint
        {offset + l_r - 1} in hit_offsets and stops trying further rules at
        that offset (first-match precedence). The outer loop continues to
        the next offset. Final boundary B = max(hit_offsets) + 1.

        Input:  Message set M, semantic rule library R (from RULE_REGISTRY),
                optional capture window [capture_start, capture_end] and
                timezone_offset for timestamp rule.
        Output: FOR-NFOR boundary B
        """
        regions = cls._scan_semantic_regions(
            X,
            capture_start=capture_start,
            capture_end=capture_end,
            timezone_offset=timezone_offset,
            direction_labels=direction_labels,
        )

        if not regions:
            return 0, set()

        hit_offsets: Set[int] = set()
        for r in regions:
            hit_offsets.add(r["offset"] + r["width"] - 1)

        boundary_B = max(hit_offsets) + 1
        return boundary_B, hit_offsets

    @classmethod
    def collect_semantic_regions(
        cls,
        X: List[bytes],
        capture_start: Optional[float] = None,
        capture_end: Optional[float] = None,
        timezone_offset: float = 0.0,
        direction_labels: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect typed semantic regions from boundary scanning.

        Returns a list of dicts: {"name", "offset", "width"}.
        - Single-field rules have width = rule width (1, 2, 4, 8).
        - Pair-field rules (address) have width = 2 * rule width.

        These regions are needed by Algorithm 2 (extract_for_candidates)
        to construct E_sem, F_sparse, and the scanning sequence S.
        """
        return cls._scan_semantic_regions(
            X,
            capture_start=capture_start,
            capture_end=capture_end,
            timezone_offset=timezone_offset,
            direction_labels=direction_labels,
        )