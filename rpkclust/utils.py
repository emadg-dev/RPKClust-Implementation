from typing import List, Dict, Any, Set, Tuple, Optional, Callable
import numpy as np

SemanticRegion = Dict[str, Any]
FORCandidate = Dict[str, Any]
SPARSE_RULES = ("sparse",)


def _region_offsets(region: SemanticRegion) -> Set[int]:
    """Return the set of byte offsets covered by a semantic region [offset, offset+width-1]."""
    start = region["offset"]
    end = start + region["width"]
    return set(range(start, end))


def _is_continuous(
    s: int,
    L: int,
    excluded_offsets: Set[int],
    max_offset: int,
) -> bool:
    """
    Paper Line 7: IsContinuous(FFOR, s, L)
    Checks that the interval [s, s+L-1] is fully inside the FOR
    and does NOT overlap any excluded semantic offset.
    """
    if s + L > max_offset:
        return False
    for pos in range(s, s + L):
        if pos in excluded_offsets:
            return False
    return True


def extract_for_candidates(
    X: List[bytes],
    boundary_B: int,
    semantic_regions: Optional[List[SemanticRegion]] = None,
    candidate_lengths: Tuple[int, ...] = (1, 2, 4),
) -> List[FORCandidate]:
    """
    Algorithm 2: Keyword Candidate Generation in FOR.

    Parameters
    ----------
    X : List[bytes]
        Message set M (raw bytes).
    boundary_B : int
        FOR-NFOR boundary from Algorithm 1.
    semantic_regions : Optional[List[SemanticRegion]]
        Typed semantic hits from boundary identification, e.g.:
        {"name": "constant", "offset": 0, "width": 2}
        {"name": "sparse",   "offset": 8, "width": 1}
        Algorithm 2 requires E_sem — if None, a warning is emitted and
        all FOR offsets become candidates (NOT paper-accurate).
    candidate_lengths : Tuple[int, ...]
        Variable sliding window widths L to try (paper: "variable sliding
        window").  Default (1, 2, 4) covers common protocol field sizes.

    Returns
    -------
    List[FORCandidate]
        Each candidate dict: {"offset", "width", "values"}.
        Only offsets that pass semantic filtering, modulo alignment, and
        continuity checks are returned.
    """
    candidates: List[FORCandidate] = []

    if X is None or len(X) == 0 or boundary_B <= 0:
        return candidates
    if not candidate_lengths or any(
        not isinstance(length, int) or isinstance(length, bool) or length <= 0
        for length in candidate_lengths
    ):
        raise ValueError("candidate_lengths must contain positive integers")

    min_len = min(len(msg) for msg in X)
    max_offset = min(boundary_B, min_len)

    # ----------------------------------------------------------
    # Lines 1–3: Set construction
    # ----------------------------------------------------------

    FOR = set(range(max_offset))                       # [0, |F_FOR| - 1]

    if semantic_regions is None:
        import warnings
        warnings.warn(
            "Algorithm 2 requires typed semantic_regions (E_sem). "
            "Without it, all FOR offsets become candidates (NOT paper-accurate). "
            "Pass regions like {'name': 'constant', 'offset': 0, 'width': 2}.",
            UserWarning,
            stacklevel=2,
        )
        semantic_regions = []

    # Partition regions into excluded (E_sem) and sparse (F_sparse).
    # Paper Line 1: E ← E_sem \ F_sparse
    #   ALL non-sparse semantic regions are excluded by default.
    #   Only sparse regions are re-introduced into the scanning sequence.
    sparse_regions: List[SemanticRegion] = []
    excluded_regions: List[SemanticRegion] = []

    for region in semantic_regions:
        name = region.get("name", "")
        if name in SPARSE_RULES:
            sparse_regions.append(region)
        else:
            excluded_regions.append(region)

    # Line 1: E ← E_sem \ F_sparse
    #   E = byte offsets covered by excluded semantic regions
    #   (sparse regions are NOT excluded — they are re-introduced in S).
    #   Offset-based subtraction handles overlaps correctly: if a sparse
    #   offset is also covered by a non-sparse region, it stays in E.
    all_semantic_offsets: Set[int] = set()
    sparse_offsets: Set[int] = set()
    sparse_starts: Set[int] = set()

    for region in excluded_regions:
        all_semantic_offsets |= _region_offsets(region) & FOR

    for region in sparse_regions:
        offsets = _region_offsets(region) & FOR
        sparse_offsets |= offsets
        if offsets:
            sparse_starts.add(region["offset"])

    # Offset-based subtraction matches E_sem \ F_sparse:
    # sparse-covered offsets are removed from E, even if semantic
    # detections overlap.
    E = all_semantic_offsets - sparse_offsets

    # F_sparse = starting offsets of sparse fields (bounded to FOR).
    F_sparse: Set[int] = sparse_starts & FOR

    # Line 2: U ← [0, |F_FOR| − 1] \ E
    #   Undetected offsets = FOR offsets not covered by excluded regions.
    U: Set[int] = FOR - E

    # Line 3: S ← U ∪ F_sparse
    #   Scanning sequence = undetected offsets + sparse field starts,
    #   bounded to the FOR.
    S: Set[int] = (U | F_sparse) & FOR

    # ----------------------------------------------------------
    # Lines 4–11: Candidate generation
    # ----------------------------------------------------------

    for L in candidate_lengths:
        for s in sorted(S):
            # Universal bounds check: candidate must fit inside FOR.
            if s + L > max_offset:
                continue

            # Line 6: s ≡ 0 (mod L)  — length-aligned offset
            if s % L != 0:
                continue

            # Line 7: continuity check for L > 1
            if L > 1 and not _is_continuous(s, L, E, max_offset):
                continue

            # Line 8: C ← C ∪ {s}
            # Extract values for downstream use.
            values = [msg[s:s + L] for msg in X]

            candidates.append({
                "type": "FOR",
                "tag": f"FOR_Offset_{s}_W{L}",
                "offset": s,
                "width": L,
                "values": values,
            })

    # Deduplicate (same (offset, width) could arise from multiple L values
    # or sparse re-introduction).
    seen: Set[Tuple[int, int]] = set()
    deduped: List[FORCandidate] = []
    for c in candidates:
        key = (c["offset"], c["width"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped

TLVRecord = Dict[str, Any]
Validator = Callable[[bytes, int, int, int, int, int, bytes], bool]


def _parse_tlv_at(
    m: bytes,
    offset: int,
    t_len: int,
    l_len: int,
) -> Optional[TLVRecord]:
    """
    Parse one TLV at the given offset within an NFOR message slice.
    Returns None if the TLV header or value overflows the message
    (paper Lines 8–11: data integrity check).
    """
    header_end = offset + t_len + l_len
    if header_end > len(m):
        return None

    type_bytes = m[offset: offset + t_len]
    len_bytes = m[offset + t_len: header_end]

    type_val = int.from_bytes(type_bytes, "big")
    len_val = int.from_bytes(len_bytes, "big")

    value_start = header_end
    value_end = value_start + len_val

    # Paper Line 8: overflow check.
    if value_end > len(m):
        return None

    value_bytes = m[value_start:value_end]

    # Paper: T-V combined as keyword candidate.
    tv_bytes = type_bytes + value_bytes

    # Paper Line 15: P stores m[start:end] (full TLV segment).
    tlv_bytes = m[offset:value_end]

    return {
        "start": offset,
        "end": value_end,
        "type_val": type_val,
        "len_val": len_val,
        "type_bytes": type_bytes,
        "len_bytes": len_bytes,
        "value_bytes": value_bytes,
        "tv_bytes": tv_bytes,
        "tlv_bytes": tlv_bytes,
    }


def _default_validate_tlv(
    m: bytes,
    offset: int,
    t_len: int,
    l_len: int,
    type_val: int,
    len_val: int,
    value_bytes: bytes,
) -> bool:
    """
    Generic fallback for ValidateTLV (paper Line 12).
    The paper does not define ValidateTLV in detail — it is a semantic
    checker that confirms valid encoding beyond the structural overflow
    check. Protocol-specific validation should be injected when available.

    Default: accept any structurally valid TLV (already passed overflow check).
    """
    return True


def _detect_repeated_tlv(
    m: bytes,
    offset: int,
    t_len: int,
    l_len: int,
    validate_tlv: Validator,
) -> Optional[TLVRecord]:
    """
    Paper Line 16: DetectRepeatedTLV(m, end, t_len, l_len).
    A repeated TLV exists if another valid TLV begins exactly at `offset`
    (immediately after the previous one). This extends boundaries for
    consecutive repeated TLV patterns.
    """
    rec = _parse_tlv_at(m, offset, t_len, l_len)
    if rec is None:
        return None

    ok = validate_tlv(
        m,
        offset,
        t_len,
        l_len,
        rec["type_val"],
        rec["len_val"],
        rec["value_bytes"],
    )
    if not ok:
        return None

    return rec


def extract_nfor_tlv_patterns(
    X: List[bytes],
    boundary_B: int,
    t_len: int = 1,
    l_len: int = 1,
    validate_tlv: Optional[Validator] = None,
    include_repeated_in_P: bool = True,
) -> Tuple[List[TLVRecord], List[Dict[str, Any]]]:
    """
    RPKClust Algorithm 3: Keyword candidate generation in NFOR.

    Parameters
    ----------
    X : List[bytes]
        Full message set M. NFOR portions are extracted internally
        starting at boundary_B.
    boundary_B : int
        FOR-NFOR boundary from Algorithm 1. NFOR = msg[boundary_B:].
    t_len : int
        Type field length in bytes.
    l_len : int
        Length field length in bytes.
    validate_tlv : Optional[Validator]
        Semantic validation function (paper Line 12). If None, a default
        that accepts any structurally valid TLV is used. Protocol-specific
        validation should be injected when available.
    include_repeated_in_P : bool
        If True (default), repeated TLV members are also appended to P.
        Strict Algorithm 3 only records the initial pattern in P (Line 15)
        before the repetition loop. Set False for strict paper compliance.
        The repeated members are always counted for B regardless.

    Returns
    -------
    P : List[TLVRecord]
        Detected TLV patterns. Each record contains:
        - type_val, type_bytes, len_val, value_bytes, tv_bytes, tlv_bytes
        - relative_start/end (within NFOR slice)
        - absolute_start/end (within original message)
        - message_index
    B : List[Dict[str, Any]]
        Repeated TLV sequence boundaries. Each entry contains:
        - relative_start/end, absolute_start/end
        - first_end (end of first TLV in the sequence)
        - repeated_count
        - types (list of type_vals in the sequence)
        - bytes (raw bytes of the repeated sequence)
    """
    if validate_tlv is None:
        validate_tlv = _default_validate_tlv

    P: List[TLVRecord] = []
    B: List[Dict[str, Any]] = []

    if X is None or len(X) == 0:
        return P, B

    if (
        not isinstance(t_len, int) or isinstance(t_len, bool) or t_len <= 0
        or not isinstance(l_len, int) or isinstance(l_len, bool) or l_len <= 0
    ):
        raise ValueError("t_len and l_len must be positive integers")

    if boundary_B < 0:
        boundary_B = 0

    # ----------------------------------------------------------
    # Paper Line 2: max_len ← min{len(m) | m ∈ M_NFOR}
    # Paper input is M_NFOR, so slice full messages into NFOR portions.
    # ----------------------------------------------------------

    M_NFOR: List[Tuple[int, bytes]] = []
    for msg_idx, msg in enumerate(X):
        if boundary_B >= len(msg):
            M_NFOR.append((msg_idx, b""))
        else:
            M_NFOR.append((msg_idx, msg[boundary_B:]))

    max_len = min((len(m) for _, m in M_NFOR), default=0)

    # ----------------------------------------------------------
    # Paper Lines 3–27: Main loop
    # ----------------------------------------------------------

    for msg_idx, m in M_NFOR:

        # Paper Line 4: offset ← 0 (relative to NFOR start)
        offset = 0

        # Paper Line 5: while offset ≤ len(m) − (t_len + l_len)
        while offset <= len(m) - (t_len + l_len):

            # Paper Lines 6–7: extract type and length fields
            # Paper Lines 8–11: data integrity (overflow) check
            rec = _parse_tlv_at(m, offset, t_len, l_len)

            if rec is None:
                # Paper Line 9: offset ← offset + 1
                offset += 1
                continue

            # Paper Line 12: ValidateTLV semantic check
            ok = validate_tlv(
                m,
                offset,
                t_len,
                l_len,
                rec["type_val"],
                rec["len_val"],
                rec["value_bytes"],
            )

            if not ok:
                # Paper Line 24: offset ← offset + 1
                offset += 1
                continue

            # Paper Line 13: start ← offset
            start = offset

            # Paper Line 14: end ← offset + t_len + l_len + len_val
            end = rec["end"]

            # Paper Line 15: P ← P ∪ {(m[start:end], type_val)}
            rec.update({
                "message_index": msg_idx,
                "relative_start": rec["start"],
                "relative_end": rec["end"],
                "absolute_start": boundary_B + rec["start"],
                "absolute_end": boundary_B + rec["end"],
            })
            P.append(rec)

            # Paper Lines 16–18: DetectRepeatedTLV loop
            repeated_count = 0
            repeated_types: List[int] = []

            while True:
                next_rec = _detect_repeated_tlv(
                    m, end, t_len, l_len, validate_tlv,
                )
                if next_rec is None:
                    break

                repeated_count += 1
                repeated_types.append(next_rec["type_val"])

                # Downstream convenience: record repeated TLV in P.
                # Strict Algorithm 3 only adds the initial pattern (Line 15).
                if include_repeated_in_P:
                    next_rec.update({
                        "message_index": msg_idx,
                        "relative_start": next_rec["start"],
                        "relative_end": next_rec["end"],
                        "absolute_start": boundary_B + next_rec["start"],
                        "absolute_end": boundary_B + next_rec["end"],
                        "is_repeated_member": True,
                    })
                    P.append(next_rec)

                # Paper Line 17: end ← end + t_len + l_len + new_len_val
                end = next_rec["end"]

            # Paper Lines 19–21: store B for repetition counts >= 1.
            # (Paper text: "For repetition counts >= 1, start-end positions
            # are stored in B.")
            if repeated_count > 0:
                B.append({
                    "message_index": msg_idx,
                    "relative_start": start,
                    "relative_end": end,
                    "absolute_start": boundary_B + start,
                    "absolute_end": boundary_B + end,
                    "first_end": rec["end"],
                    "repeated_count": repeated_count,
                    "types": [rec["type_val"]] + repeated_types,
                    "bytes": m[start:end],
                })

            # Paper Line 22: offset ← end
            offset = end

    return P, B


# ------------------------------------------------------------------
#  Aggregation wrapper (downstream convenience, NOT Algorithm 3)
# ------------------------------------------------------------------

def extract_nfor_tlv_candidates(
    X: List[bytes],
    boundary_B: int,
    t_len: int = 1,
    l_len: int = 1,
    validate_tlv: Optional[Validator] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregation wrapper that calls Algorithm 3 (extract_nfor_tlv_patterns)
    and converts the output into a candidate list format compatible with
    downstream keyword inference.

    This is NOT part of Algorithm 3 — it is a convenience layer that
    aggregates P by type_val across messages, using T-V combined values
    (paper: "use T-V as the combined keyword field candidates").

    Returns
    -------
    List[Dict[str, Any]]
        Each candidate dict:
        - tag_val: the TLV type value
        - values: T-V combined bytes per message (None if absent)
        - offsets: absolute offset per message (None if absent)
        - patterns: all TLVRecords for this type
        - repeated_boundaries: B entries involving this type
    """
    P, B = extract_nfor_tlv_patterns(
        X, boundary_B, t_len=t_len, l_len=l_len, validate_tlv=validate_tlv,
    )

    # Group by type_val, keeping first occurrence per message.
    by_type: Dict[int, Dict[int, TLVRecord]] = {}

    for rec in P:
        type_val = rec["type_val"]
        msg_idx = rec["message_index"]

        if type_val not in by_type:
            by_type[type_val] = {}

        if msg_idx not in by_type[type_val]:
            by_type[type_val][msg_idx] = rec

    candidates: List[Dict[str, Any]] = []
    n = len(X)

    for type_val, per_msg in sorted(by_type.items()):
        values: List[Optional[bytes]] = []
        offsets: List[Optional[int]] = []

        for i in range(n):
            if i in per_msg:
                rec = per_msg[i]
                # Paper: T-V combined as keyword candidate.
                values.append(rec["tv_bytes"])
                offsets.append(rec["absolute_start"])
            else:
                values.append(None)
                offsets.append(None)

        valid_count = sum(1 for v in values if v is not None)

        candidates.append({
            "type": "NFOR",
            "tag": f"NFOR_TV_Type_{type_val}",
            "tag_val": type_val,
            "values": values,
            "offsets": offsets,
            "valid_count": valid_count,
            "patterns": list(per_msg.values()),
            "repeated_boundaries": [
                b for b in B if type_val in b["types"]
            ],
        })

    return candidates