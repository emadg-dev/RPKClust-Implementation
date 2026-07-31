from typing import List, Dict, Any, Set, Tuple, Optional, Callable
import numpy as np

SemanticRegion = Dict[str, Any]
FORCandidate = Dict[str, Any]
SPARSE_RULES = ("sparse",)


def _region_offsets(region: SemanticRegion) -> Set[int]:
    start = region["offset"]
    end = start + region["width"]
    return set(range(start, end))


def _is_continuous(
    s: int,
    L: int,
    excluded_offsets: Set[int],
    max_offset: int,
) -> bool:
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
    FOR = set(range(max_offset))

    if semantic_regions is None:
        import warnings
        warnings.warn(
            "Algorithm 2 requires typed semantic_regions (E_sem).",
        )
        semantic_regions = []

    sparse_regions: List[SemanticRegion] = []
    excluded_regions: List[SemanticRegion] = []

    for region in semantic_regions:
        name = region.get("name", "")
        if name in SPARSE_RULES:
            sparse_regions.append(region)
        else:
            excluded_regions.append(region)

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

    E = all_semantic_offsets - sparse_offsets

    F_sparse: Set[int] = sparse_starts & FOR

    U: Set[int] = FOR - E

    S: Set[int] = (U | F_sparse) & FOR

    for L in candidate_lengths:
        for s in sorted(S):
            if s + L > max_offset:
                continue

            if s % L != 0:
                continue

            if L > 1 and not _is_continuous(s, L, E, max_offset):
                continue

            values = [msg[s:s + L] for msg in X]

            candidates.append({
                "type": "FOR",
                "tag": f"FOR_Offset_{s}_W{L}",
                "offset": s,
                "width": L,
                "values": values,
            })

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

    header_end = offset + t_len + l_len
    if header_end > len(m):
        return None

    type_bytes = m[offset: offset + t_len]
    len_bytes = m[offset + t_len: header_end]

    type_val = int.from_bytes(type_bytes, "big")
    len_val = int.from_bytes(len_bytes, "big")

    value_start = header_end
    value_end = value_start + len_val

    if value_end > len(m):
        return None

    value_bytes = m[value_start:value_end]

    tv_bytes = type_bytes + value_bytes

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

    # Verify exact length match
    if len(value_bytes) != len_val:
        return False
        
    # Prevent absurdly large length values from false-positive type/length bytes
    if len_val > len(m) - (offset + t_len + l_len):
        return False

    return True


def _detect_repeated_tlv(
    m: bytes,
    offset: int,
    t_len: int,
    l_len: int,
    validate_tlv: Validator,
) -> Optional[TLVRecord]:

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
        raise ValueError("t_len and l_len not valid")

    if boundary_B < 0:
        boundary_B = 0

    M_NFOR: List[Tuple[int, bytes]] = []
    for msg_idx, msg in enumerate(X):
        if boundary_B >= len(msg):
            M_NFOR.append((msg_idx, b""))
        else:
            M_NFOR.append((msg_idx, msg[boundary_B:]))

    max_len = min((len(m) for _, m in M_NFOR), default=0)

    for msg_idx, m in M_NFOR:

        offset = 0

        while offset <= len(m) - (t_len + l_len):

            rec = _parse_tlv_at(m, offset, t_len, l_len)

            if rec is None:
                offset += 1
                continue

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
                offset += 1
                continue

            start = offset

            end = rec["end"]

            rec.update({
                "message_index": msg_idx,
                "relative_start": rec["start"],
                "relative_end": rec["end"],
                "absolute_start": boundary_B + rec["start"],
                "absolute_end": boundary_B + rec["end"],
            })
            P.append(rec)

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

                end = next_rec["end"]

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

            offset = end

    return P, B

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