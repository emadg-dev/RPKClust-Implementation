"""
RPKClust Candidate Generation Utilities (utils.py)
Region-partitioned candidate extraction for FOR and NFOR message regions.
"""

import numpy as np


def extract_for_candidates(X, boundary_B, widths=(1, 2, 4)):
    """
    Algorithm 2: Fixed-Offset Region (FOR) Candidate Generation.
    Extracts fixed byte slice candidates for offsets < boundary_B.
    """
    candidates = []
    if (X is None or len(X) == 0) or boundary_B <= 0:
        return candidates

    min_len = min(len(msg) for msg in X)
    max_offset = min(boundary_B, min_len)

    for offset in range(max_offset):
        for width in widths:
            if offset + width <= max_offset:
                vals = [msg[offset:offset + width] for msg in X]
                candidates.append({
                    'type': 'FOR',
                    'tag': f"FOR_Offset_{offset}_W{width}",
                    'offset': offset,
                    'width': width,
                    'values': vals
                })

    return candidates


def extract_nfor_tlv_candidates(X, boundary_B, t_len=1, l_len=1):
    """
    Algorithm 3: Non-Fixed-Offset Region (NFOR) TLV Candidate Generation.
    Generic, protocol-independent TLV sequence parser starting at boundary_B.
    """
    msg_options = []
    all_tags = set()

    for msg in X:
        opts = {}
        offset = boundary_B

        while offset + t_len + l_len <= len(msg):
            tag_bytes = msg[offset: offset + t_len]
            tag_val = int.from_bytes(tag_bytes, 'big')

            length = int.from_bytes(msg[offset + t_len: offset + t_len + l_len], 'big')
            val_start = offset + t_len + l_len
            val_end = val_start + length

            # Structural TLV verification: verify length payload fits in message
            if 0 < length <= (len(msg) - val_start):
                val_bytes = msg[val_start:val_end]

                if tag_val not in opts:
                    opts[tag_val] = {
                        'value': val_bytes,
                        'offset': offset
                    }
                    all_tags.add(tag_val)

                offset = val_end
            else:
                # Advance 1 byte if alignment breaks
                offset += 1

        msg_options.append(opts)

    nfor_candidates = []
    for tag_val in sorted(all_tags):
        vals = [opts[tag_val]['value'] if tag_val in opts else None for opts in msg_options]
        offsets = [opts[tag_val]['offset'] for opts in msg_options if tag_val in opts]

        valid_cnt = sum(1 for v in vals if v is not None)
        if valid_cnt > 0:
            avg_offset = float(np.mean(offsets)) if offsets else boundary_B
            nfor_candidates.append({
                'type': 'NFOR',
                'tag': f"Option_{tag_val}",
                'tag_val': tag_val,
                'values': vals,
                'offset': avg_offset
            })

    return nfor_candidates