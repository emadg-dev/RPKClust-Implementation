import numpy as np

def calculate_entropy(byte_array_list):
    """Calculates Shannon entropy of a list of byte fragments."""
    values = [bytes(b) for b in byte_array_list]
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def extract_for_candidate(messages, offset, length=1):
    """Extracts a Fixed-Offset Region (FOR) candidate from all messages."""
    return [m[offset:offset+length] if len(m) >= offset+length else None for m in messages]

def extract_nfor_candidate(messages, tlv_tag, t_len=1, l_len=1):
    """Extracts a Non-Fixed-Offset Region (NFOR) candidate using TLV parsing."""
    extracted = []
    for m in messages:
        found = False
        i = 0
        while i < len(m) - (t_len + l_len):
            # Parse Tag
            tag = int.from_bytes(m[i:i+t_len], byteorder='big')
            if tag == tlv_tag:
                # Parse Length
                length = int.from_bytes(m[i+t_len:i+t_len+l_len], byteorder='big')
                val_start = i + t_len + l_len
                val_end = val_start + length
                if val_end <= len(m):
                    extracted.append(m[val_start:val_end])
                    found = True
                    break
            i += 1
        if not found:
            extracted.append(None)
    return extracted

def compute_empirical_bit_prob(values, msb):
    """
    Computes Q(k): the empirical probability that bit k is 1.
    k=0 is the LSB, k=msb is the MSB.
    """
    valid_vals = [int.from_bytes(v, 'big') for v in values if v is not None]
    if not valid_vals:
        return np.zeros(msb + 1)
        
    q_k = np.zeros(msb + 1)
    for k in range(msb + 1):
        # Count how many values have the k-th bit set to 1
        bit_mask = 1 << k
        count_ones = sum(1 for v in valid_vals if (v & bit_mask))
        q_k[k] = count_ones / len(valid_vals)
    return q_k