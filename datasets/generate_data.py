import time
import numpy as np

def generate_simple_for(n=1000, seed=42):
    """
    Dataset 1: Simple Fixed-Offset Region (FOR) validation.
    Structure:
      - Byte 0: OpCode (0x01, 0x02, 0x03) -> True Keyword / Cluster Label
      - Byte 1: Incrementing Sequence Counter (Semantic Rule: Sequence)
      - Bytes 2-9: Uniform random noise payload
    """
    np.random.seed(seed)
    opcodes = [b'\x01', b'\x02', b'\x03']
    messages = []
    labels = []
    
    for i in range(n):
        cluster_id = i % 3
        op = opcodes[cluster_id]
        seq = (i % 256).to_bytes(1, 'big')
        payload = np.random.bytes(8)
        
        messages.append(op + seq + payload)
        labels.append(cluster_id)
        
    return messages, np.array(labels)

def generate_nfor_tlv(n=1000, seed=42):
    """
    Dataset 2: Challenging Non-Fixed-Offset Region (NFOR) TLV dataset.
    Structure:
      - Bytes 0-19: Fixed header with noise and timestamp semantic rule
      - Bytes 20+: NFOR zone with variable padding and embedded TLV tag
      - Tag 0x0A, 0x0B, 0x0C, 0x0D determines true cluster
    """
    np.random.seed(seed)
    current_time = int(time.time())
    tags = [b'\x0a', b'\x0b', b'\x0c', b'\x0d']
    messages = []
    labels = []
    
    for i in range(n):
        cluster_id = i % 4
        # 1. FOR Header (20 bytes total)
        # Bytes 0-3: Timestamp (Rule 3 match)
        ts = (current_time + (i % 100)).to_bytes(4, 'big')
        # Bytes 4-19: Random noise header
        header_noise = np.random.bytes(16)
        for_header = ts + header_noise
        
        # 2. NFOR Body (Variable padding + TLV)
        pad_len = np.random.randint(0, 10)
        nfor_padding = np.random.bytes(pad_len)
        
        tag = tags[cluster_id]
        val_len = np.random.randint(2, 6)
        length_byte = val_len.to_bytes(1, 'big')
        val_data = np.random.bytes(val_len)
        
        tlv_block = tag + length_byte + val_data
        tail_noise = np.random.bytes(np.random.randint(2, 8))
        
        msg = for_header + nfor_padding + tlv_block + tail_noise
        messages.append(msg)
        labels.append(cluster_id)
        
    return messages, np.array(labels)

def generate_high_dimensional(n=1000, feature_len=256, n_clusters=6, seed=42):
    """
    Dataset 3: High-dimensional scalability test dataset.
    Structure:
      - Extended fixed header + large payload (256 bytes)
      - Opcode placed at offset 12 determining cluster among n_clusters
      - High noise-to-signal ratio to test PCA and scalability
    """
    np.random.seed(seed)
    messages = []
    labels = []
    
    for i in range(n):
        cluster_id = i % n_clusters
        prefix = np.random.bytes(12)
        opcode = cluster_id.to_bytes(1, 'big')
        suffix = np.random.bytes(feature_len - 13)
        
        msg = prefix + opcode + suffix
        messages.append(msg)
        labels.append(cluster_id)
        
    return messages, np.array(labels)