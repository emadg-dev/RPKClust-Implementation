"""
RPKClust Generic Dataset Generator (generate_data.py)
Generates synthetic binary message traces for Fixed-Offset Region (FOR) 
and Non-Fixed-Offset Region (NFOR TLV) benchmark evaluation.
"""

import struct
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class GenericDatasetGenerator:
    """
    Object-oriented generator for synthetic protocol traffic datasets.
    Provides structured ground-truth metadata alongside binary payloads.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_for_dataset(
        self, num_messages: int = 1000
    ) -> Tuple[List[bytes], np.ndarray, Dict[str, Any]]:
        """
        Generates a generic Fixed-Offset Region (FOR) binary message dataset.

        Structure (Total fixed header = 12 bytes):
            - Magic Constant  (2B) : Offset 0..1 (0xABCD)
            - Version         (1B) : Offset 2    (0x01)
            - OpCode / Type   (1B) : Offset 3    [KEYWORD -> Cluster Ground Truth]
            - Timestamp       (4B) : Offset 4..7 (Sequential integer)
            - Sequence Num    (4B) : Offset 8..11 (Sequential integer)
            - Variable Body   (4-16B): Random Payload

        Returns:
            X (List[bytes]): List of binary messages.
            y (np.ndarray): Target cluster labels (derived from OpCode).
            metadata (Dict[str, Any]): Ground-truth parameters for evaluation.
        """
        if not isinstance(num_messages, int) or isinstance(num_messages, bool) or num_messages < 0:
            raise ValueError("num_messages must be a non-negative integer")
        rng = np.random.default_rng(self.seed)
        X: List[bytes] = []
        y: List[int] = []
        interaction_metadata: List[Dict[str, Any]] = []

        opcodes = [0x01, 0x02, 0x03, 0x04]

        for i in range(num_messages):
            magic = 0xABCD
            version = 1
            # Adjacent messages model a request/response transaction and
            # intentionally share an opcode for remote-coupling evaluation.
            opcode = int(rng.choice(opcodes)) if i % 2 == 0 else y[-1]
            timestamp = 10000 + i
            sequence = i
            
            # Variable length tail payload
            payload_len = int(rng.integers(4, 17))
            payload = rng.bytes(payload_len)

            # Pack fixed header (12 bytes total)
            fixed_header = struct.pack(">HBBII", magic, version, opcode, timestamp, sequence)
            msg = fixed_header + payload

            X.append(msg)
            y.append(opcode)
            interaction_metadata.append({
                "session_id": i // 2,
                "direction": "client" if i % 2 == 0 else "server",
                "timestamp": float(i),
            })

        metadata = {
            "dataset_type": "FOR",
            "true_boundary_B": 12,        # Fixed header portion
            "true_keyword_offset": 3,     # OpCode byte index
            "true_keyword_width": 1,
            "num_clusters": len(opcodes),
            "interaction_metadata": interaction_metadata,
        }

        return X, np.array(y, dtype=int), metadata

    def generate_nfor_dataset(
        self, num_messages: int = 1000
    ) -> Tuple[List[bytes], np.ndarray, Dict[str, Any]]:
        """
        Generates a generic Non-Fixed-Offset Region (NFOR TLV) binary message dataset.

        Header (Fixed Boundary B = 7 bytes):
            - Magic Constant (2B) : Offset 0..1 (0xABCD)
            - Version        (1B) : Offset 2    (0x01)
            - Timestamp      (4B) : Offset 3..6

        Body (Variable NFOR TLV Region):
            - Command TLV  (Type=0x0A, Len=1, Value=[0x10, 0x20, 0x30, 0x40]) -> Target Keyword
            - Data TLV     (Type=0x0B, Len=2..8, Data=random)
            - Optional TLV (Type=0x0C, Len=1..4, Data=random) [30% occurrence]

        Returns:
            X (List[bytes]): List of binary messages.
            y (np.ndarray): Target cluster labels (derived from Command TLV value).
            metadata (Dict[str, Any]): Ground-truth parameters for evaluation.
        """
        if not isinstance(num_messages, int) or isinstance(num_messages, bool) or num_messages < 0:
            raise ValueError("num_messages must be a non-negative integer")
        rng = np.random.default_rng(self.seed)
        X: List[bytes] = []
        y: List[int] = []
        interaction_metadata: List[Dict[str, Any]] = []

        cmd_values = [0x10, 0x20, 0x30, 0x40]

        for i in range(num_messages):
            # Adjacent messages model a request/response transaction.
            cmd_val = int(rng.choice(cmd_values)) if i % 2 == 0 else y[-1]

            # Fixed Header (7 Bytes)
            magic = 0xABCD
            version = 1
            timestamp = 20000 + i
            header = struct.pack(">HBI", magic, version, timestamp)

            # Required Command TLV (3 Bytes)
            tlv_cmd = struct.pack(">BBB", 0x0A, 1, cmd_val)

            # Required Data TLV (2 + data_len Bytes)
            data_len = int(rng.integers(2, 9))
            data = rng.bytes(data_len)
            tlv_data = struct.pack(">BB", 0x0B, data_len) + data

            tlvs = [tlv_cmd, tlv_data]

            # Optional TLV (30% probability)
            if rng.random() < 0.30:
                opt_len = int(rng.integers(1, 5))
                opt_data = rng.bytes(opt_len)
                tlv_optional = struct.pack(">BB", 0x0C, opt_len) + opt_data
                tlvs.append(tlv_optional)

            # Shuffle TLV sequence in the NFOR body to induce position variance
            rng.shuffle(tlvs)

            msg = header + b"".join(tlvs)

            X.append(msg)
            y.append(cmd_val)
            interaction_metadata.append({
                "session_id": i // 2,
                "direction": "client" if i % 2 == 0 else "server",
                "timestamp": float(i),
            })

        metadata = {
            "dataset_type": "NFOR",
            "true_boundary_B": 7,
            "keyword_tag": "TLV_Type_10",
            "num_clusters": len(cmd_values),
            "interaction_metadata": interaction_metadata,
        }

        return X, np.array(y, dtype=int), metadata


# =====================================================================
# Functional Wrapper Interface (Maintains Backward Compatibility)
# =====================================================================

def generate_generic_for_dataset(
    num_messages: int = 1000, seed: int = 54761161
) -> Tuple[List[bytes], np.ndarray]:
    """Functional wrapper for generating FOR datasets."""
    generator = GenericDatasetGenerator(seed=seed)
    X, y, _ = generator.generate_for_dataset(num_messages=num_messages)
    return X, y


def generate_generic_nfor_dataset(
    num_messages: int = 1000, seed: int = 841561854
) -> Tuple[List[bytes], np.ndarray]:
    """Functional wrapper for generating NFOR datasets."""
    generator = GenericDatasetGenerator(seed=seed)
    X, y, _ = generator.generate_nfor_dataset(num_messages=num_messages)
    return X, y