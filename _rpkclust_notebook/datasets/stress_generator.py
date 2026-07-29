"""
RPKClust Protocol Stress Generator (stress_generator.py)
Generates noisy, corrupted binary message traces to stress-test 
boundary identification and TLV candidate extraction under real-world noise.
"""

import struct
from typing import List, Tuple, Dict, Any
import numpy as np


class BinaryProtocolStressGenerator:
    """
    Generates noisy binary message traces (list[bytes]) to stress-test 
    RPKClust's boundary identification and TLV candidate extraction.
    """

    def __init__(
        self,
        num_messages: int = 1000,
        noise_level: float = 0.2,
        seed: int = 42,
    ):
        if not isinstance(num_messages, int) or isinstance(num_messages, bool) or num_messages < 0:
            raise ValueError("num_messages must be a non-negative integer")
        if not isinstance(noise_level, (int, float)) or not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0.0 and 1.0")
        self.num_messages = num_messages
        self.noise_level = float(noise_level)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[List[bytes], np.ndarray]:
        """
        Generates noisy message list and target labels.
        (Backward-compatible 2-tuple return).
        """
        X, y, _ = self.generate_with_metadata()
        return X, y

    def generate_with_metadata(
        self,
    ) -> Tuple[List[bytes], np.ndarray, Dict[str, Any]]:
        """
        Generates noisy message traces along with ground-truth metadata 
        for evaluation reporting.
        """
        X: List[bytes] = []
        y: List[int] = []
        interaction_metadata: List[Dict[str, Any]] = []
        opcodes = [0x01, 0x02, 0x03, 0x04]

        for i in range(self.num_messages):
            opcode = int(self.rng.choice(opcodes)) if i % 2 == 0 else y[-1]
            magic = 0xABCD
            version = 1
            sequence = i

            # 1. Non-monotonic timestamp & sequence noise
            if self.rng.random() < self.noise_level:
                timestamp = int(self.rng.integers(0, 10000))
            else:
                timestamp = 1000 + i

            if self.rng.random() < self.noise_level:
                sequence = int(self.rng.integers(0, 5000))
            else:
                sequence = i

            # 2. Fixed Header binary packing (12 Bytes total)
            # Offset 0..1: Magic (2B)
            # Offset 2   : Version (1B)
            # Offset 3   : OpCode (1B) -> Target Keyword Offset = 3
            # Offset 4..7: Timestamp (4B)
            # Offset 8..11: Sequence (4B)
            header = struct.pack(
                ">HBBII",
                magic,
                version,
                opcode,
                timestamp,
                sequence,
            )

            # 3. Dynamic / Corrupted TLV noise payload (NFOR portion)
            if self.rng.random() < self.noise_level:
                # Corrupted/Malformed TLV (random length/tag)
                tlv_tag = int(self.rng.integers(0x80, 0xFF))
                tlv_len = int(self.rng.integers(1, 10))
                tlv_val = self.rng.bytes(tlv_len)
                payload = struct.pack(">BB", tlv_tag, tlv_len) + tlv_val
            else:
                # Valid TLV Options
                tlvs = [struct.pack(">BBB", 0x0A, 1, opcode)]

                if self.rng.random() < 0.5:
                    data = self.rng.bytes(4)
                    tlvs.append(struct.pack(">BB", 0x0B, 4) + data)

                self.rng.shuffle(tlvs)
                payload = b"".join(tlvs)

            msg = header + payload
            X.append(msg)
            y.append(opcode)
            interaction_metadata.append({
                "session_id": i // 2,
                "direction": "client" if i % 2 == 0 else "server",
                "timestamp": float(i),
            })

        metadata = {
            "dataset_type": "Binary_Stress",
            "true_boundary_B": 12,        # Fixed header length before variable TLVs
            "true_keyword_offset": 3,     # Opcode byte location
            "noise_level": self.noise_level,
            "num_clusters": len(opcodes),
            "interaction_metadata": interaction_metadata,
        }

        return X, np.array(y, dtype=int), metadata


# =====================================================================
# Functional Wrapper
# =====================================================================

def generate_stress_dataset(
    num_messages: int = 1000, noise_level: float = 0.2, seed: int = 42
) -> Tuple[List[bytes], np.ndarray]:
    """Functional helper to generate binary stress dataset directly."""
    generator = BinaryProtocolStressGenerator(
        num_messages=num_messages, noise_level=noise_level, seed=seed
    )
    return generator.generate()
