import struct
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
        seed: int = 42
    ):
        self.num_messages = num_messages
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def generate(self) -> tuple[list[bytes], np.ndarray]:
        X = []
        y = []
        opcodes = [0x01, 0x02, 0x03, 0x04]

        for i in range(self.num_messages):
            opcode = int(self.rng.choice(opcodes))
            magic = 0xABCD
            version = 1
            sequence = i
            
            # 1. Non-monotonic timestamp noise (simulates out-of-order packets)
            if self.rng.random() < self.noise_level:
                timestamp = int(self.rng.integers(0, 10000))
            else:
                timestamp = 1000 + i

                if self.rng.random() < self.noise_level:
                    sequence = int(self.rng.integers(0, 5000))
                else:
                    sequence = i

            # 2. Header binary packing

            header = struct.pack(
                ">HBBII",
                magic,
                version,
                opcode,
                timestamp,
                sequence
            )

            # 3. Dynamic/corrupted TLV noise payload
            if self.rng.random() < self.noise_level:
                # Corrupted/Malformed TLV (random length/tag)
                tlv_tag = int(self.rng.integers(0x80, 0xFF))
                tlv_len = int(self.rng.integers(1, 10))
                tlv_val = self.rng.bytes(tlv_len)
                payload = struct.pack('>BB', tlv_tag, tlv_len) + tlv_val
            else:
                # Valid TLV Option
                tlvs = [
                    struct.pack(">BBB",0x0A,1,opcode)
                ]

                if self.rng.random() < 0.5:
                    data = self.rng.bytes(4)
                    tlvs.append(
                        struct.pack(">BB",0x0B,4)+data
                    )

                self.rng.shuffle(tlvs)

                payload = b"".join(tlvs)

            msg = header + payload
            X.append(msg)
            y.append(opcode)

        return X, np.array(y, dtype=int)