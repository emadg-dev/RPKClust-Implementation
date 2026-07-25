"""
RPKClust Generic Dataset Generator (dataset_generator.py / generate_data.py)
Generates generic binary message traces for FOR and NFOR benchmark evaluation.
"""

import struct
import numpy as np


def generate_generic_for_dataset(num_messages=200, seed=42):
    """
    Generates generic Fixed-Offset Region (FOR) binary message trace.
    Offset 0: OpCode (Keyword determining cluster label)
    Offset 1-4: Sequential Timestamp
    Offset 5-8: Random Payload
    """
    np.random.seed(seed)
    X = []
    y = []

    opcodes = [0x01, 0x02, 0x03, 0x04]

    for i in range(num_messages):
        magic = 0xABCD
        version = 1
        opcode = int(np.random.choice(opcodes))
        timestamp = 1000 + i
        sequence = i
        # Explicitly specify 64-bit integer type to accommodate 32-bit unsigned upper bound (0xFFFFFFFF)
        # random_payload = int(np.random.randint(0, 0xFFFFFFFF, dtype=np.int64))
        payload = np.random.bytes(np.random.randint(4,17))

        # Binary packing: OpCode (1B), Timestamp (4B), Random Payload (4B)
        # msg = struct.pack('>BI I', opcode, timestamp, random_payload)
        msg = (
            struct.pack(">HBBII",
                magic,
                version,
                opcode,
                timestamp,
                sequence
            )
            + payload
        )
        X.append(msg)
        y.append(opcode)

    return X, np.array(y, dtype=int)


def generate_generic_nfor_dataset(num_messages=200, seed=42):
    """
    Generates a generic NFOR TLV dataset.

    Header (Boundary B = 7):
        Magic      : 2 Bytes
        Version    : 1 Byte
        Timestamp  : 4 Bytes

    Body:
        TLV Command (required)
        TLV Data (required)
        TLV Optional (30% probability)
    """
    np.random.seed(seed)

    X = []
    y = []

    cmd_values = [0x10, 0x20, 0x30, 0x40]

    for i in range(num_messages):

        cmd_val = int(np.random.choice(cmd_values))

        # ---------- Header ----------
        magic = 0xABCD
        version = 1
        timestamp = 2000 + i

        header = struct.pack(">HBI", magic, version, timestamp)

        # ---------- Required Command TLV ----------
        tlv_cmd = struct.pack(">BBB", 0x0A, 1, cmd_val)

        # ---------- Required Data TLV ----------
        data_len = np.random.randint(2, 9)      # 2~8 bytes
        data = np.random.bytes(data_len)
        tlv_data = (
            struct.pack(">BB", 0x0B, data_len)
            + data
        )

        tlvs = [tlv_cmd, tlv_data]

        # ---------- Optional TLV ----------
        if np.random.rand() < 0.30:
            opt_len = np.random.randint(1, 5)
            opt_data = np.random.bytes(opt_len)

            tlv_optional = (
                struct.pack(">BB", 0x0C, opt_len)
                + opt_data
            )

            tlvs.append(tlv_optional)

        # ---------- Random TLV Order ----------
        np.random.shuffle(tlvs)

        msg = header + b"".join(tlvs)

        X.append(msg)
        y.append(cmd_val)

    return X, np.array(y, dtype=int)