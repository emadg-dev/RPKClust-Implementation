"""
Dataset loader for fetching and parsing real-world network packet captures (.pcap).
"""

import os
from pathlib import Path
import urllib.request
import struct
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder


class PcapDatasetLoader:
    """
    Downloads and extracts raw application layer messages from classic PCAP files.
    Supports Ethernet/IPv4 TCP and UDP frames using only the standard library.
    """

    # DEFAULT_URL = "https://raw.githubusercontent.com/wireshark/wireshark/master/test/captures/dhcp.pcap"
    DEFAULT_URL = "https://mcfp.felk.cvut.cz/dataset/CTU-Malware-Capture-Botnet-42/botnet-capture-20110810-neris.pcap](https://mcfp.felk.cvut.cz/dataset/CTU-Malware-Capture-Botnet-42/botnet-capture-20110810-neris.pcap"

    def __init__(self, target_dir: str = "datasets/downloads", allow_synthetic_fallback: bool = False):
        self.target_dir = target_dir
        self.allow_synthetic_fallback = allow_synthetic_fallback
        # Per-payload metadata from the most recent extraction. It aligns
        # positionally with the messages returned by extract_payloads().
        self.last_metadata: List[Dict[str, Any]] = []
        os.makedirs(self.target_dir, exist_ok=True)

    def download_pcap(self, url: str = DEFAULT_URL, filename= "sample_protocol.pcap") -> str:
        """Downloads a PCAP file if it does not already exist locally."""
        file_path = os.path.join(self.target_dir, filename)
        if not os.path.exists(file_path):
            print(f"[PcapLoader] Fetching dataset from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as exc:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise RuntimeError(f"Unable to download PCAP from {url}") from exc
            print(f"[PcapLoader] Dataset saved to {file_path}")
        else:
            print(f"[PcapLoader] Using cached dataset at {file_path}")
        return file_path

    def extract_payloads(self, pcap_path: str, min_length: int = 8) -> Tuple[List[bytes], np.ndarray]:
        """
        Extracts UDP/TCP transport payloads (raw application binary messages) 
        from a global-header PCAP file.
        """
        payloads: List[bytes] = []
        labels: List[int] = []
        metadata: List[Dict[str, Any]] = []
        session_initiators: Dict[str, Tuple[str, int]] = {}

        with open(pcap_path, "rb") as f:
            header = f.read(24)
            if len(header) < 24:
                raise ValueError("Invalid PCAP file: Header too short.")

            magic_number = header[:4]
            # Classic PCAP supports microsecond and nanosecond timestamps.
            if magic_number == b"\xa1\xb2\xc3\xd4":
                endian, timestamp_scale = ">", 1_000_000
            elif magic_number == b"\xa1\xb2\x3c\x4d":
                endian, timestamp_scale = ">", 1_000_000_000
            elif magic_number == b"\xd4\xc3\xb2\xa1":
                endian, timestamp_scale = "<", 1_000_000
            elif magic_number == b"\x4d\x3c\xb2\xa1":
                endian, timestamp_scale = "<", 1_000_000_000
            else:
                raise ValueError("Unsupported capture format; expected a classic PCAP file")
            network = struct.unpack(f"{endian}I", header[20:24])[0]
            if network != 1:  # DLT_EN10MB (Ethernet)
                raise ValueError(f"Unsupported PCAP link type: {network}; only Ethernet is supported")

            while True:
                packet_hdr = f.read(16)
                if len(packet_hdr) < 16:
                    break
                
                ts_sec, ts_fraction, incl_len, _ = struct.unpack(f"{endian}IIII", packet_hdr)
                pkt_data = f.read(incl_len)
                if len(pkt_data) != incl_len:
                    raise ValueError("Invalid PCAP file: truncated packet data")
                transport = self._extract_transport_info(pkt_data)
                if transport is not None and len(transport["payload"]) >= min_length:
                    payload = transport["payload"]
                    session_id = transport["session_id"]
                    source = (transport["source_ip"], transport["source_port"])
                    if session_id not in session_initiators:
                        session_initiators[session_id] = self._infer_initiator(transport)
                    direction = (
                        "client"
                        if source == session_initiators[session_id]
                        else "server"
                    )
                    payloads.append(payload)
                    metadata.append({
                        "session_id": session_id,
                        "direction": direction,
                        "timestamp": ts_sec + ts_fraction / timestamp_scale,
                        "source_ip": transport["source_ip"],
                        "source_port": transport["source_port"],
                        "destination_ip": transport["destination_ip"],
                        "destination_port": transport["destination_port"],
                        "protocol": transport["protocol"],
                    })
                    # Labels are only a convenience for benchmark captures, not protocol truth.
                    labels.append(int(payload[0]))

        if not payloads:
            if self.allow_synthetic_fallback:
                print("[PcapLoader] No qualifying payloads; generating explicit synthetic fallback traffic.")
                payloads, labels_array = self._generate_fallback_traffic(min_length)
                self.last_metadata = []
                return payloads, labels_array
            raise ValueError("No TCP or UDP application payloads meeting min_length were found")

        self.last_metadata = metadata
        return payloads, np.array(labels, dtype=int)

    def extract_payloads_with_metadata(
        self, pcap_path: str, min_length: int = 8
    ) -> Tuple[List[bytes], np.ndarray, List[Dict[str, Any]]]:
        """Extract payloads and their interaction metadata in one call."""
        payloads, labels = self.extract_payloads(pcap_path, min_length)
        return payloads, labels, self.last_metadata.copy()


    def extract_folder_dataset(self, folder_path: str):
        """
        Loads every PCAP inside a folder.

        Each PCAP file represents one class.

        Returns
        -------
        X : list[bytes]
        y : np.ndarray
        metadata : list[dict]
        """

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(folder)

        pcap_files = sorted(folder.glob("*.pcap"))

        if not pcap_files:
            raise RuntimeError(f"No PCAP files found in {folder}")

        X = []
        y = []
        metadata = []

        for pcap in pcap_files:

            class_name = pcap.stem

            # print(f"Loading {class_name}")

            X_part, _, meta_part = self.extract_payloads_with_metadata(
                str(pcap)
            )

            X.extend(X_part)

            y.extend([class_name] * len(X_part))

            if meta_part:
                metadata.extend(meta_part)

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        print("\nClasses:")

        for i, c in enumerate(encoder.classes_):
            print(f"{i} -> {c}")

        return X, y, metadata

    @staticmethod
    def _extract_transport_payload(packet: bytes) -> Optional[bytes]:
        """Return a TCP or UDP payload from an Ethernet/IPv4 frame, if present."""
        transport = PcapDatasetLoader._extract_transport_info(packet)
        return None if transport is None else transport["payload"]

    @staticmethod
    def _extract_transport_info(packet: bytes) -> Optional[Dict[str, Any]]:
        """Parse an Ethernet/IPv4 TCP or UDP frame and retain flow details."""
        if len(packet) < 14:
            return None
        ethertype = int.from_bytes(packet[12:14], "big")
        offset = 14
        while ethertype in (0x8100, 0x88A8):
            if len(packet) < offset + 4:
                return None
            ethertype = int.from_bytes(packet[offset + 2:offset + 4], "big")
            offset += 4
        if ethertype != 0x0800 or len(packet) < offset + 20:
            return None
        version_ihl = packet[offset]
        if version_ihl >> 4 != 4:
            return None
        ip_header_len = (version_ihl & 0x0F) * 4
        if ip_header_len < 20 or len(packet) < offset + ip_header_len:
            return None
        total_length = int.from_bytes(packet[offset + 2:offset + 4], "big")
        ip_end = min(len(packet), offset + total_length) if total_length else len(packet)
        protocol_number = packet[offset + 9]
        protocol = {6: "tcp", 17: "udp"}.get(protocol_number)
        if protocol is None:
            return None
        transport_offset = offset + ip_header_len
        min_header_len = 20 if protocol == "tcp" else 8
        if ip_end < transport_offset + min_header_len:
            return None
        source_ip = ".".join(str(value) for value in packet[offset + 12:offset + 16])
        destination_ip = ".".join(str(value) for value in packet[offset + 16:offset + 20])
        source_port = int.from_bytes(packet[transport_offset:transport_offset + 2], "big")
        destination_port = int.from_bytes(packet[transport_offset + 2:transport_offset + 4], "big")
        tcp_syn = False
        tcp_ack = False
        if protocol == "tcp":
            header_len = (packet[transport_offset + 12] >> 4) * 4
            if header_len < 20 or ip_end < transport_offset + header_len:
                return None
            payload_offset = transport_offset + header_len
            flags = packet[transport_offset + 13]
            tcp_syn, tcp_ack = bool(flags & 0x02), bool(flags & 0x10)
        else:
            payload_offset = transport_offset + 8
        endpoints = sorted(((source_ip, source_port), (destination_ip, destination_port)))
        session_id = f"{protocol}:{endpoints[0][0]}:{endpoints[0][1]}-{endpoints[1][0]}:{endpoints[1][1]}"
        return {
            "payload": packet[payload_offset:ip_end],
            "source_ip": source_ip,
            "source_port": source_port,
            "destination_ip": destination_ip,
            "destination_port": destination_port,
            "protocol": protocol,
            "session_id": session_id,
            "tcp_syn": tcp_syn,
            "tcp_ack": tcp_ack,
        }

    @staticmethod
    def _infer_initiator(transport: Dict[str, Any]) -> Tuple[str, int]:
        """Infer the client endpoint without fabricating missing capture facts."""
        source = (transport["source_ip"], transport["source_port"])
        destination = (transport["destination_ip"], transport["destination_port"])
        if transport["protocol"] == "tcp" and transport["tcp_syn"] and not transport["tcp_ack"]:
            return source
        # For UDP, and TCP captures starting after the handshake, a privileged
        # port is usually the service endpoint. Otherwise use first observed
        # direction and retain the resulting inference in the documentation.
        if source[1] <= 1024 < destination[1]:
            return destination
        return source

    @staticmethod
    def _generate_fallback_traffic(min_length: int) -> Tuple[List[bytes], np.ndarray]:
        """Generate deterministic demo traffic when explicitly requested."""
        rng = np.random.default_rng(42)
        payloads, labels = [], []
        for _ in range(300):
            msg_type = int(rng.choice([0x01, 0x02, 0x05]))
            body = bytes([msg_type, 0x00, 0x04]) + rng.bytes(max(16, min_length - 3))
            payloads.append(body)
            labels.append(msg_type)
        return payloads, np.array(labels, dtype=int)