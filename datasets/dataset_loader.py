"""
Dataset loader for evaluating packet-clustering algorithms against
NetPlier-compatible ground truth.

The loader:
  1. Reads classic PCAP *and* PCAPNG captures.
  2. Parses Ethernet/IPv4 TCP, UDP and ICMP.
  3. Applies the protocol-specific preprocessing used by NetPlier
     (``netplier/processing.py``, Netzob 1.0.2, ``importLayer=5``).
  4. Computes NetPlier-compatible direction.
  5. Computes NetPlier-compatible ground-truth message labels.

Message-granularity contract
----------------------------
NetPlier imports traces with::

    PCAPImporter.readFile(filePath=..., importLayer=5)   # layer 3 for ICMP

Netzob 1.0.2 (the version pinned by NetPlier) emits **one message per
TCP/UDP packet payload**; ``mergePacketsInFlow`` defaults to ``False``
and no stream reassembly or protocol framing is performed.  This loader
therefore reproduces that granularity exactly, because the ground-truth
byte offsets below are only meaningful on NetPlier's message set.

Guide item 9 (TCP reassembly) is consequently *not* applied by default.
It is available as an opt-in, off-by-default experiment so the deviation
can be measured rather than assumed.  See ``reassemble_tcp``.

IMPORTANT:
The returned ``y`` values (and the ``gt_*`` metadata keys) are ONLY
evaluation ground truth.  They must never be supplied as input features
to RPKClust.  ``main.py`` projects metadata through an allow-list
before calling ``RPKClust.fit``.
"""

import os
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Capture format constants
# --------------------------------------------------------------------------

# magic -> (endianness, timestamp divisor)
_CLASSIC_PCAP_MAGICS: Dict[bytes, Tuple[str, float]] = {
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000.0),
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000.0),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000.0),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000.0),
}

_PCAPNG_SHB = 0x0A0D0D0A
_PCAPNG_IDB = 0x00000001
_PCAPNG_PB = 0x00000002
_PCAPNG_SPB = 0x00000003
_PCAPNG_EPB = 0x00000006

_PCAPNG_BYTE_ORDER_MAGIC = b"\x4d\x3c\x2b\x1a"  # 0x1A2B3C4D, little-endian

_OPT_ENDOFOPT = 0
_OPT_IF_TSRESOL = 9
_OPT_IF_TSOFFSET = 14

# Link types we can decode (values are the classic DLT_* / LINKTYPE_* ids)
_DLT_NULL = 0
_DLT_EN10MB = 1
_DLT_LINUX_SLL = 113
_DLT_RAW = 101
_DLT_IPV4 = 228
_DLT_IPV6 = 229
_DLT_SLL2 = 276

_SUPPORTED_LINKTYPES = {
    _DLT_NULL,
    _DLT_EN10MB,
    _DLT_LINUX_SLL,
    _DLT_RAW,
    _DLT_IPV4,
    _DLT_SLL2,
}

# IPv4 ethertypes
_ETHERTYPE_IPV4 = 0x0800
_VLAN_ETHERTYPES = (0x8100, 0x88A8, 0x9100)

_IPPROTO_ICMP = 1
_IPPROTO_TCP = 6
_IPPROTO_UDP = 17

_MAX_CAPTURED_PACKET = 16 * 1024 * 1024


class PcapDatasetLoader:
    """
    Load PCAP/PCAPNG application messages and generate NetPlier-compatible GT.

    Supported NetPlier protocols:
        dhcp, dnp3, icmp, modbus, ntp, smb, smb2, tftp, zeroaccess

    The loader intentionally does NOT use:
        - first payload byte as a generic label
        - one-PCAP-one-class labeling
        - a fixed min_length=8 filter
    """

    SUPPORTED_PROTOCOLS = {
        "dhcp",
        "dnp3",
        "icmp",
        "modbus",
        "ntp",
        "smb",
        "smb2",
        "tftp",
        "zeroaccess",
    }

    #: NetPlier's ``Processing.MAX_LEN``.
    MAX_MESSAGE_LENGTH = 500

    def __init__(
        self,
        target_dir: str = "datasets/downloads",
        allow_synthetic_fallback: bool = False,
    ):
        self.target_dir = target_dir
        self.allow_synthetic_fallback = allow_synthetic_fallback

        # Metadata for the most recent extraction.
        self.last_metadata: List[Dict[str, Any]] = []

        # Diagnostics for the most recent extraction.
        self.last_stats: Dict[str, Any] = {}

        os.makedirs(self.target_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_pcap(
        self,
        url: str,
        filename: str,
    ) -> str:
        """Download a PCAP if it is not already cached locally."""

        file_path = os.path.join(self.target_dir, filename)

        if not os.path.exists(file_path):
            print(f"[PcapLoader] Fetching dataset from {url}...")

            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as exc:
                if os.path.exists(file_path):
                    os.remove(file_path)

                raise RuntimeError(
                    f"Unable to download PCAP from {url}"
                ) from exc

            print(f"[PcapLoader] Dataset saved to {file_path}")

        else:
            print(f"[PcapLoader] Using cached dataset at {file_path}")

        return file_path

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract_payloads(
        self,
        pcap_path: str,
        protocol: str,
        min_length: int = 1,
        reassemble_tcp: bool = False,
    ) -> Tuple[List[bytes], np.ndarray]:
        """
        Extract messages and NetPlier-compatible GT labels.

        Parameters
        ----------
        pcap_path:
            Path to a classic PCAP or PCAPNG capture.

        protocol:
            One of the protocols supported by NetPlier.

        min_length:
            Minimum application payload length.

            Default is 1 because NetPlier's benchmark contains short
            protocol messages such as TFTP ACK.  NetPlier itself has no
            minimum; it only skips empty L4 payloads.

        reassemble_tcp:
            Off by default.  When True, TCP payloads are reassembled into
            contiguous streams and split with protocol framing where a
            reliable length field exists.  This DEVIATES from NetPlier
            (Netzob 1.0.2 emits one message per packet) and is provided
            only to measure the difference.

        Returns
        -------
        payloads:
            Application-layer messages after NetPlier-compatible
            preprocessing.

        labels:
            Integer-encoded GT labels.

            The canonical human-readable GT is also available in
            ``last_metadata[i]["gt_label"]``.
        """

        protocol = protocol.lower()

        if protocol not in self.SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported protocol '{protocol}'. "
                f"Supported protocols: {sorted(self.SUPPORTED_PROTOCOLS)}"
            )

        # ----------------------------------------------------------
        # Pass 1: read the capture into raw transport records.
        # ----------------------------------------------------------

        records, capture_stats = self._read_transport_records(
            pcap_path,
            min_length=min_length,
        )

        if not records:
            raise ValueError(
                f"No transport payloads found in {pcap_path}"
            )

        # ----------------------------------------------------------
        # Pass 2: optional TCP stream reassembly.
        # ----------------------------------------------------------

        reassembly_stats: Dict[str, Any] = {}

        if reassemble_tcp and protocol in ("dnp3", "modbus", "smb", "smb2"):
            records, reassembly_stats = self._reassemble_tcp_streams(
                records, protocol
            )

        # ----------------------------------------------------------
        # Pass 3: protocol-specific direction.
        #
        # TFTP has no protocol-level direction field; NetPlier derives
        # it from sessions, which requires the whole record list.
        # ----------------------------------------------------------

        session_directions: Optional[List[str]] = None

        if protocol == "tftp":
            session_directions = self._assign_session_directions(records)

        # ----------------------------------------------------------
        # Pass 4: preprocess, direction, ground truth.
        # ----------------------------------------------------------

        payloads: List[bytes] = []
        metadata: List[Dict[str, Any]] = []

        counters = {
            "packets_read": capture_stats["packets_read"],
            "link_types": capture_stats["link_types"],
            "transport_payloads": len(records),
            "skipped_empty": capture_stats["skipped_empty"],
            "skipped_unsupported": capture_stats["skipped_unsupported"],
            "skipped_truncated": capture_stats["skipped_truncated"],
            "skipped_preprocess": 0,
            "skipped_no_gt": 0,
            "unknown_direction": 0,
            "reassembly": reassembly_stats,
        }

        for index, record in enumerate(records):

            payload = self._preprocess_payload(
                record["payload"],
                protocol,
            )

            if payload is None or len(payload) < min_length:
                counters["skipped_preprocess"] += 1
                continue

            if protocol == "tftp":
                direction = session_directions[index]
            else:
                direction = self._get_netplier_direction(
                    payload,
                    record,
                    protocol,
                )

            if direction is None:
                # Guide item 12: represent as unknown, never silently drop
                # and never fabricate a neutral value.
                direction = "unknown"
                counters["unknown_direction"] += 1

            gt_value = self._get_netplier_gt(
                payload,
                protocol,
            )

            if gt_value is None:
                counters["skipped_no_gt"] += 1
                continue

            gt_label = self._format_gt_label(
                protocol,
                direction,
                gt_value,
            )

            metadata.append(
                {
                    "packet_index": record["packet_index"],

                    "timestamp": record["timestamp"],

                    "protocol": protocol,

                    "transport": record["protocol"],

                    "direction": direction,

                    "gt_value": gt_value,

                    "gt_label": gt_label,

                    "source_ip": record["source_ip"],
                    "source_port": record["source_port"],

                    "destination_ip": record["destination_ip"],
                    "destination_port": record["destination_port"],

                    "session_id": record["session_id"],

                    "payload_length": len(payload),

                    # Useful for debugging/evaluation.
                    "gt_source": "NetPlier",
                }
            )

            payloads.append(payload)

        counters["messages"] = len(payloads)

        self.last_metadata = metadata

        if not payloads:

            if self.allow_synthetic_fallback:
                print(
                    "[PcapLoader] No qualifying messages; "
                    "generating explicit synthetic fallback."
                )

                payloads, labels = self._generate_fallback_traffic(
                    min_length
                )

                self.last_metadata = []
                self.last_stats = counters

                return payloads, labels

            counters["messages"] = 0
            self.last_stats = counters

            raise ValueError(
                f"No qualifying {protocol} messages found in {pcap_path} "
                f"(stats: {counters})"
            )

        # Integer labels are convenience labels only.
        # The canonical GT remains metadata["gt_label"].
        label_names = [
            item["gt_label"]
            for item in metadata
        ]

        label_to_id = {
            label: i
            for i, label in enumerate(
                sorted(set(label_names))
            )
        }

        labels = np.array(
            [
                label_to_id[label]
                for label in label_names
            ],
            dtype=int,
        )

        self.last_stats = counters

        return payloads, labels

    # ------------------------------------------------------------------
    # Metadata API
    # ------------------------------------------------------------------

    def extract_payloads_with_metadata(
        self,
        pcap_path: str,
        protocol: str,
        min_length: int = 1,
        reassemble_tcp: bool = False,
    ) -> Tuple[
        List[bytes],
        np.ndarray,
        List[Dict[str, Any]],
    ]:
        """Extract payloads, GT labels and metadata."""

        payloads, labels = self.extract_payloads(
            pcap_path,
            protocol,
            min_length,
            reassemble_tcp=reassemble_tcp,
        )

        return (
            payloads,
            labels,
            self.last_metadata.copy(),
        )

    # ------------------------------------------------------------------
    # Folder dataset
    # ------------------------------------------------------------------

    def extract_folder_dataset(
        self,
        folder_path: str,
        protocol: str,
        min_length: int = 1,
    ):
        """
        Load all PCAPs in a directory.

        IMPORTANT:
        A PCAP is NOT treated as a class.

        Ground truth comes from the protocol-specific NetPlier
        message-type field.
        """

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(folder)

        pcap_files = sorted(
            list(folder.glob("*.pcap")) + list(folder.glob("*.pcapng"))
        )

        if not pcap_files:
            raise RuntimeError(
                f"No capture files found in {folder}"
            )

        X: List[bytes] = []
        y: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for pcap in pcap_files:

            print(
                f"[PcapLoader] Loading {pcap.name} "
                f"as protocol={protocol}"
            )

            X_part, _, meta_part = (
                self.extract_payloads_with_metadata(
                    str(pcap),
                    protocol,
                    min_length,
                )
            )

            X.extend(X_part)

            y.extend(
                item["gt_label"]
                for item in meta_part
            )

            metadata.extend(meta_part)

        # Keep canonical string labels.
        y_array = np.asarray(y, dtype=object)

        return X, y_array, metadata

    # ------------------------------------------------------------------
    # Capture readers
    # ------------------------------------------------------------------

    @classmethod
    def _iter_capture(
        cls,
        pcap_path: str,
    ) -> Iterator[Tuple[float, int, bytes, int]]:
        """
        Yield ``(timestamp, linktype, packet_bytes, packet_index)``.

        Supports classic PCAP (all four byte-order/us or ns magics) and
        PCAPNG (Section Header, Interface Description, Enhanced Packet,
        Simple Packet and legacy Packet blocks).

        The byte order of a PCAPNG section is established by its byte
        order magic and stays valid until the next Section Header Block.
        """

        with open(pcap_path, "rb") as handle:

            magic = handle.read(4)

            if magic in _CLASSIC_PCAP_MAGICS:
                yield from cls._iter_classic_pcap(
                    handle,
                    *_CLASSIC_PCAP_MAGICS[magic],
                )
                return

            if magic != b"\x0a\x0d\x0d\x0a":
                raise ValueError(
                    "Unsupported capture format: expected classic PCAP "
                    "(a1b2c3d4/d4c3b2a1) or PCAPNG (0a0d0d0a); "
                    f"got magic {magic.hex()}"
                )

            # _iter_pcapng starts at the Section Header Block itself.
            handle.seek(0)

            yield from cls._iter_pcapng(handle)

    @staticmethod
    def _iter_classic_pcap(
        handle,
        endian: str,
        timestamp_divisor: float,
    ) -> Iterator[Tuple[float, int, bytes, int]]:

        header = handle.read(20)

        if len(header) < 20:
            raise ValueError(
                "Invalid PCAP file: header too short."
            )

        linktype = struct.unpack(
            f"{endian}I", header[16:20]
        )[0]

        packet_index = 0

        while True:

            packet_header = handle.read(16)

            if len(packet_header) < 16:
                return

            ts_sec, ts_fraction, incl_len, _orig_len = struct.unpack(
                f"{endian}IIII",
                packet_header,
            )

            if incl_len > _MAX_CAPTURED_PACKET:
                return

            packet = handle.read(incl_len)

            if len(packet) != incl_len:
                # Truncated trailing record: stop cleanly, as NetPlier's
                # pcapy-based reader effectively does.
                return

            packet_index += 1

            yield (
                ts_sec + ts_fraction / timestamp_divisor,
                linktype,
                packet,
                packet_index,
            )

    @staticmethod
    def _iter_pcap_options(
        endian: str,
        body: bytes,
        start: int,
    ):
        """Yield ``(code, value)`` pairs from a PCAPNG option block."""

        position = start
        total = len(body)

        while position + 4 <= total:

            code, length = struct.unpack_from(
                f"{endian}HH", body, position
            )

            position += 4

            if code == _OPT_ENDOFOPT:
                return

            value = body[position:position + length]

            if len(value) != length:
                return

            yield code, value

            position += length + ((4 - (length % 4)) % 4)

    @classmethod
    def _iter_pcapng(
        cls,
        handle,
    ) -> Iterator[Tuple[float, int, bytes, int]]:

        # The first block is always a Section Header Block, whose type is
        # a byte palindrome, so it can be recognised before the section
        # byte order is known.
        endian = "<"
        interfaces: List[Tuple[int, float, float]] = []
        packet_index = 0

        while True:

            block_header = handle.read(8)

            if len(block_header) < 8:
                return

            if block_header[0:4] == b"\x0a\x0d\x0d\x0a":

                byte_order = handle.read(4)

                if len(byte_order) < 4:
                    return

                endian = (
                    "<"
                    if byte_order == _PCAPNG_BYTE_ORDER_MAGIC
                    else ">"
                )

                total_length = struct.unpack(
                    f"{endian}I", block_header[4:8]
                )[0]

                if total_length < 16:
                    return

                # byte_order magic already consumed.
                remainder = handle.read(total_length - 16 + 4)

                if len(remainder) != total_length - 16 + 4:
                    return

                interfaces = []
                packet_index = 0
                continue

            block_type = struct.unpack(
                f"{endian}I", block_header[0:4]
            )[0]

            total_length = struct.unpack(
                f"{endian}I", block_header[4:8]
            )[0]

            if total_length < 12:
                return

            body = handle.read(total_length - 12)

            if len(body) != total_length - 12:
                return

            handle.read(4)  # trailing total length

            if block_type == _PCAPNG_IDB:

                if len(body) < 8:
                    continue

                linktype, _reserved, _snaplen = struct.unpack_from(
                    f"{endian}HHI", body, 0
                )

                divisor = 1_000_000.0
                offset_seconds = 0.0

                for code, value in cls._iter_pcap_options(
                    endian, body, 8
                ):

                    if (
                        code == _OPT_IF_TSRESOL
                        and len(value) == 1
                    ):
                        resolution = value[0]
                        if resolution & 0x80:
                            divisor = float(2 ** (resolution & 0x7F))
                        else:
                            divisor = float(10 ** resolution)

                    elif (
                        code == _OPT_IF_TSOFFSET
                        and len(value) == 8
                    ):
                        offset_seconds = float(
                            struct.unpack(f"{endian}q", value)[0]
                        )

                interfaces.append(
                    (linktype, divisor, offset_seconds)
                )

            elif block_type == _PCAPNG_EPB:

                if len(body) < 20:
                    continue

                (
                    interface_id,
                    ts_high,
                    ts_low,
                    cap_len,
                    _orig_len,
                ) = struct.unpack_from(f"{endian}IIIII", body, 0)

                if interface_id >= len(interfaces):
                    continue

                linktype, divisor, offset_seconds = interfaces[interface_id]

                packet = body[20:20 + cap_len]

                if len(packet) != cap_len:
                    continue

                packet_index += 1

                raw_timestamp = (ts_high << 32) | ts_low

                yield (
                    raw_timestamp / divisor + offset_seconds,
                    linktype,
                    packet,
                    packet_index,
                )

            elif block_type == _PCAPNG_PB:

                if len(body) < 24:
                    continue

                (
                    interface_id,
                    _drops,
                    ts_high,
                    ts_low,
                    cap_len,
                    _orig_len,
                ) = struct.unpack_from(f"{endian}IIIIII", body, 0)

                if interface_id >= len(interfaces):
                    continue

                linktype, divisor, offset_seconds = interfaces[interface_id]

                packet = body[24:24 + cap_len]

                if len(packet) != cap_len:
                    continue

                packet_index += 1

                raw_timestamp = (ts_high << 32) | ts_low

                yield (
                    raw_timestamp / divisor + offset_seconds,
                    linktype,
                    packet,
                    packet_index,
                )

            elif block_type == _PCAPNG_SPB:

                if len(body) < 4 or not interfaces:
                    continue

                orig_len = struct.unpack_from(f"{endian}I", body, 0)[0]

                cap_len = min(orig_len, len(body) - 4)

                packet = body[4:4 + cap_len]

                if cap_len <= 0:
                    continue

                linktype, _divisor, _offset = interfaces[0]

                packet_index += 1

                # Simple Packet Blocks carry no timestamp.
                yield (0.0, linktype, packet, packet_index)

    # ------------------------------------------------------------------
    # Capture -> transport records
    # ------------------------------------------------------------------

    def _read_transport_records(
        self,
        pcap_path: str,
        min_length: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse a capture into transport records.

        Each record describes exactly one application-layer message
        candidate, matching NetPlier's per-packet granularity.
        """

        records: List[Dict[str, Any]] = []

        link_types: Dict[int, int] = defaultdict(int)
        skipped_unsupported = 0
        skipped_truncated = 0
        skipped_empty = 0
        packets_read = 0

        for (
            timestamp,
            linktype,
            packet,
            packet_index,
        ) in self._iter_capture(pcap_path):

            packets_read += 1
            link_types[linktype] += 1

            transport = self._extract_transport_info(
                packet,
                linktype,
            )

            if transport is None:
                skipped_unsupported += 1
                continue

            if transport.get("truncated"):
                skipped_truncated += 1
                continue

            payload = transport["payload"]

            if len(payload) == 0:
                # NetPlier/Netzob: `if len(l4Payload) == 0: return`.
                skipped_empty += 1
                continue

            if len(payload) < min_length:
                skipped_empty += 1
                continue

            records.append(
                {
                    "packet_index": packet_index,
                    "timestamp": timestamp,
                    "linktype": linktype,
                    "protocol": transport["protocol"],
                    "payload": payload,
                    "source_ip": transport["source_ip"],
                    "source_port": transport["source_port"],
                    "destination_ip": transport["destination_ip"],
                    "destination_port": transport["destination_port"],
                    "session_id": transport["session_id"],
                    "ip_header_len": transport["ip_header_len"],
                    "tcp_sequence": transport["tcp_sequence"],
                    "tcp_flags": transport["tcp_flags"],
                    "ip_total_length": transport["ip_total_length"],
                }
            )

        stats = {
            "packets_read": packets_read,
            "link_types": dict(link_types),
            "skipped_unsupported": skipped_unsupported,
            "skipped_truncated": skipped_truncated,
            "skipped_empty": skipped_empty,
        }

        return records, stats

    # ------------------------------------------------------------------
    # Optional TCP reassembly (deviation from NetPlier; opt-in)
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_length(
        buffer: bytes,
        position: int,
        protocol: str,
    ) -> Optional[int]:
        """
        Return the length of the framed message at ``position``.

        Only framing rules that are unambiguous for the given protocol are
        used.  ``None`` means "no reliable framing available here", which
        makes the caller emit the rest of the buffer as one message.
        """

        if protocol == "modbus":
            if position + 6 > len(buffer):
                return None
            length = int.from_bytes(
                buffer[position + 4:position + 6], "big", signed=False
            )
            total = length + 6
            if total <= 6 or position + total > len(buffer):
                return None
            return total

        if protocol in ("smb", "smb2"):
            if position + 8 > len(buffer):
                return None
            signature = b"\xffSMB" if protocol == "smb" else b"\xfeSMB"
            if buffer[position + 4:position + 8] != signature:
                return None
            # NetBIOS Session Service length: 24-bit big-endian at [1:4]
            # in these captures.
            length = int.from_bytes(
                buffer[position + 1:position + 4], "big"
            )
            total = length + 4
            if total <= 4 or position + total > len(buffer):
                return None
            return total

        if protocol == "dnp3":
            if position + 3 > len(buffer):
                return None
            if buffer[position:position + 2] != b"\x05\x64":
                return None
            length = buffer[position + 2]
            total = length + 7
            if total <= 7 or position + total > len(buffer):
                return None
            return total

        return None

    @staticmethod
    def _next_frame_boundary(
        buffer: bytes,
        position: int,
        protocol: str,
    ) -> int:
        """
        Fallback cut point when the declared length does not fit.

        Used only by the opt-in reassembly path.  DNP3 link frames start
        with the 0x0564 magic, so the next magic occurrence is used as a
        boundary; otherwise the rest of the stream is one message.
        """

        if protocol != "dnp3":
            return len(buffer)

        search = buffer.find(b"\x05\x64", position + 3)

        if search == -1:
            return len(buffer)

        return search

    def _reassemble_tcp_streams(
        self,
        records: List[Dict[str, Any]],
        protocol: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Reassemble TCP payloads per 5-tuple and split with protocol framing.

        This DEVIATES from NetPlier, which uses one message per packet.
        Enable it only to measure the impact.
        """

        streams: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

        for record in records:
            key = (
                record["protocol"],
                tuple(sorted(
                    [
                        (record["source_ip"], record["source_port"]),
                        (
                            record["destination_ip"],
                            record["destination_port"],
                        ),
                    ]
                )),
            )
            streams[key].append(record)

        output: List[Dict[str, Any]] = []
        retransmissions = 0
        merged_segments = 0

        for key, members in streams.items():

            if key[0] != "tcp":
                output.extend(members)
                continue

            members = sorted(
                members,
                key=lambda r: (r["timestamp"], r["packet_index"]),
            )

            base_sequence: Optional[int] = None
            buffer = bytearray()
            covered = bytearray()

            for record in members:

                sequence = record["tcp_sequence"]

                if base_sequence is None:
                    base_sequence = sequence

                relative = (sequence - base_sequence) % (1 << 32)

                if relative > (1 << 31):
                    # Out-of-window / older segment: ignore.
                    continue

                if relative + len(covered) > len(covered):
                    covered.extend(
                        b"\x00" * (relative + len(record["payload"]) - len(covered))
                    )

                inserted = False

                for offset, byte in enumerate(record["payload"]):
                    position = relative + offset
                    if covered[position] == 0:
                        buffer[position] = byte
                        covered[position] = 1
                        inserted = True

                if not inserted and len(record["payload"]):
                    retransmissions += 1
                else:
                    merged_segments += 1

            stream_bytes = bytes(buffer)

            position = 0
            while position < len(stream_bytes):

                if covered[position] == 0:
                    position += 1
                    continue

                length = self._frame_length(
                    stream_bytes, position, protocol
                )

                if length is None:
                    end = self._next_frame_boundary(
                        stream_bytes, position, protocol
                    )
                    end = max(end, position + 1)
                    remainder = stream_bytes[position:end]
                else:
                    remainder = stream_bytes[position:position + length]
                    end = position + length

                if not remainder:
                    break

                template = members[0]

                output.append(
                    {
                        **template,
                        "payload": remainder,
                        "from_reassembly": True,
                    }
                )

                position = end

        output.sort(
            key=lambda r: (r["timestamp"], r["packet_index"])
        )

        stats = {
            "streams": len(streams),
            "merged_segments": merged_segments,
            "retransmissions_ignored": retransmissions,
            "messages_out": len(output),
        }

        return output, stats

    # ------------------------------------------------------------------
    # TFTP session direction
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_session_directions(
        records: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Reproduce NetPlier's ``get_msgs_directionlist_by_sessions()``.

        NetPlier builds true sessions from Netzob's ``Session`` objects
        (one session per unique ``ip:port`` endpoint couple) and then
        declares the source of the chronologically first message of each
        session to be the requester.
        """

        groups: Dict[Any, List[int]] = defaultdict(list)

        for index, record in enumerate(records):

            source = (record["source_ip"], record["source_port"])
            destination = (
                record["destination_ip"],
                record["destination_port"],
            )

            key = (
                record["protocol"],
                tuple(sorted([source, destination])),
            )

            groups[key].append(index)

        directions: List[str] = ["unknown"] * len(records)

        for indices in groups.values():

            ordered = sorted(
                indices,
                key=lambda i: (
                    records[i]["timestamp"],
                    records[i]["packet_index"],
                ),
            )

            initiator = (
                records[ordered[0]]["source_ip"],
                records[ordered[0]]["source_port"],
            )

            for index in ordered:

                record = records[index]
                current = (
                    record["source_ip"],
                    record["source_port"],
                )

                if current == initiator:
                    directions[index] = "request"
                else:
                    directions[index] = "response"

        return directions

    # ------------------------------------------------------------------
    # Transport parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _format_endpoint(ip: Optional[str], port: Optional[int]) -> str:
        if ip is None:
            return "*"
        if port is None:
            return str(ip)
        return f"{ip}:{port}"

    @classmethod
    def _build_session_id(
        cls,
        protocol: str,
        source_ip: Optional[str],
        source_port: Optional[int],
        destination_ip: Optional[str],
        destination_port: Optional[int],
    ) -> str:
        """
        Build a bidirectional session identifier.

        Endpoints are sorted so that both halves of a conversation map
        to the same session.  ICMP has no ports, so the identifier falls
        back to the IP pair only.
        """

        endpoints = sorted(
            [
                cls._format_endpoint(source_ip, source_port),
                cls._format_endpoint(destination_ip, destination_port),
            ]
        )

        return f"{protocol}:{endpoints[0]}-{endpoints[1]}"

    @classmethod
    def _strip_link_layer(
        cls,
        packet: bytes,
        linktype: int,
    ) -> Optional[Tuple[bytes, Optional[int]]]:
        """
        Remove the link layer.

        Returns ``(network_layer_bytes, ethertype)``.
        """

        if linktype == _DLT_EN10MB:

            if len(packet) < 14:
                return None

            ethertype = int.from_bytes(packet[12:14], "big")
            offset = 14

            while ethertype in _VLAN_ETHERTYPES:

                if len(packet) < offset + 4:
                    return None

                ethertype = int.from_bytes(
                    packet[offset + 2:offset + 4], "big"
                )
                offset += 4

            return packet[offset:], ethertype

        if linktype == _DLT_RAW:

            if not packet:
                return None

            version = packet[0] >> 4
            ethertype = (
                _ETHERTYPE_IPV4 if version == 4 else 0x86DD
            )
            return packet, ethertype

        if linktype == _DLT_IPV4:
            return packet, _ETHERTYPE_IPV4

        if linktype == _DLT_NULL:

            if len(packet) < 4:
                return None

            family = struct.unpack("<I", packet[0:4])[0]

            if family > 0xFFFF:
                family = struct.unpack(">I", packet[0:4])[0]

            ethertype = (
                _ETHERTYPE_IPV4 if family == 2 else 0x86DD
            )
            return packet[4:], ethertype

        if linktype == _DLT_LINUX_SLL:

            if len(packet) < 16:
                return None

            ethertype = int.from_bytes(packet[14:16], "big")
            return packet[16:], ethertype

        if linktype == _DLT_SLL2:

            if len(packet) < 20:
                return None

            ethertype = int.from_bytes(packet[0:2], "big")
            return packet[20:], ethertype

        return None

    @classmethod
    def _extract_transport_info(
        cls,
        packet: bytes,
        linktype: int = _DLT_EN10MB,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse the link, IPv4 and transport layers.

        For ICMP the returned ``payload`` is the ICMP message (the IPv4
        header has already been removed), matching NetPlier's
        ``layer=3`` import followed by its IHL-based header strip.
        """

        stripped = cls._strip_link_layer(packet, linktype)

        if stripped is None:
            return None

        network_layer, ethertype = stripped

        if ethertype != _ETHERTYPE_IPV4:
            # NetPlier/Netzob supports IP only.
            return None

        if len(network_layer) < 20:
            return None

        version_ihl = network_layer[0]

        if version_ihl >> 4 != 4:
            return None

        ip_header_len = (version_ihl & 0x0F) * 4

        if ip_header_len < 20:
            return None

        if len(network_layer) < ip_header_len:
            return None

        total_length = int.from_bytes(
            network_layer[2:4], "big"
        )

        ip_end = (
            min(len(network_layer), total_length)
            if total_length
            else len(network_layer)
        )

        if ip_end < ip_header_len:
            return None

        fragment_field = int.from_bytes(
            network_layer[6:8], "big"
        )
        is_fragment = bool(fragment_field & 0x1FFF) or bool(
            fragment_field & 0x2000
        )

        protocol_number = network_layer[9]

        protocol = {
            _IPPROTO_ICMP: "icmp",
            _IPPROTO_TCP: "tcp",
            _IPPROTO_UDP: "udp",
        }.get(protocol_number)

        if protocol is None:
            return None

        source_ip = ".".join(
            str(value) for value in network_layer[12:16]
        )
        destination_ip = ".".join(
            str(value) for value in network_layer[16:20]
        )

        transport_offset = ip_header_len

        tcp_sequence = 0
        tcp_flags = 0
        source_port: Optional[int] = None
        destination_port: Optional[int] = None

        if protocol == "icmp":
            # NetPlier imports ICMP at layer 3, i.e. the message data is
            # the whole IPv4 packet, then removes the IPv4 header using
            # the IHL field before extracting the keyword.
            payload = network_layer[ip_header_len:ip_end]

            session_id = cls._build_session_id(
                "icmp",
                source_ip,
                None,
                destination_ip,
                None,
            )

            return {
                "payload": payload,
                "protocol": "icmp",
                "source_ip": source_ip,
                "source_port": None,
                "destination_ip": destination_ip,
                "destination_port": None,
                "session_id": session_id,
                "ip_header_len": ip_header_len,
                "tcp_sequence": 0,
                "tcp_flags": 0,
                "ip_total_length": total_length,
                "truncated": False,
            }

        if is_fragment:
            # Non-first fragments carry no transport header.
            return None

        if protocol == "tcp":
            min_header_len = 20
        else:
            min_header_len = 8

        if ip_end < (transport_offset + min_header_len):
            return None

        source_port = int.from_bytes(
            network_layer[transport_offset:transport_offset + 2],
            "big",
        )
        destination_port = int.from_bytes(
            network_layer[
                transport_offset + 2:transport_offset + 4
            ],
            "big",
        )

        if protocol == "tcp":

            header_len = (
                network_layer[transport_offset + 12] >> 4
            ) * 4

            if header_len < 20:
                return None

            if ip_end < (transport_offset + header_len):
                return None

            tcp_sequence = int.from_bytes(
                network_layer[
                    transport_offset + 4:transport_offset + 8
                ],
                "big",
            )
            tcp_flags = network_layer[transport_offset + 13]

            payload = network_layer[
                transport_offset + header_len:ip_end
            ]

        else:

            udp_length = int.from_bytes(
                network_layer[
                    transport_offset + 4:transport_offset + 6
                ],
                "big",
            )

            payload_end = ip_end

            if udp_length >= min_header_len:
                payload_end = min(
                    ip_end, transport_offset + udp_length
                )

            payload = network_layer[
                transport_offset + min_header_len:payload_end
            ]

        session_id = cls._build_session_id(
            protocol,
            source_ip,
            source_port,
            destination_ip,
            destination_port,
        )

        return {
            "payload": payload,
            "protocol": protocol,
            "source_ip": source_ip,
            "source_port": source_port,
            "destination_ip": destination_ip,
            "destination_port": destination_port,
            "session_id": session_id,
            "ip_header_len": ip_header_len,
            "tcp_sequence": tcp_sequence,
            "tcp_flags": tcp_flags,
            "ip_total_length": total_length,
            "truncated": bool(
                total_length and len(network_layer) < total_length
            ),
        }

    # ------------------------------------------------------------------
    # NetPlier preprocessing
    # ------------------------------------------------------------------

    @classmethod
    def _preprocess_payload(
        cls,
        payload: bytes,
        protocol: str,
    ) -> Optional[bytes]:
        """
        Apply the protocol-specific preprocessing performed by NetPlier.

        NetPlier's ``Processing.import_messages()`` performs the
        protocol-specific step first and then applies
        ``MAX_LEN = 500`` to every remaining message.
        """

        max_len = cls.MAX_MESSAGE_LENGTH

        # SMB: keep only frames carrying the SMB signature.
        if protocol == "smb":

            if len(payload) < 8:
                return None

            if payload[4:8] != b"\xffSMB":
                return None

            return payload[:max_len]

        # SMB2: keep only frames carrying the SMB2 signature.
        if protocol == "smb2":

            if len(payload) < 8:
                return None

            if payload[4:8] != b"\xfeSMB":
                return None

            return payload[:max_len]

        # Modbus: some segments carry more than one MBAP frame.
        if protocol == "modbus":

            if len(payload) < 6:
                return None

            length = int.from_bytes(
                payload[4:6],
                byteorder="big",
                signed=True,
            )

            if length + 6 <= 0:
                return None

            expected_length = length + 6

            if len(payload) != expected_length:
                payload = payload[:expected_length]

            return payload[:max_len]

        # ZeroAccess: decrypt before the keyword is read.
        if protocol == "zeroaccess":

            if len(payload) < 4:
                return None

            return cls._decrypt_zeroaccess(payload)[:max_len]

        # DHCP, DNP3, ICMP, NTP and TFTP need no extra preprocessing.
        return payload[:max_len]

    @staticmethod
    def _decrypt_zeroaccess(
        encrypted: bytes,
    ) -> bytes:
        """
        Reproduce NetPlier's ZeroAccess decryption.
        """

        if len(encrypted) < 4:
            return b""

        crc32 = struct.unpack("<I", encrypted[0:4])[0]

        if crc32 == 0:
            return encrypted

        key = 0x66747032
        result = []

        for i in range(0, len(encrypted), 4):

            # NetPlier stops on `i + 4 >= len(...)`.
            if (i + 4) >= len(encrypted):
                break

            sub_data = struct.unpack(
                "<I", encrypted[i:i + 4]
            )[0]

            decrypted = sub_data ^ key

            result.append(
                struct.pack("<I", decrypted).hex()
            )

            key = (
                ((key << 1) & 0xFFFFFFFF)
                | (key >> 31)
            )

        return bytes.fromhex("".join(result))

    # ------------------------------------------------------------------
    # NetPlier direction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_netplier_direction(
        payload: bytes,
        record: Dict[str, Any],
        protocol: str,
    ) -> Optional[str]:
        """
        Return "request", "response" or None (unknown).

        Mirrors NetPlier's ``get_msg_direction_by_specification``.
        TFTP is handled separately by ``_assign_session_directions``.
        """

        if not payload:
            return None

        if protocol == "dhcp":

            if payload[0] == 1:
                return "request"

            if payload[0] == 2:
                return "response"

            return None

        if protocol == "dnp3":

            if len(payload) < 4:
                return None

            direction_bit = (payload[3] >> 7) & 0x01

            return "request" if direction_bit == 1 else "response"

        if protocol == "icmp":

            if len(payload) < 1:
                return None

            # Applied after the IPv4 header has been removed.
            icmp_type = payload[0]

            if icmp_type in (8, 13, 15, 17, 10):
                return "request"

            if icmp_type in (0, 3, 4, 5, 11, 12, 14, 16, 18, 9):
                return "response"

            return None

        if protocol == "modbus":

            source_port = record.get("source_port")
            destination_port = record.get("destination_port")

            if source_port == 502:
                return "response"

            if destination_port == 502:
                return "request"

            return None

        if protocol == "ntp":

            mode = payload[0] & 0x07

            if mode in (1, 3, 5):
                return "request"

            if mode in (2, 4, 6):
                return "response"

            return None

        if protocol == "smb":

            if len(payload) <= 13:
                return None

            smb_flag = payload[13]

            direction = smb_flag & 0x80

            if direction == 0:
                return "request"

            if direction == 128:
                return "response"

            return None

        if protocol == "smb2":

            if len(payload) < 24:
                return None

            flags = struct.unpack("<I", payload[20:24])[0]

            direction = flags & 0x1

            if direction == 0:
                return "request"

            return "response"

        if protocol == "zeroaccess":

            if len(payload) < 8:
                return None

            value = payload[7]

            if value == ord("g"):
                return "request"

            if value in (ord("r"), ord("n")):
                return "response"

            return None

        return None

    # ------------------------------------------------------------------
    # NetPlier ground truth
    # ------------------------------------------------------------------

    @staticmethod
    def _get_netplier_gt(
        payload: bytes,
        protocol: str,
    ) -> Optional[Any]:
        """
        Extract the exact GT field used by NetPlier's
        ``Processing.get_true_keyword``.

        This is evaluation ground truth, NOT an input to RPKClust.
        """

        if protocol == "dhcp":

            if len(payload) < 243:
                return None

            return payload[242:243].hex()

        if protocol == "dnp3":

            if len(payload) < 13:
                return None

            return payload[12:13].hex()

        if protocol == "icmp":

            if len(payload) < 2:
                return None

            return payload[0:2].hex()

        if protocol == "modbus":

            if len(payload) < 8:
                return None

            return payload[7:8].hex()

        if protocol == "ntp":

            if len(payload) < 1:
                return None

            return payload[0] & 0x07

        if protocol == "smb":

            if len(payload) < 9:
                return None

            # NetPlier: message.data[4 + 4]
            return payload[4 + 4:4 + 5].hex()

        if protocol == "smb2":

            if len(payload) < 18:
                return None

            # NetPlier: struct.unpack("<H", data[4 + 12:4 + 12 + 2])[0]
            return struct.unpack(
                "<H", payload[4 + 12:4 + 12 + 2]
            )[0]

        if protocol == "tftp":

            if len(payload) < 2:
                return None

            return payload[0:2].hex()

        if protocol == "zeroaccess":

            if len(payload) < 8:
                return None

            return payload[4:8].hex()

        return None

    @staticmethod
    def _format_gt_label(
        protocol: str,
        direction: str,
        gt_value: Any,
    ) -> str:
        """
        Canonical label used only for evaluation.

        Direction is included because NetPlier evaluates request and
        response separately.
        """

        return f"{direction}:{protocol}:{gt_value}"

    # ------------------------------------------------------------------
    # Legacy transport helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_transport_payload(
        packet: bytes,
        linktype: int = _DLT_EN10MB,
    ) -> Optional[bytes]:

        transport = PcapDatasetLoader._extract_transport_info(
            packet, linktype
        )

        return (
            None
            if transport is None
            else transport["payload"]
        )

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_fallback_traffic(
        min_length: int,
    ) -> Tuple[
        List[bytes],
        np.ndarray,
    ]:

        rng = np.random.default_rng(42)

        payloads = []
        labels = []

        for _ in range(300):

            msg_type = int(
                rng.choice([0x01, 0x02, 0x05])
            )

            body = (
                bytes(
                    [
                        msg_type,
                        0x00,
                        0x04,
                    ]
                )
                + rng.bytes(
                    max(16, min_length - 3)
                )
            )

            payloads.append(body)
            labels.append(msg_type)

        return (
            payloads,
            np.asarray(labels, dtype=int),
        )