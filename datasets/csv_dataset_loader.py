"""
Dataset loader for the **CNNPRE** pre-processed CSV benchmark.

The CNNPRE repository (Garshasbi & Teimouri, *CNNPRE: A CNN-Based Protocol
Reverse Engineering Method*, IEEE Access 2023) ships one CSV per protocol
under ``CNNPRE/Data/``::

    ARP1275-hex.csv   DHCP921-hex.csv   DNS4876-hex.csv   FTP1620-hex.csv
    GSM5000-hex.csv   HTTP1963-hex.csv  ICMP3231-hex.csv  NBNS2642-hex.csv
    NTP1043-hex.csv   POP1080-hex.csv   SMTP1172-hex.csv  SYSLOG1792-hex.csv

Schema (4 columns, header row present)::

    direction,type,hex,Full
    1,0000000100000011,01010600573fa192...,0011000100000001...

``direction``
    A capture-side flag, either ``0`` or ``1``.

``type``
    **The ground-truth message type** -- i.e. the isolated message-type
    field of the protocol.  Its rendering is heterogeneous: usually a
    *bit string* of the field's bytes (e.g. FTP ``001100100011001000110000``
    == ``0x32 0x32 0x30`` == ``"220"``), but for some protocols it is the
    literal text (HTTP ``POST``).  It is used verbatim as the label.

``hex``
     The raw message bytes, hex encoded.  This is the ``X`` payload.

``Full``
    The same message rendered as a one-character-per-bit string.  It is
    the input to CNNPRE's CNN and is **not needed here**, so it is never
    materialised.

This loader returns exactly the same tuple shape as
:class:`datasets.dataset_loader.PcapDatasetLoader`, so ``main.py`` can
treat both sources identically::

    X        : list[bytes]              application messages, fed to RPKClust
    y        : np.ndarray[int]          integer-encoded ground-truth types
    metadata : list[dict]               per-message interaction metadata

Metadata contract
-----------------
RPKClust consumes only ``main.RPKCLUST_METADATA_FIELDS`` (timestamp,
protocol, transport, direction, IP/port endpoints, session_id,
packet_index, payload_length).  The remaining ``gt_*`` keys are evaluation
ground truth and are stripped by ``main.project_metadata_for_rpkclust``
before ``RPKClust.fit`` is called -- the same isolation rule that applies
to the PCAP path.

Fields the CSV cannot supply
----------------------------
The CSV is a *flattened message list*: it carries no capture timestamps,
no IP/port endpoints and no flow identifiers.  Rather than fabricate
ground-truth-adjacent values, the loader makes three explicit,
documented inferences:

``timestamp``
    ``None`` when ``synthetic_timestamps=False``.  By default a
    monotonically increasing sequence number is used, because the CSV row
    order *is* arrival order and the two-stage optimiser's remote-coupling
    constraint needs an ordering to pair requests with responses.  This is
    an ordering hint, not a measured time.

``session_id``
    One session per source file (``"<file stem>"``).  A CNNPRE CSV is a
    single protocol capture, so treating each file as one interaction
    session is the natural granularity and keeps request/response pairing
    correct when several CSVs are loaded as one benchmark.

``source_ip`` / ``source_port`` / ``destination_ip`` / ``destination_port``
    ``None`` -- genuinely absent from the CSV.

Ground-truth label
------------------
By default the label is the message type alone (``"<protocol>:<type>"``),
which is exactly the ground truth CNNPRE evaluates against
(``adjusted_rand_score(packet['type'], predicts)``) and therefore the
form that makes results comparable with the CNNPRE paper.  Passing
``include_direction_in_label=True`` additionally prefixes the interaction
role, reproducing the ``"<direction>:<protocol>:<value>"`` convention used
by the PCAP/NetPlier path in this repository.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CsvDatasetLoader:
    """Load CNNPRE ``*-hex.csv`` message datasets for RPKClust evaluation.

    Drop-in companion to :class:`datasets.dataset_loader.PcapDatasetLoader`:
    both expose ``extract_payloads_with_metadata`` / ``load_folder`` and
    populate the same ``last_stats`` / ``last_metadata`` attributes.
    """

    #: NetPlier's ``Processing.MAX_LEN`` -- applied for parity with the PCAP
    #: path so messages are capped identically across both sources.
    MAX_MESSAGE_LENGTH = 500

    #: ``<PROTOCOL><count>-hex.csv`` -> protocol name.
    _FILENAME_PATTERN = re.compile(r"^(?P<protocol>[A-Za-z][A-Za-z0-9]*?)(?P<count>\d+)?(?:-hex)?$")

    #: ``direction`` column value -> interaction role understood by
    #: :class:`rpkclust.constraints.ClusteringConstraints`
    #: (``CLIENT_LABELS = ("client", "request")``,
    #:  ``SERVER_LABELS = ("server", "response")``).
    #:
    #: ``0`` is the server/response side and ``1`` the client/request side
    #: for the majority of the CNNPRE TCP captures (FTP ``220`` banner,
    #: SMTP ``220`` greeting and POP ``+OK`` greeting are all ``0``; the
    #: HTTP ``POST`` request is ``1``).  Captures that follow the opposite
    #: convention can be handled by passing ``direction_map``.
    DEFAULT_DIRECTION_MAP: Dict[Any, str] = {
        0: "response",
        1: "request",
        "0": "response",
        "1": "request",
    }

    #: Protocol -> transport guess.  Used for reporting/metadata only;
    #: nothing in RPKClust depends on it.
    TRANSPORT_MAP: Dict[str, str] = {
        "arp": "arp",
        "dhcp": "udp",
        "dns": "udp",
        "ftp": "tcp",
        "gsm": "gsm",
        "http": "tcp",
        "icmp": "icmp",
        "nbns": "udp",
        "ntp": "udp",
        "pop": "tcp",
        "smtp": "tcp",
        "syslog": "udp",
    }

    #: Columns the loader needs.  ``Full`` is deliberately excluded so the
    #: (very large) bit-string column is never parsed.
    _USECOLS = ("direction", "type", "hex")

    def __init__(
        self,
        target_dir: str = ".",
        direction_map: Optional[Dict[Any, str]] = None,
        synthetic_timestamps: bool = True,
        include_direction_in_label: bool = False,
        max_message_length: Optional[int] = None,
    ):
        self.target_dir = target_dir

        self.direction_map = (
            dict(direction_map)
            if direction_map is not None
            else dict(self.DEFAULT_DIRECTION_MAP)
        )

        self.synthetic_timestamps = synthetic_timestamps
        self.include_direction_in_label = include_direction_in_label
        self.max_message_length = (
            int(max_message_length)
            if max_message_length is not None
            else self.MAX_MESSAGE_LENGTH
        )

        #: Metadata for the most recent extraction.
        self.last_metadata: List[Dict[str, Any]] = []

        #: Diagnostics for the most recent extraction.
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def discover_csv_files(folder: str) -> List[Path]:
        """Return the ``*.csv`` files in ``folder``, sorted by name."""

        path = Path(folder)

        if not path.exists():
            raise FileNotFoundError(
                f"CSV dataset directory does not exist:\n    {path}"
            )

        if path.is_file():
            return [path]

        files = sorted(path.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in:\n    {path}"
            )

        return files

    @classmethod
    def infer_protocol(cls, filename: str) -> str:
        """``"GSM5000-hex.csv"`` -> ``"gsm"``; ``"NTP1043-hex.csv"`` -> ``"ntp"``."""

        stem = Path(filename).stem

        match = cls._FILENAME_PATTERN.match(stem)

        if not match:
            # Fall back to everything before the first digit.
            prefix = re.split(r"\d", stem, maxsplit=1)[0]
            prefix = prefix.strip("-_ ")
            return (prefix or stem).lower()

        return match.group("protocol").lower()

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def _read_rows(self, csv_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Read the needed columns of one CSV, tolerating ragged rows."""

        read_kwargs: Dict[str, Any] = {
            "dtype": str,
            "keep_default_na": False,
            "engine": "python",
        }

        # Only request the columns that exist in this file (``Full`` is
        # large and unused).
        try:
            header = pd.read_csv(csv_path, nrows=0, dtype=str)
            wanted = [c for c in header.columns if c in self._USECOLS]
            if not wanted:
                raise ValueError(
                    f"{csv_path} has none of the required columns "
                    f"{self._USECOLS}; found {list(header.columns)}"
                )
            read_kwargs["usecols"] = wanted
        except pd.errors.ParserError:
            # Fall back to a plain read; the caller validates the columns.
            pass

        df = pd.read_csv(csv_path, **read_kwargs)

        for column in self._USECOLS:
            if column not in df.columns:
                df[column] = ""

        counts = {
            "rows": int(len(df)),
            "missing_type": int((df["type"].astype(str).str.strip() == "").sum()),
        }

        return df, counts

    def load_csv(
        self,
        csv_path: str,
        protocol: Optional[str] = None,
        limit: Optional[int] = None,
        min_length: int = 1,
    ) -> Tuple[List[bytes], np.ndarray, List[Dict[str, Any]]]:
        """Load one CNNPRE CSV into ``(X, y, metadata)``.

        ``y`` is an integer encoding of the ground-truth message types,
        local to this file; the canonical strings live in
        ``metadata[i]["gt_label"]`` (same contract as the PCAP loader).
        """

        csv_path = str(csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        if protocol is None:
            protocol = self.infer_protocol(csv_path)

        protocol = protocol.lower()

        df, read_counts = self._read_rows(csv_path)

        session_id = Path(csv_path).stem

        X: List[bytes] = []
        label_names: List[str] = []
        metadata: List[Dict[str, Any]] = []

        counters: Dict[str, Any] = {
            "source": "csv",
            "file": os.path.basename(csv_path),
            "rows_read": read_counts["rows"],
            "packets_read": read_counts["rows"],
            "link_types": "n/a (csv)",
            "transport_payloads": 0,
            "extracted_messages": 0,
            "skipped_empty": 0,
            "skipped_min_length": 0,
            "skipped_unsupported": 0,
            "skipped_truncated": 0,
            "skipped_preprocess": 0,
            "skipped_no_gt": read_counts["missing_type"],
            "unknown_direction": 0,
            "reassembly": {},
            "import_backend": "cnnpre-csv",
            "direction_map": dict(self.direction_map),
            "synthetic_timestamps": bool(self.synthetic_timestamps),
        }

        row_index = -1

        for record in df.itertuples(index=False):
            row_index += 1

            if limit is not None and limit > 0 and len(X) >= limit:
                break

            raw_hex = getattr(record, "hex", "")
            raw_type = getattr(record, "type", "")
            raw_direction = getattr(record, "direction", "")

            raw_hex = "" if raw_hex is None else str(raw_hex).strip()
            raw_type = "" if raw_type is None else str(raw_type).strip()

            if not raw_hex:
                counters["skipped_empty"] += 1
                continue

            try:
                payload = bytes.fromhex(raw_hex)
            except ValueError:
                counters["skipped_unsupported"] += 1
                continue

            if len(payload) < min_length:
                counters["skipped_min_length"] += 1
                continue

            counters["transport_payloads"] += 1

            if len(payload) > self.max_message_length:
                payload = payload[: self.max_message_length]
                counters["skipped_truncated"] += 1

            if not raw_type:
                counters["skipped_no_gt"] += 1
                continue

            direction = self.direction_map.get(str(raw_direction).strip())

            if direction is None:
                direction = "unknown"
                counters["unknown_direction"] += 1

            if self.include_direction_in_label:
                gt_label = f"{direction}:{protocol}:{raw_type}"
            else:
                gt_label = f"{protocol}:{raw_type}"

            timestamp = float(row_index) if self.synthetic_timestamps else None

            metadata.append(
                {
                    "packet_index": row_index,

                    "timestamp": timestamp,

                    "protocol": protocol,

                    "transport": self.TRANSPORT_MAP.get(protocol, "unknown"),

                    "direction": direction,

                    # --- evaluation ground truth (never fed to RPKClust) ---
                    "gt_value": raw_type,
                    "gt_label": gt_label,
                    "gt_source": "CNNPRE",

                    # --- genuinely absent in the CSV ---
                    "source_ip": None,
                    "source_port": None,
                    "destination_ip": None,
                    "destination_port": None,

                    "session_id": session_id,

                    "payload_length": len(payload),

                    # --- traceability (harmless extras) ---
                    "source_file": os.path.basename(csv_path),
                    "direction_raw": str(raw_direction).strip(),
                }
            )

            X.append(payload)
            label_names.append(gt_label)

        counters["extracted_messages"] = len(X)

        if self.include_direction_in_label:
            counters["label_mode"] = "direction:protocol:type"
        else:
            counters["label_mode"] = "protocol:type"

        self.last_metadata = metadata
        self.last_stats = counters

        if not X:
            raise ValueError(
                f"No qualifying messages found in {csv_path} "
                f"(stats: {counters})"
            )

        unique_labels = sorted(set(label_names))
        label_to_id = {label: index for index, label in enumerate(unique_labels)}

        y = np.array(
            [label_to_id[label] for label in label_names],
            dtype=int,
        )

        counters["gt_classes"] = len(unique_labels)
        counters["messages"] = len(X)

        return X, y, metadata

    # ------------------------------------------------------------------
    # API parity with PcapDatasetLoader
    # ------------------------------------------------------------------

    def extract_payloads_with_metadata(
        self,
        csv_path: str,
        protocol: Optional[str] = None,
        min_length: int = 1,
        limit: Optional[int] = None,
        **_ignored: Any,
    ) -> Tuple[List[bytes], np.ndarray, List[Dict[str, Any]]]:
        """Alias of :meth:`load_csv` matching the PCAP loader's signature."""

        return self.load_csv(
            csv_path,
            protocol=protocol,
            limit=limit,
            min_length=min_length,
        )

    # ------------------------------------------------------------------
    # Folder dataset
    # ------------------------------------------------------------------

    def load_folder(
        self,
        folder_path: str,
        limit: Optional[int] = None,
        min_length: int = 1,
        protocols: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[bytes], np.ndarray, List[Dict[str, Any]]]:
        """Load every CSV in a directory as one benchmark.

        ``limit`` caps the number of messages **per file**.  The integer
        labels in ``y`` are encoded over the *combined* label space, so
        datasets are directly comparable in the summary table.

        ``protocols`` optionally overrides the filename-inferred protocol
        per file (``{"NTP1043-hex.csv": "ntp"}``).
        """

        files = self.discover_csv_files(folder_path)

        protocols = protocols or {}

        X: List[bytes] = []
        label_names: List[str] = []
        metadata: List[Dict[str, Any]] = []

        per_file_stats: List[Dict[str, Any]] = []

        global_index = 0

        for csv_file in files:
            protocol = protocols.get(csv_file.name) or self.infer_protocol(csv_file.name)

            file_loader = CsvDatasetLoader(
                target_dir=self.target_dir,
                direction_map=self.direction_map,
                synthetic_timestamps=self.synthetic_timestamps,
                include_direction_in_label=self.include_direction_in_label,
                max_message_length=self.max_message_length,
            )

            X_part, _y_part, meta_part = file_loader.load_csv(
                str(csv_file),
                protocol=protocol,
                limit=limit,
                min_length=min_length,
            )

            for record in meta_part:
                record["packet_index"] = global_index
                global_index += 1
                label_names.append(record["gt_label"])

            # ``meta_part`` and ``X_part`` are index-aligned.
            X.extend(X_part)
            metadata.extend(meta_part)

            stats = dict(file_loader.last_stats)
            stats["protocol"] = protocol
            per_file_stats.append(stats)

            print(
                f"[CsvLoader] {csv_file.name:<20} protocol={protocol:<8} "
                f"messages={len(X_part):>6}  classes={file_loader.last_stats.get('gt_classes')}"
            )

        if not X:
            raise ValueError(f"No qualifying messages found in {folder_path}")

        unique_labels = sorted(set(label_names))
        label_to_id = {label: index for index, label in enumerate(unique_labels)}

        y = np.array(
            [label_to_id[label] for label in label_names],
            dtype=int,
        )

        self.last_metadata = metadata
        self.last_stats = {
            "source": "csv",
            "folder": str(folder_path),
            "files": [str(f) for f in files],
            "rows_read": sum(s["rows_read"] for s in per_file_stats),
            "packets_read": sum(s["packets_read"] for s in per_file_stats),
            "link_types": "n/a (csv)",
            "transport_payloads": sum(s["transport_payloads"] for s in per_file_stats),
            "extracted_messages": len(X),
            "messages": len(X),
            "skipped_empty": sum(s["skipped_empty"] for s in per_file_stats),
            "skipped_min_length": sum(s["skipped_min_length"] for s in per_file_stats),
            "skipped_unsupported": sum(s["skipped_unsupported"] for s in per_file_stats),
            "skipped_truncated": sum(s["skipped_truncated"] for s in per_file_stats),
            "skipped_preprocess": 0,
            "skipped_no_gt": sum(s["skipped_no_gt"] for s in per_file_stats),
            "unknown_direction": sum(s["unknown_direction"] for s in per_file_stats),
            "reassembly": {},
            "import_backend": "cnnpre-csv",
            "gt_classes": len(unique_labels),
            "per_file": per_file_stats,
            "limit_per_file": limit,
        }

        return X, y, metadata