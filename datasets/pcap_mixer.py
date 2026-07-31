import csv
from nt import read
import os
from pathlib import Path
import random
from collections import defaultdict
from typing import List

from scapy.all import rdpcap, wrpcap
from scapy.layers.inet import IP, TCP


class PcapMixer:
    """
    Merge multiple PCAP files into one.

    Features
    --------
    ✔ Preserve every packet exactly as-is.
    ✔ Preserve all protocol headers and metadata.
    ✔ Can shuffle individual packets.
    ✔ Can shuffle complete TCP sessions (recommended).
    ✔ Can optionally regenerate timestamps.
    """

    def __init__(self, random_seed=None):
        self.random = random.Random(random_seed)

    # ---------------------------------------------------------

    def merge(
        self,
        input_files: List[str],
        output_file: str,
        shuffle_mode: str = "flow",
        rewrite_timestamps: bool = False,
        packet_interval: float = 0.001,
    ):
        """
        Parameters
        ----------
        input_files : list[str]
        output_file : str

        shuffle_mode:
            "none"   -> simply concatenate
            "packet" -> shuffle every packet
            "flow"   -> shuffle TCP sessions (recommended)

        rewrite_timestamps:
            If True timestamps become continuous.

        packet_interval:
            Used only if rewrite_timestamps=True
        """

        packets = []
        flow_labels = {}
        label_map = {}
        next_label = 0

        for file in input_files:
            if not os.path.exists(file):
                raise FileNotFoundError(file)

            print(f"Loading {file}")

            pcap = rdpcap(file)

            label_name = Path(file).stem

            if label_name not in label_map:
                label_map[label_name] = next_label
                next_label += 1

            for pkt in pcap:
                pkt.original_file = label_name
                packets.append(pkt)

        print(f"Loaded {len(packets)} packets.")

        if shuffle_mode == "packet":
            packets = self._shuffle_packets(packets)

        elif shuffle_mode == "flow":
            packets, ground_truth = self._shuffle_flows(packets)

        elif shuffle_mode == "none":
            pass

        else:
            raise ValueError(
                "shuffle_mode must be one of: none, packet, flow"
            )

        if rewrite_timestamps:
            self._rewrite_timestamps(
                packets,
                interval=packet_interval
            )

        wrpcap(output_file, packets)

        csv_file = output_file.replace(".pcap", "_ground_truth.csv")

        with open(csv_file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "flow_id",
                    "label",
                    "label_id"
                ]
            )

            for flow_id, label in ground_truth:

                writer.writerow(
                    [
                        flow_id,
                        label,
                        label_map[label]
                    ]
                )

        print(f"Saved {len(packets)} packets to:")
        print(output_file)

    # ---------------------------------------------------------

    def _shuffle_packets(self, packets):
        packets = list(packets)
        self.random.shuffle(packets)
        return packets

    # ---------------------------------------------------------

    def _shuffle_flows(self, packets):
        """
        Shuffle TCP sessions while preserving
        packet order inside each session.
        """

        flows = defaultdict(lambda: {
            "packets": [],
            "label": None
        })
        others = []

        for pkt in packets:

            if IP in pkt and TCP in pkt:

                ip = pkt[IP]
                tcp = pkt[TCP]

                # Bidirectional flow key
                endpoints = sorted([
                    (ip.src, tcp.sport),
                    (ip.dst, tcp.dport)
                ])

                key = (
                    endpoints[0][0],
                    endpoints[0][1],
                    endpoints[1][0],
                    endpoints[1][1],
                )

                flows[key]["packets"].append(pkt)

                if flows[key]["label"] is None:
                    flows[key]["label"] = pkt.original_file

            else:
                others.append(pkt)

        flow_list = list(flows.values())
        
        print(f"Detected {len(flow_list)} TCP flows.")

        self.random.shuffle(flow_list)

        mixed = []
        ground_truth = []

        flow_id = 0

        for flow in flow_list:

            mixed.extend(flow["packets"])

            ground_truth.append(
                (
                    flow_id,
                    flow["label"]
                )
            )

            flow_id += 1

        return mixed, ground_truth

    # ---------------------------------------------------------

    def _rewrite_timestamps(self, packets, interval=0.001):
        t = 0.0

        for pkt in packets:
            pkt.time = t
            t += interval
    
    from pathlib import Path


    def merge_folder(
        self,
        folder: str,
        shuffle_mode: str = "flow",
        rewrite_timestamps: bool = False,
        packet_interval: float = 0.001,
    ):
        """
        Merge every pcap inside a folder.

        Output file name = folder name.
        """

        folder = Path(folder)

        if not folder.exists():
            raise FileNotFoundError(folder)

        pcap_files = sorted(
            list(folder.glob("*.pcap")) +
            list(folder.glob("*.pcapng"))
        )

        if len(pcap_files) == 0:
            raise RuntimeError(f"No PCAP files found in {folder}")

        output_file = folder / f"{folder.name}.pcap"

        print(f"Found {len(pcap_files)} PCAP files.")

        self.merge(
            input_files=[str(f) for f in pcap_files],
            output_file=str(output_file),
            shuffle_mode=shuffle_mode,
            rewrite_timestamps=rewrite_timestamps,
            packet_interval=packet_interval,
        )

mixer = PcapMixer(random_seed=42)
folder = input()
mixer.merge_folder(
    folder=folder,
    shuffle_mode="flow"
)