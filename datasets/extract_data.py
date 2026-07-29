"""
AMQP PCAP extractor for RPKClust evaluation.

Input:
    .pcap file

Output:
    X:
        List[bytes]
        Raw AMQP frames suitable for RPKClust

Metadata:
        Connection information
        Frame type
        Channel
        Payload size
        Direction
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple

from scapy.all import (
    rdpcap,
    TCP,
    Raw,
    IP
)


@dataclass
class AMQPFrameMetadata:
    connection: Tuple
    frame_type: int
    channel: int
    payload_length: int
    raw_length: int


class AMQPExtractor:
    """
    Extracts AMQP 0-9-1 frames from PCAP files.

    Designed for protocol clustering algorithms such as RPKClust.

    The output messages are raw binary protocol messages.
    """

    AMQP_PORT = 5672
    FRAME_END = 0xCE

    AMQP_HEADER = b"AMQP"

    FRAME_TYPES = {
        1: "METHOD",
        2: "HEADER",
        3: "BODY",
        8: "HEARTBEAT"
    }


    def __init__(
        self,
        pcap_file: str,
        client_only=True,
        remove_handshake=True,
        remove_heartbeat=True
    ):

        self.pcap_file = pcap_file

        self.client_only = client_only
        self.remove_handshake = remove_handshake
        self.remove_heartbeat = remove_heartbeat


        self.streams = {}

        self.frames = []

        self.metadata = []
        self._extracted = False


    # -----------------------------------------------------
    # TCP extraction
    # -----------------------------------------------------

    def _extract_tcp_streams(self):

        packets = rdpcap(self.pcap_file)


        streams = defaultdict(bytearray)


        for pkt in packets:


            if not (
                IP in pkt
                and TCP in pkt
                and Raw in pkt
            ):
                continue


            tcp = pkt[TCP]


            # client -> server
            if self.client_only:

                if tcp.dport != self.AMQP_PORT:
                    continue


            else:

                if (
                    tcp.sport != self.AMQP_PORT
                    and
                    tcp.dport != self.AMQP_PORT
                ):
                    continue



            key = (
                pkt[IP].src,
                tcp.sport,
                pkt[IP].dst,
                tcp.dport
            )


            streams[key].extend(
                bytes(pkt[Raw].load)
            )


        self.streams = dict(streams)

        return self.streams



    # -----------------------------------------------------
    # AMQP frame parsing
    # -----------------------------------------------------

    def _parse_frames(
        self,
        stream: bytes,
        connection
    ):

        frames = []


        offset = 0


        while offset + 7 < len(stream):


            # Skip the eight-byte AMQP protocol header only when requested.
            if self.remove_handshake and stream[offset:offset+4] == self.AMQP_HEADER:
                offset += 8
                continue



            frame_type = stream[offset]


            channel = int.from_bytes(
                stream[
                    offset+1:
                    offset+3
                ],
                "big"
            )


            payload_size = int.from_bytes(
                stream[
                    offset+3:
                    offset+7
                ],
                "big"
            )


            frame_end = (
                offset
                +
                7
                +
                payload_size
                +
                1
            )


            if frame_end > len(stream):
                break



            frame = stream[offset:frame_end]


            if frame[-1] != self.FRAME_END:

                offset += 1
                continue



            # remove heartbeat frames
            if (
                self.remove_heartbeat
                and frame_type == 8
            ):

                offset = frame_end
                continue



            frames.append(frame)


            self.metadata.append(
                AMQPFrameMetadata(
                    connection=connection,
                    frame_type=frame_type,
                    channel=channel,
                    payload_length=payload_size,
                    raw_length=len(frame)
                )
            )


            offset = frame_end



        return frames



    # -----------------------------------------------------
    # Public extraction API
    # -----------------------------------------------------

    def extract(self):

        # Extraction is repeatable: metadata must describe only this run.
        self.frames = []
        self.metadata = []
        self._extracted = False
        self._extract_tcp_streams()


        all_frames = []


        for connection, stream in self.streams.items():


            frames = self._parse_frames(
                bytes(stream),
                connection
            )


            all_frames.extend(frames)



        self.frames = all_frames
        self._extracted = True


        return self.frames



    def get_messages(self):

        """
        Returns output compatible with:

            RPKClust.fit(X)

        """

        if not self._extracted:

            self.extract()


        return self.frames



    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    def summary(self):

        if not self._extracted:

            self.extract()


        print("="*60)
        print("AMQP EXTRACTION SUMMARY")
        print("="*60)


        print(
            "TCP Streams:",
            len(self.streams)
        )


        print(
            "AMQP Frames:",
            len(self.frames)
        )


        print("\nFrame Types:")


        counts = defaultdict(int)


        for m in self.metadata:

            counts[
                self.FRAME_TYPES.get(
                    m.frame_type,
                    str(m.frame_type)
                )
            ] += 1


        for k,v in counts.items():

            print(
                f" {k:<12}: {v}"
            )


        print("\nFrame Size Statistics:")

        sizes = [
            m.raw_length
            for m in self.metadata
        ]


        if sizes:

            print(
                "Min:",
                min(sizes)
            )

            print(
                "Max:",
                max(sizes)
            )

            print(
                "Avg:",
                sum(sizes)/len(sizes)
            )



    def get_metadata(self):

        return self.metadata