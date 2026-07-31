import random
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class MessageType(IntEnum):
    """
    Supported message types for the simulated realistic binary protocol.
    Each message type represents a distinct control or data operation.
    """
    CONNECT = 1
    CONNECT_ACK = 2
    READ = 3
    READ_RESPONSE = 4
    WRITE = 5
    WRITE_RESPONSE = 6
    HEARTBEAT = 7
    STATUS = 8
    CONFIG = 9
    ERROR = 10


class Direction(Enum):
    """
    Traffic flow direction relative to client and server roles.
    """
    CLIENT_TO_SERVER = "client_to_server"
    SERVER_TO_CLIENT = "server_to_client"


class TLVType(IntEnum):
    """
    Types for optional Type-Length-Value (TLV) extension fields inside payloads.
    """
    AUTH_TOKEN = 1
    DEVICE_INFO = 2
    ERROR_DETAILS = 3
    PADDING = 4
    CONFIG_PARAM = 5


@dataclass
class NetworkEndpoint:
    """
    Represents a network client or server endpoint with an IP address and port.
    """
    ip: str
    port: int

    def __str__(self) -> str:
        return f"{self.ip}:{self.port}"


@dataclass
class SessionContext:
    """
    Tracks state, sequence numbers, and timestamps for an active communication session.
    """
    session_id: int
    client: NetworkEndpoint
    server: NetworkEndpoint
    start_timestamp: float
    current_timestamp: float
    client_seq: int = 1
    server_seq: int = 1
    transaction_id_counter: int = 100


class ChecksumUtils:
    """
    Utility class for computing protocol checksums and error-checking fields.
    """

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        """
        Calculates standard CRC-16-CCITT checksum over given input bytes.
        
        Args:
            data: Binary payload over which checksum is computed.
            
        Returns:
            16-bit unsigned integer checksum value.
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc & 0xFFFF


class GenericDatasetGenerator:
    """
    Realistic synthetic binary protocol traffic generator for reverse engineering evaluations.
    Constructs message streams adhering to real-world industrial and network protocol properties
    including fixed header fields, variable-length payloads, TLVs, and session correlations.
    """

    MAGIC_NUMBER = b'\x52\x50'  # ASCII 'RP' (RPKClust Protocol)
    HEADER_FORMAT = '>2sBBBHIIHH'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # Exactly 19 bytes
    KEYWORD_OFFSET = 4  # Offset of Message Type byte in the header
    KEYWORD_LENGTH = 1

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """
        Initializes the dataset generator.

        Args:
            random_seed: Optional seed for reproducible pseudorandom generation.
        """
        self._seed = random_seed
        self._rng = random.Random(random_seed)

    def _generate_network_topology(
        self, num_sessions: int, rng: random.Random
    ) -> List[Tuple[NetworkEndpoint, NetworkEndpoint]]:
        """
        Generates realistic client and server IP/port endpoints.

        Args:
            num_sessions: Number of sessions to generate topology for.
            rng: Local random instance.

        Returns:
            List of (client_endpoint, server_endpoint) pairs.
        """
        client_ips = [f"192.168.1.{rng.randint(10, 250)}" for _ in range(max(5, num_sessions // 3))]
        server_ips = [f"10.0.0.{rng.randint(2, 50)}" for _ in range(max(1, num_sessions // 10))]

        clients = [NetworkEndpoint(ip, rng.randint(49152, 65535)) for ip in client_ips]
        servers = [NetworkEndpoint(ip, rng.choice([502, 1883, 8080, 20000])) for ip in server_ips]

        topology = []
        for _ in range(num_sessions):
            client = rng.choice(clients)
            server = rng.choice(servers)
            topology.append((client, server))
        return topology

    def _build_tlv(self, tlv_type: TLVType, value: bytes) -> bytes:
        """
        Constructs a binary Type-Length-Value (TLV) structure.

        Args:
            tlv_type: Enum indicating the TLV type.
            value: Raw byte content of the TLV value field.

        Returns:
            Packed TLV bytes.
        """
        length = len(value)
        return struct.pack('>BB', tlv_type.value, length & 0xFF) + value

    def _build_message_header(
        self,
        msg_type: MessageType,
        flags: int,
        transaction_id: int,
        sequence_number: int,
        timestamp_ms: int,
        session_id: int,
        payload_length: int,
        version: int = 1,
    ) -> bytes:
        """
        Packs fixed header fields into a 19-byte binary structure.

        Layout:
        - Magic Number (2B): 0x5250 ("RP")
        - Version (1B): Protocol version
        - Flags (1B): Bitfield (0x01=Req, 0x02=Resp, 0x04=Error, 0x08=TLV)
        - Message Type (1B): Opcode / Keyword field [Offset 4]
        - Transaction ID (2B): Correlation identifier
        - Sequence Number (4B): Monotonic counter
        - Timestamp (4B): Session time offset in milliseconds
        - Session ID (2B): Unique session identifier
        - Payload Length (2B): Length of body + TLVs + trailer
        """
        return struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC_NUMBER,
            version & 0xFF,
            flags & 0xFF,
            msg_type.value & 0xFF,
            transaction_id & 0xFFFF,
            sequence_number & 0xFFFFFFFF,
            timestamp_ms & 0xFFFFFFFF,
            session_id & 0xFFFF,
            payload_length & 0xFFFF,
        )

    def _build_message_body(
        self,
        msg_type: MessageType,
        rng: random.Random,
        req_ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bytes, int, int]:
        """
        Constructs realistic binary body payloads and optional TLVs for each message type.

        Args:
            msg_type: Type of message to construct body for.
            rng: Local random instance.
            req_ctx: Optional context from a paired request message.

        Returns:
            Tuple of (full_payload_bytes, core_body_length, tlv_length).
        """
        tlv_bytes = bytearray()

        # Optional TLVs based on probabilities
        if rng.random() < 0.20 and msg_type in (MessageType.CONNECT, MessageType.STATUS, MessageType.CONFIG):
            token = rng.randbytes(8)
            tlv_bytes.extend(self._build_tlv(TLVType.AUTH_TOKEN, token))

        if rng.random() < 0.15:
            info = f"DEV-{rng.randint(100, 999)}".encode('utf-8')
            tlv_bytes.extend(self._build_tlv(TLVType.DEVICE_INFO, info))

        if rng.random() < 0.10:
            padding_len = rng.randint(1, 4)
            padding = b'\x00' * padding_len
            tlv_bytes.extend(self._build_tlv(TLVType.PADDING, padding))

        body_stream = bytearray()

        if msg_type == MessageType.CONNECT:
            client_version = rng.randint(1, 3)
            client_id = rng.randint(10000, 99999)
            body_stream.extend(struct.pack('>HI', client_version, client_id))

        elif msg_type == MessageType.CONNECT_ACK:
            status_code = 0 if rng.random() > 0.05 else rng.randint(1, 5)
            session_tag = rng.randint(0x1000, 0xFFFF)
            server_uptime = rng.randint(100, 100000)
            body_stream.extend(struct.pack('>BHI', status_code, session_tag, server_uptime))

        elif msg_type == MessageType.READ:
            address = rng.choice([100, 200, 300, 1000, 2000]) + rng.randint(0, 10)
            quantity = rng.randint(1, 15)
            read_flags = rng.choice([0x01, 0x02, 0x04])
            body_stream.extend(struct.pack('>HHB', address, quantity, read_flags))

        elif msg_type == MessageType.READ_RESPONSE:
            reg_count = req_ctx.get('quantity', 4) if req_ctx else rng.randint(1, 10)
            base_val = rng.randint(200, 500)
            data_bytes = bytearray()
            for i in range(reg_count):
                val = base_val + i * 2 + rng.randint(-3, 3)
                data_bytes.extend(struct.pack('>H', val & 0xFFFF))
            body_stream.extend(struct.pack('>H', len(data_bytes)))
            body_stream.extend(data_bytes)

        elif msg_type == MessageType.WRITE:
            target_addr = rng.choice([100, 200, 300, 1000])
            val_len = rng.choice([2, 4, 8, 16])
            val_bytes = rng.randbytes(val_len)
            body_stream.extend(struct.pack('>HH', target_addr, val_len))
            body_stream.extend(val_bytes)

        elif msg_type == MessageType.WRITE_RESPONSE:
            target_addr = req_ctx.get('target_addr', 100) if req_ctx else 100
            status = 0 if rng.random() > 0.05 else 1
            body_stream.extend(struct.pack('>HB', target_addr, status))

        elif msg_type == MessageType.HEARTBEAT:
            uptime = rng.randint(1000, 500000)
            health_status = rng.choice([0x01, 0x01, 0x01, 0x02])
            body_stream.extend(struct.pack('>IH', uptime, health_status))

        elif msg_type == MessageType.STATUS:
            subsystem_id = rng.randint(1, 8)
            metric_type = rng.randint(1, 4)
            metric_val = rng.randint(10, 100)
            body_stream.extend(struct.pack('>BBH', subsystem_id, metric_type, metric_val))

        elif msg_type == MessageType.CONFIG:
            param_id = rng.randint(1, 50)
            param_val = rng.randint(0, 255)
            body_stream.extend(struct.pack('>HB', param_id, param_val))

        elif msg_type == MessageType.ERROR:
            err_code = rng.choice([400, 401, 403, 404, 500])
            failed_opcode = req_ctx.get('failed_opcode', 3) if req_ctx else 3
            body_stream.extend(struct.pack('>HB', err_code, failed_opcode))
            err_msg = f"ERR_{err_code}".encode('utf-8')
            tlv_bytes.extend(self._build_tlv(TLVType.ERROR_DETAILS, err_msg))

        full_payload = bytes(body_stream) + bytes(tlv_bytes)
        return full_payload, len(body_stream), len(tlv_bytes)

    def _create_message(
        self,
        ctx: SessionContext,
        msg_type: MessageType,
        direction: Direction,
        is_request: bool,
        is_response: bool,
        is_error: bool,
        request_id: Optional[int],
        response_to: Optional[int],
        rng: random.Random,
        req_ctx: Optional[Dict[str, Any]] = None,
        inject_noise: bool = True,
    ) -> Tuple[bytes, MessageType, Dict[str, Any]]:
        """
        Assembles a full message binary packet and produces ground truth metadata.

        Args:
            ctx: Current session state context.
            msg_type: Message opcode type.
            direction: Direction of transmission.
            is_request: Flag indicating if message is a request.
            is_response: Flag indicating if message is a response.
            is_error: Flag indicating error status.
            request_id: Associated request transaction ID.
            response_to: Associated response transaction ID.
            rng: Local random generator.
            req_ctx: Context passed from paired request message.
            inject_noise: Whether to inject minor realistic protocol noise.

        Returns:
            Tuple of (raw_bytes, msg_type, metadata_dict).
        """
        flags = 0
        if is_request:
            flags |= 0x01
        if is_response:
            flags |= 0x02
        if is_error:
            flags |= 0x04

        if direction == Direction.CLIENT_TO_SERVER:
            seq_num = ctx.client_seq
            ctx.client_seq += 1
            src_ep, dst_ep = ctx.client, ctx.server
        else:
            seq_num = ctx.server_seq
            ctx.server_seq += 1
            src_ep, dst_ep = ctx.server, ctx.client

        if is_request:
            trans_id = ctx.transaction_id_counter
            ctx.transaction_id_counter += 1
        else:
            trans_id = request_id if request_id is not None else ctx.transaction_id_counter

        ts_ms = int((ctx.current_timestamp - ctx.start_timestamp) * 1000)

        body_bytes, core_body_len, tlv_len = self._build_message_body(msg_type, rng, req_ctx)

        if tlv_len > 0:
            flags |= 0x08

        payload_len = len(body_bytes) + 2  # +2 for trailer CRC checksum

        version = 1
        if rng.random() < 0.02 and inject_noise:
            version = 2

        header = self._build_message_header(
            msg_type=msg_type,
            flags=flags,
            transaction_id=trans_id,
            sequence_number=seq_num,
            timestamp_ms=ts_ms,
            session_id=ctx.session_id,
            payload_length=payload_len,
            version=version,
        )

        data_to_checksum = header + body_bytes
        checksum = ChecksumUtils.compute_crc16(data_to_checksum)

        corrupt_checksum = False
        if rng.random() < 0.01 and inject_noise:
            checksum ^= 0xFFFF
            corrupt_checksum = True

        checksum_bytes = struct.pack('>H', checksum)
        raw_message = header + body_bytes + checksum_bytes

        metadata: Dict[str, Any] = {
            "message_type": msg_type.name,
            "message_type_id": msg_type.value,
            "session_id": ctx.session_id,
            "transaction_id": trans_id,
            "sequence_number": seq_num,
            "timestamp": round(ctx.current_timestamp, 4),
            "direction": direction.value,
            "client": str(ctx.client),
            "server": str(ctx.server),
            "source_ip": src_ep.ip,
            "destination_ip": dst_ep.ip,
            "source_port": src_ep.port,
            "destination_port": dst_ep.port,
            "payload_length": payload_len,
            "header_length": self.HEADER_SIZE,
            "total_length": len(raw_message),
            "request_id": trans_id if is_request else request_id,
            "response_to": response_to,
            "is_request": is_request,
            "is_response": is_response,
            "is_error": is_error,
            "flags": flags,
            "checksum": checksum,
            "corrupt_checksum": corrupt_checksum,
            "cluster_label": msg_type.name,
            "keyword_offset": self.KEYWORD_OFFSET,
            "keyword_length": self.KEYWORD_LENGTH,
            "body_offset": self.HEADER_SIZE,
            "body_length": len(body_bytes),
        }

        return raw_message, msg_type, metadata

    def generate_realistic_dataset(
        self,
        num_sessions: int = 50,
        min_messages_per_session: int = 10,
        max_messages_per_session: int = 40,
        protocol_style: str = "request_response",
        random_seed: Optional[int] = None,
    ) -> Tuple[List[bytes], List[Union[int, str]], List[Dict[str, Any]]]:
        """
        Generates realistic binary protocol dataset with session structures and ground truth metadata.

        Args:
            num_sessions: Number of independent simulated sessions.
            min_messages_per_session: Minimum messages generated per session.
            max_messages_per_session: Maximum messages generated per session.
            protocol_style: Style of protocol interaction flow ("request_response").
            random_seed: Optional seed for reproducible generation.

        Returns:
            Tuple of (X, y, metadata):
                - X: List of raw binary message byte-strings.
                - y: List of ground-truth Message Type labels (string names).
                - metadata: List of ground-truth metadata dictionaries per message.
        """
        rng = random.Random(random_seed) if random_seed is not None else self._rng

        X: List[bytes] = []
        y: List[Union[int, str]] = []
        metadata: List[Dict[str, Any]] = []

        endpoints = self._generate_network_topology(num_sessions, rng)
        base_time = 1700000000.0

        for session_idx in range(num_sessions):
            client_ep, server_ep = endpoints[session_idx % len(endpoints)]
            session_id = 1000 + session_idx

            ctx = SessionContext(
                session_id=session_id,
                client=client_ep,
                server=server_ep,
                start_timestamp=base_time + rng.uniform(0, 3600),
                current_timestamp=base_time + rng.uniform(0, 3600),
                transaction_id_counter=rng.randint(1, 500),
            )

            msg_count = rng.randint(min_messages_per_session, max_messages_per_session)

            # Handshake exchange: CONNECT -> CONNECT_ACK
            msg_raw, msg_type, meta = self._create_message(
                ctx, MessageType.CONNECT, Direction.CLIENT_TO_SERVER,
                is_request=True, is_response=False, is_error=False,
                request_id=None, response_to=None, rng=rng
            )
            X.append(msg_raw)
            y.append(msg_type.name)
            metadata.append(meta)

            ctx.current_timestamp += rng.uniform(0.002, 0.015)

            msg_raw_ack, msg_type_ack, meta_ack = self._create_message(
                ctx, MessageType.CONNECT_ACK, Direction.SERVER_TO_CLIENT,
                is_request=False, is_response=True, is_error=False,
                request_id=meta['transaction_id'], response_to=meta['transaction_id'], rng=rng
            )
            X.append(msg_raw_ack)
            y.append(msg_type_ack.name)
            metadata.append(meta_ack)

            generated = 2
            while generated < msg_count:
                ctx.current_timestamp += rng.uniform(0.010, 0.200)
                p = rng.random()

                if p < 0.40:
                    # READ -> READ_RESPONSE (40%)
                    req_raw, req_type, req_meta = self._create_message(
                        ctx, MessageType.READ, Direction.CLIENT_TO_SERVER,
                        is_request=True, is_response=False, is_error=False,
                        request_id=None, response_to=None, rng=rng
                    )
                    X.append(req_raw)
                    y.append(req_type.name)
                    metadata.append(req_meta)
                    generated += 1

                    if generated < msg_count:
                        ctx.current_timestamp += rng.uniform(0.002, 0.025)
                        resp_raw, resp_type, resp_meta = self._create_message(
                            ctx, MessageType.READ_RESPONSE, Direction.SERVER_TO_CLIENT,
                            is_request=False, is_response=True, is_error=False,
                            request_id=req_meta['transaction_id'], response_to=req_meta['transaction_id'],
                            rng=rng, req_ctx={'quantity': rng.randint(1, 10)}
                        )
                        X.append(resp_raw)
                        y.append(resp_type.name)
                        metadata.append(resp_meta)
                        generated += 1

                elif p < 0.65:
                    # WRITE -> WRITE_RESPONSE (25%)
                    req_raw, req_type, req_meta = self._create_message(
                        ctx, MessageType.WRITE, Direction.CLIENT_TO_SERVER,
                        is_request=True, is_response=False, is_error=False,
                        request_id=None, response_to=None, rng=rng
                    )
                    X.append(req_raw)
                    y.append(req_type.name)
                    metadata.append(req_meta)
                    generated += 1

                    if generated < msg_count:
                        ctx.current_timestamp += rng.uniform(0.002, 0.030)
                        resp_raw, resp_type, resp_meta = self._create_message(
                            ctx, MessageType.WRITE_RESPONSE, Direction.SERVER_TO_CLIENT,
                            is_request=False, is_response=True, is_error=False,
                            request_id=req_meta['transaction_id'], response_to=req_meta['transaction_id'],
                            rng=rng, req_ctx={'target_addr': 100}
                        )
                        X.append(resp_raw)
                        y.append(resp_type.name)
                        metadata.append(resp_meta)
                        generated += 1

                elif p < 0.80:
                    # HEARTBEAT (15%)
                    direction = Direction.CLIENT_TO_SERVER if rng.random() < 0.7 else Direction.SERVER_TO_CLIENT
                    hb_raw, hb_type, hb_meta = self._create_message(
                        ctx, MessageType.HEARTBEAT, direction,
                        is_request=True, is_response=False, is_error=False,
                        request_id=None, response_to=None, rng=rng
                    )
                    X.append(hb_raw)
                    y.append(hb_type.name)
                    metadata.append(hb_meta)
                    generated += 1

                elif p < 0.90:
                    # STATUS (10%)
                    st_raw, st_type, st_meta = self._create_message(
                        ctx, MessageType.STATUS, Direction.CLIENT_TO_SERVER,
                        is_request=True, is_response=False, is_error=False,
                        request_id=None, response_to=None, rng=rng
                    )
                    X.append(st_raw)
                    y.append(st_type.name)
                    metadata.append(st_meta)
                    generated += 1

                elif p < 0.95:
                    # CONFIG (5%)
                    cfg_raw, cfg_type, cfg_meta = self._create_message(
                        ctx, MessageType.CONFIG, Direction.CLIENT_TO_SERVER,
                        is_request=True, is_response=False, is_error=False,
                        request_id=None, response_to=None, rng=rng
                    )
                    X.append(cfg_raw)
                    y.append(cfg_type.name)
                    metadata.append(cfg_meta)
                    generated += 1

                else:
                    # ERROR (5%)
                    err_raw, err_type, err_meta = self._create_message(
                        ctx, MessageType.ERROR, Direction.SERVER_TO_CLIENT,
                        is_request=False, is_response=True, is_error=True,
                        request_id=ctx.transaction_id_counter, response_to=ctx.transaction_id_counter,
                        rng=rng, req_ctx={'failed_opcode': rng.choice([3, 5])}
                    )
                    X.append(err_raw)
                    y.append(err_type.name)
                    metadata.append(err_meta)
                    generated += 1

                # Retransmission noise (2% probability)
                if rng.random() < 0.02 and len(X) > 0:
                    X.append(X[-1])
                    y.append(y[-1])
                    metadata.append(dict(metadata[-1]))

        return X, y, metadata

    def generate_for_dataset(
        self,
        num_messages: int = 200,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[bytes], List[Union[int, str]], List[Dict[str, Any]]]:
        """
        Generates a dataset with a Fixed-Offset Region (FOR) keyword field.
        Maintained for backward compatibility.

        Args:
            num_messages: Total number of messages to generate.
            random_seed: Optional random seed.

        Returns:
            Tuple of (X, y, metadata).
        """
        rng = random.Random(random_seed) if random_seed is not None else self._rng
        X: List[bytes] = []
        y: List[Union[int, str]] = []
        metadata: List[Dict[str, Any]] = []

        types = [MessageType.CONNECT, MessageType.READ, MessageType.WRITE, MessageType.HEARTBEAT]

        for i in range(num_messages):
            msg_type = rng.choice(types)
            header = struct.pack('>2sBBBH', b'RP', 1, 0x01, msg_type.value, i % 65535)
            body = rng.randbytes(rng.randint(10, 30))
            raw = header + body

            X.append(raw)
            y.append(msg_type.name)
            metadata.append({
                "message_type": msg_type.name,
                "message_type_id": msg_type.value,
                "cluster_label": msg_type.name,
                "keyword_offset": 4,
                "keyword_length": 1,
                "total_length": len(raw),
                "body_offset": len(header),
                "body_length": len(body),
            })

        return X, y, metadata

    def generate_nfor_dataset(
        self,
        num_messages: int = 200,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[bytes], List[Union[int, str]], List[Dict[str, Any]]]:
        """
        Generates a dataset with Non-Fixed-Offset Region (NFOR) keyword fields (e.g., TLV encapsulated).
        Maintained for backward compatibility.

        Args:
            num_messages: Total number of messages to generate.
            random_seed: Optional random seed.

        Returns:
            Tuple of (X, y, metadata).
        """
        rng = random.Random(random_seed) if random_seed is not None else self._rng
        X: List[bytes] = []
        y: List[Union[int, str]] = []
        metadata: List[Dict[str, Any]] = []

        types = [MessageType.READ, MessageType.WRITE, MessageType.STATUS, MessageType.CONFIG]

        for i in range(num_messages):
            msg_type = rng.choice(types)
            prefix_len = rng.randint(2, 10)
            prefix = rng.randbytes(prefix_len)

            tlv = self._build_tlv(TLVType.CONFIG_PARAM, bytes([msg_type.value]))
            body = rng.randbytes(rng.randint(8, 25))

            raw = prefix + tlv + body

            X.append(raw)
            y.append(msg_type.name)
            metadata.append({
                "message_type": msg_type.name,
                "message_type_id": msg_type.value,
                "cluster_label": msg_type.name,
                "keyword_offset": prefix_len + 2,
                "keyword_length": 1,
                "total_length": len(raw),
                "body_offset": prefix_len,
                "body_length": len(tlv) + len(body),
            })

        return X, y, metadata