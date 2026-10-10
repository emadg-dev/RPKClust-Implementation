# RPKClust Implementation Pipeline Documentation

## Overview

This document describes the standalone Python implementation of the RPKClust algorithm located in the `Implementation/` directory. This is a self-contained, scratch-built version of RPKClust that implements the full pipeline from PCAP/CSV loading through boundary detection, candidate generation, two-stage Bayesian inference, clustering, and evaluation.

### Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   PCAP/CSV   │────▶│  datasets/           │────▶│  RPKClust.fit()      │
│  Datasets    │     │  loader.py           │     │  (pipeline facade)   │
└──────────────┘     └─────────────────┬────┘     └──────────┬───────────┘
                                        │                     │
                                        │  X: List[bytes]      │
                                        │  y: np.ndarray       │
                                        │  metadata: List[dict]│
                                        ▼                     │
┌──────────────────────┐   ┌──────────────────────┐          │
│  constraints.py      │   │  semantic_rules.py   │          │
│  (4 constraints)     │   │  (6 semantic rules)  │          │
│  Stage 1 scoring      │   │  Boundary detection  │          │
└──────────┬───────────┘   └──────────┬───────────┘          │
           │                        │                        │
           │  Stage 2 scoring        │  BoundaryResult        │
           ▼                        ▼                        │
┌──────────────────────┐   ┌──────────────────────┐          │
│  utils.py            │   │  optimizer.py        │          │
│  Candidate gen       │   │  Two-stage Bayesian  │          │
│  FOR + NFOR          │   │  inference           │          │
└──────────┬───────────┘   └──────────┬───────────┘          │
           │                        │                        │
           │  candidates              │  KeywordResult        │
           │  scores                  │  best_candidate        │
           ▼                        ▼                        │
┌──────────────────────┐   ┌──────────────────────┐          │
│  metrics.py          │   │  cluster.py (?)      │          │
│  Evaluation          │   │  (inline in fit)     │          │
└──────────────────────┘   └──────────────────────┘          │
                                                              │
                                                              ▼
┌────────────────────────────────────────────────────────┐
│  Result: best_candidate, labels_, boundary_B,          │
│  semantic_regions, candidates, diagnostics               │
└────────────────────────────────────────────────────────┘

┌─────────────────────────┐     ┌─────────────────────────────┐
│  main.py                │     │  temp_evaluator.py          │
│  Benchmark driver       │────▶│  RPKClustEvaluator         │
│  (dataset orchestration)│     │  (diagnostics + artifacts)  │
└─────────────────────────┘     └─────────────────────────────┘
```

---

## Entry Points

### `main.py` — Benchmark Driver

**Purpose**: Loads NetPlier PCAP or CNNPRE CSV datasets, enforces ground-truth isolation, runs RPKClust evaluation, and exports artifacts.

### Key Constants
| Constant | Description |
|---|---|
| `NETPLIER_DATASETS` | 8 entries: dhcp, dnp3, icmp, modbus, ntp, smb, smb2, tftp |
| `CNNPRE_DATASETS` | 12 entries: dhcp, icmp, ntp, dns, http, ftp, smtp, pop3, nbns, arp, gsm, syslog |
| `RPKCLUST_METADATA_FIELDS` | 11-field allow-list (strips ground-truth before pipeline) |
| `GROUND_TRUTH_FIELDS` | 5 forbidden fields (gt_value, gt_label, gt_source, true_keyword, keyword_offset) |

### Data Flow
```
PCAP/CSV → PcapDatasetLoader / CNNPRECSVDatasetLoader
  → X: List[bytes], y: np.ndarray, metadata: List[dict]
  → project_metadata_for_rpkclust(metadata) → fit_metadata (GT stripped)
  → RPKClustEvaluator.run_diagnostics(X, y, dataset_name, fit_kwargs)
  → RPKClust.fit(X, interaction_metadata=fit_metadata, ...)
```

### `rpkclust/api_runner.py` — JSON API (shared from RPKClust-UI)
Not in this folder but shared from the main repo for API-level evaluation.

---

## Stage 1: Data Loading

### `datasets/dataset_loader.py` (2417 lines)

**Purpose**: Full PCAP/PCAPNG parser with NetPlier-compatible ground-truth extraction for 9 protocols.

### Module-Level Constants
| Constant | Value |
|---|---|
| `_CLASSIC_PCAP_MAGICS` | Dict: 4 magic → (endianness, timestamp_divisor) |
| `_PCAPNG_*` | Block type codes (SHB, IDB, PB, SPB, EPB) |
| `_SUPPORTED_LINKTYPES` | DLT_NULL, DLT_EN10MB(eth), DLT_LINUX_SLL, DLT_RAW, DLT_IPV4, DLT_IPV6, DLT_SLL2 |
| `_MAX_CAPTURED_PACKET` | 16 MB (16 × 1024 × 1024) |
| `_NETPLIER_GT_OFFSETS` | Per-protocol GT byte offsets (evaluation only) |

### Class: `PcapDatasetLoader`

```
PcapDatasetLoader(target_dir="datasets/downloads", allow_synthetic_fallback=False, use_netzob=False)
```

#### PCAP Parsing Pipeline
```
extract_payloads(pcap_path, protocol, min_length=1, reassemble_tcp=False)
  → _read_transport_records_custom(pcap_path, min_length)
    → _iter_capture(pcap_path)
      → _iter_classic_pcap() OR _iter_pcapng()
        → _strip_link_layer()  (Ethernet/SLL/Raw)
          → _extract_transport_info()  (IPv4 → TCP/UDP/ICMP)
            → _preprocess_payload()  (protocol-specific)
    → [_reassemble_tcp_streams()]  (optional, OFF by default)
    → _assign_session_directions()  (TFTP only)
    → _get_netplier_direction()  (per-protocol direction)
    → _get_netplier_gt()  (ground-truth keyword extraction)
```

#### PCAPNG Reader (`_iter_pcapng`)
- Parses block types: SHB (byte order magic `0x1A2B3C4D`), IDB (interface link type + options), EPB (Enhanced Packet), PB (Packet), SPB (Simple Packet).
- Establishes endianness from SHB, yields `(timestamp, linktype, packet, packet_index)` tuples.

#### Classic PCAP Reader (`_iter_classic_pcap`)
- Reads 20-byte global header, extracts linktype from bytes 16-20.
- Reads 16-byte per-packet headers: `ts_sec, ts_fraction, incl_len, orig_len`.
- Yields `(timestamp, linktype, packet, packet_index)`.

#### Transport Info Extraction (`_extract_transport_info`)
- Parses IPv4 header: version (4 bits), IHL, total_length, source/dest IP.
- Parses TCP: source/dest port, sequence number, flags, payload.
- Parses UDP: source/dest port, length, payload.
- Parses ICMP: payload only (IHL-stripped for IPv4/6).
- Returns dict with payload, protocol, IPs, ports, session_id, tcp_sequence, tcp_flags, ip_total_length.

#### Protocol Preprocessing (`_preprocess_payload`)
| Protocol | Preprocessing |
|---|---|
| **Modbus** | Parse MBAP length at `[4:6]` (signed), truncate to expected length |
| **SMB** | Filter: `payload[4:8] == b"\xffSMB"` |
| **SMB2** | Filter: `payload[4:8] == b"\xfeSMB"` |
| **ZeroAccess** | Decrypt via `_decrypt_zeroaccess()` |
| **DHCP/DNP3/ICMP/NTP/TFTP** | No extra preprocessing, truncate to 500 (`MAX_MESSAGE_LENGTH`) |

#### ZeroAccess Decryption (`_decrypt_zeroaccess`)
1. Reads 4-byte CRC32 header.
2. XOR-decrypts payload in 4-byte chunks with key `0x66747032`.
3. Rotates key left by 1 bit each iteration.

#### Direction Assignment (`_get_netplier_direction`)
| Protocol | Request condition | Response condition |
|---|---|---|
| dhcp | `payload[0] == 1` | `payload[0] == 2` |
| dnp3 | `payload[3] & 0x80 == 1` | `payload[3] & 0x80 == 0` |
| icmp | `type ∈ {8,13,15,17,10}` | `type ∈ {0,3,4,5,11,12,14,16,18,9}` |
| modbus | `dst_port == 502` | `src_port == 502` |
| ntp | `(payload[0] & 0x07) ∈ {1,3,5}` | `(payload[0] & 0x07) ∈ {2,4,6}` |
| smb | `payload[13] & 0x80 == 0` | `payload[13] & 0x80 != 0` |
| smb2 | `flags & 0x1 == 0` | `flags & 0x1 != 0` |
| zeroaccess | `payload[7] == 'g'` | `payload[7] ∈ {'r','n'}` |

#### Ground-Truth Extraction (`_get_netplier_gt`) — Evaluation Only
| Protocol | GT Field |
|---|---|
| dhcp | `payload[242:243].hex()` |
| dnp3 | `payload[12:13].hex()` |
| icmp | `payload[0:2].hex()` |
| modbus | `payload[7:8].hex()` |
| ntp | `payload[0] & 0x07` (int, bits 0-2) |
| smb | `payload[8]` (int, not hex) |
| smb2 | `struct.unpack("<H", payload[16:18])[0]` |
| tftp | `payload[0:2].hex()` |
| zeroaccess | `payload[4:8].hex()` |

#### TCP Reassembly (`_reassemble_tcp_streams`) — Optional, Deviates from NetPlier
- Reassembles TCP payloads per 5-tuple.
- Splits streams using protocol-specific framing (`_frame_length`):
  - **Modbus**: MBAP length at `[4:6]`
  - **SMB/SMB2**: NetBIOS signature + 24-bit length
  - **DNP3**: `0x0564` magic + length byte

#### Output
Returns `(payloads: List[bytes], labels: np.ndarray, metadata: List[dict])`.
- `metadata` contains: `packet_index`, `timestamp`, `protocol`, `transport`, `direction`, `gt_value`, `gt_label`, `gt_source`, `source_ip`, `source_port`, `destination_ip`, `destination_port`, `session_id`, `payload_length`, `source_file`.

---

## Stage 2: Boundary Detection

### `rpkclust/semantic_rules.py` (539 lines)

**Purpose**: Six semantic detectors for boundary identification with first-match precedence.

### Class: `SemanticRules`

**`RULE_REGISTRY`** — 6 rule dicts defining the detection pipeline:

| Order | Rule | Widths | Type | Function |
|---|---|---|---|---|
| 1 | `constant` | [8, 4, 2, 1] | single | `is_constant()` |
| 2 | `sequence` | [1, 2, 3, 4] | single | `is_sequence()` |
| 3 | `timestamp` | [4, 8] | single | `is_timestamp()` |
| 4 | `sparse` | [1, 2] | single | `is_sparse()` |
| 5 | `address` | [2, 3, 4] | pair | `is_address()` |
| 6 | `checksum` | [1, 2] | single | `is_checksum()` |

### Rule Details

**Rule 1: Constant** (`is_constant`)
- All fragments identical at the offset across messages.
- **Production**: requires `len(fragments) >= 4` (was `>= 2` in notebook version).
- Excludes all-zero and all-0xFF fields.

**Rule 2: Sequence** (`is_sequence`)
- All values form an arithmetic sequence with positive constant delta.
- Requires `len >= 3` messages.

**Rule 3: Timestamp** (`is_timestamp`)
- All decoded values fall within `[capture_start - 86400, capture_end + 86400]` (±24h tolerance).
- Returns False if capture window not provided.

**Rule 4: Sparse** (`is_sparse`)
- `unique_count / 2^(8*width) <= threshold` (0.01 for width=1, 0.02 for width=2).
- Zero values excluded from the unique count.

**Rule 5: Address** (`is_address`)
- **With direction_labels**: checks within each direction class, field-1 and field-2 are each constant, and classes use swapped pairs.
- **Without direction_labels**: reciprocal-pair heuristic + Pearson correlation `rho <= -0.8`.

**Rule 6: Checksum** (`is_checksum`)
- Width 1: XOR-8 (`_calc_xor8`).
- Width 2: CRC-16/CCITT (`_calc_crc16_ccitt`) or RFC 1071 (`_calc_internet_checksum`).
- Width 4: CRC-32 (strict=False only).
- **Strict mode**: exact match required. Non-strict: tolerance for non-deterministic fields.
- **Supported checksum algorithms**: XOR-8, CRC-16/CCITT-FALSE, RFC 1071 Internet Checksum, CRC-32.

### Boundary Scanning (`_scan_semantic_regions`)

**Algorithm** (iterates offsets 0 to `min_len - 1`):

1. **Continuity guard**: If `offset > current_max_boundary + allowed_gap`, break (FOR region ended).
2. **NFOR TLV guard**: If `offset >= 2` and `_is_tlv_start()` returns True for this offset, break (valid TLV structure = NFOR boundary).
3. **Rule evaluation**: For each rule in `RULE_REGISTRY`, try all widths in order. First match wins (first-match precedence).
   - Single rules: extract fragments at `(offset, width)`, call `is_*()`.
   - Pair rules: extract two adjacent fields, call `is_address()`.
4. **Record hit**: `{"name": rule_name, "offset": offset, "width": matched_width}`, update `current_max_boundary = offset + width - 1`.
5. Returns list of region dicts.

### Boundary Computation (`identify_boundary`)
- Calls `_scan_semantic_regions()`.
- For each region, adds `offset + width - 1` to `hit_offsets`.
- Returns `(B = max(hit_offsets) + 1, hit_offsets_set)`.
- If no regions found: returns `(0, set())`.

---

## Stage 3: Candidate Generation

### `rpkclust/utils.py` (418 lines)

### FOR Candidate Generation (`extract_for_candidates`)

**Algorithm 2: Region-Partitioned Keyword Candidates in FOR.**

1. **Set construction**:
   - `FOR = set(range(0, min(boundary_B, min_len)))`.
   - Partition `semantic_regions` into `sparse_regions` (name == "sparse") and `excluded_regions` (all others).
   - `E = all_semantic_offsets - sparse_offsets` (excluded offsets minus sparse keyword-like fields).
   - `F_sparse = sparse_starts & FOR` (starting offsets of detected sparse fields).
   - `U = FOR - E` (undetected offsets).
   - `S = (U | F_sparse) & FOR` (scanning sequence: undetected + sparse starts).

2. **Candidate extraction**: For each length `L` in `candidate_lengths` (default: `(1, 2, 4)`), for each offset `s` in sorted `S`:
   - Skip if `s + L > max_offset`.
   - Skip if `s % L != 0` (length-aligned requirement).
   - Skip if `L > 1` and interval overlaps excluded offsets.
   - Extract `msg[s:s+L]` from all messages.
   - Create candidate: `{"type": "FOR", "tag": f"FOR_Offset_{s}_W{L}", "offset": s, "width": L, "values": [...]}`.

### NFOR TLV Candidate Generation

#### `_parse_tlv_at` (single TLV parse)
- Extracts type_bytes (`t_len`), len_bytes (`l_len`), value_bytes (`len_val`), tv_bytes (type+value), tlv_bytes (full TLV).

#### `extract_nfor_tlv_patterns` (Algorithm 3: Keyword Candidate Generation in NFOR)

1. Slice messages to NFOR: `msg[boundary_B:]`.
2. For each message, scan offsets 0 to `len(m) - (t_len + l_len)`:
   - Parse TLV at offset via `_parse_tlv_at`.
   - If invalid: `offset += 1`, continue.
   - If `validate_tlv()` rejects: `offset += 1`, continue.
   - Record in `P` (pattern list) with message_index, relative/absolute offsets.
   - **Repeated TLV detection**: While `_detect_repeated_tlv` succeeds at the `end` position, append to `P` (if `include_repeated_in_P`), track in `B` (repeated boundary list).
   - Advance `offset = end`.

#### `extract_nfor_tlv_candidates` (aggregation wrapper — not part of Algorithm 3)
- Calls `extract_nfor_tlv_patterns()`.
- Groups TLV records by `type_val` across messages (first occurrence per message).
- For each type: builds `values` (T-V combined bytes per message), `offsets`, `valid_count`, `patterns`, `repeated_boundaries`.
- Returns candidate dicts: `{"type": "NFOR", "tag": f"NFOR_TV_Type_{type_val}", ...}`.

---

## Stage 4: Two-Stage Bayesian Inference

### `rpkclust/optimizer.py` (399 lines)

**Purpose**: Two-stage probabilistic model combining clustering constraints with self-constraints for keyword field identification.

### Class: `RPKClustOptimizer`

| Constant | Value | Description |
|---|---|---|
| `CONSTRAINT_CLIP` | `(0.05, 0.95)` | Clip range for constraint probabilities before fusion |

### Stage 1: Constraint Probabilities

**4 NetPlier-style constraints** (from `constraints.py`):

| # | Constraint | Function | Description |
|---|---|---|---|
| 1 | **Message Similarity** | `message_similarity` | O(n²) all-pairs byte similarity. `p_m = 1 - (FMR + FNMR) / 2`. |
| 2 | **Remote Coupling** | `remote_coupling` | Chance-corrected request→response cluster purity. |
| 3 | **Structural Consistency** | `structural_consistency` | Intra-cluster length + byte agreement consistency. |
| 4 | **Dimensional** | `dimensional_constraint` | Distinct/total ratio penalty. |

#### Constraint 1: Message Similarity
```
message_similarity(labels, X) -> float
```
- Computes byte similarity for ALL message pairs: `sim = matching_bytes / max(len(a), len(b))`.
- Splits into intra-cluster and inter-cluster score distributions.
- Threshold = midpoint of intra/inter means.
- `FMR = mean(inter_scores >= threshold)`, `FNMR = mean(intra_scores <= threshold)`.
- `p_m = 1 - (FMR + FNMR) / 2`, clipped to [0, 1].
- **Performance bottleneck**: O(n²) all-pairs loop without optimization.

#### Constraint 2: Remote Coupling
```
remote_coupling(labels, X, interaction_metadata=None) -> float
```
- **Chance-corrected** request→response cluster purity.
- Pairs client messages with nearest subsequent server messages by timestamp (within session if available).
- Only client clusters with ≥2 pairs are informative.
- `observed = weighted_purity(client_labels, server_labels)`.
- `baseline = mean(weighted_purity(client_labels, rng.permutation(server_labels)))` over 20 permutations (seed=0).
- `corrected = (observed - baseline) / (1 - baseline)`, clipped to [-1, 1].
- `p = 0.5 + 0.5 * corrected`, shrunk by coverage.
- Returns 0.5 (neutral) when metadata missing, single-direction, or no pairs.

#### Constraint 3: Structural Consistency
```
structural_consistency(labels, candidate, X=None) -> float
```
- **With X (preferred)**: Checks intra-cluster length consistency (coefficient of variation) and byte-level agreement at non-keyword offsets. Combined: `0.4 * len_score + 0.6 * byte_agreement`, shrunk toward 0.5 by singleton coverage.
- **Without X (fallback, warns)**: Candidate-only checks. Labeled "NOT paper-accurate due to circularity."

#### Constraint 4: Dimensional Constraint
```
dimensional_constraint(labels) -> float
```
- **Production version (graded)**: `d = min(1.0, r_distinct / 0.5)`, `penalty = max(d², r_single)`, returns `0.95 - 0.85 * min(1.0, penalty)`. Range: [0.1, 0.95].
- **Root cause of saturation**: The binary version (0.95/0.1) saturates Stage 1 scores.

#### Stage 1 Scoring

**`_constraint_log_odds(c1, c2, c3, c4, prior=0.1) -> float`** (production)
- Naive-Bayes log-odds: `sum(log(c_i) - log1p(-c_i)) + (log(prior) - log1p(-prior))`.
- Uses log-odds form to prevent probability ceiling ties.

**`_constraint_bayesian_update(c1, c2, c3, c4, prior=0.1) -> float`** (notebook version)
- Naive Bayes posterior: `prod(c_i) * prior / (prod(c_i) * prior + prod(1-c_i) * (1-prior))`.

**`compute_stage1_probability(candidate, X, interaction_metadata=None, prior=0.1) -> Tuple[float, Dict]`**
- Clusters messages by candidate values, computes 4 constraints.
- `p_f = clip(sigmoid(log_odds), 1e-6, 1-1e-6)`.

### Stage 2: Self-Constraints

#### Bit-Use Constraint (Equations 7-10)
```
compute_p_bit(values) -> float
```
- Tests how well a field's MSB distribution matches uniform-random.
- Guards: empty → 1e-6, single distinct → 1e-6, max_val=0 → 1e-6, MSB > 64 → 1e-6.
- `Q(k) = proportion of values with MSB >= k`.
- `P(k) = 1 - 1/2^(MSB+1-k)` (uniform-random model).
- `D = sqrt(sum((Q(k) - P(k))²))`.
- `D_max` computed via cumulative sums (O(MSB) complexity).
- `p_bit = 1 - D / D_max`, clipped to [1e-6, 1-1e-6].
- **Key insight**: Rewards high-entropy fields over true low-cardinality keywords (root cause 2).

#### Position Constraint (Equation 11)
```
compute_p_offset(candidate_type, offset=0, boundary_B=0) -> float
```
- **FOR**: `max(0.95 - 0.01 * offset, 0.70)` — base 0.95, slope -0.01/byte, floor 0.70.
- **NFOR**: Fixed 0.60.
- **Root cause**: Position prior favors low-offset fields (root cause 3).

#### Stage 2 Scoring

**`final_log_odds(p_bit, p_offset, stage1_log_odds) -> float`** (production)
- `logit(p_bit) + logit(p_offset) + stage1_log_odds` — unclipped log-odds for ranking.

**`bayesian_update(p_bit, p_offset, p_f) -> float`** (paper Equations 12-15)
- `M = p_bit * p_offset * p_f`, `N = (1-p_bit)(1-p_offset)(1-p_f)`.
- Posterior = `M / (M + N)`.

**`score_candidate(candidate, p_f, boundary_B) -> Dict[str, float]`**
- Shared Stage 2 scorer used by both `fit()` and `infer_keyword()`.
- Computes p_bit, p_offset, calls `final_log_odds`, returns dict with all values.

### Full Inference (`infer_keyword`)
1. Stage 1: `rank_stage1_candidates()` using log-odds.
2. Optional top_k truncation.
3. Stage 2: `score_candidate()` for each ranked candidate.
4. Sort by log-odds descending.
5. Returns `(best_candidate, best_prob, all_scored)`.

---

## Stage 5: Clustering

### Inline in `RPKClust.fit()` and `optimizer.py`

```
RPKClustOptimizer.cluster_by_candidate(values: List[Any]) -> np.ndarray
```
- Maps each unique candidate field value to an integer cluster label.
- Uses `_MISSING` sentinel for None values (NFOR TLV absent).

```
RPKClust._assign_clusters(X) -> np.ndarray
```
- Clusters messages by best candidate's values.
- Returns all-zeros if no best candidate found.

---

## Stage 6: Evaluation Metrics

### `rpkclust/metrics.py` (292 lines)

```
convert_bytes_to_feature_matrix(messages, max_len=None) -> np.ndarray
```
- Pads/crops messages to fixed width, returns float64 matrix.

```
clustering_accuracy(labels_true, labels_pred) -> float
```
- Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for optimal label assignment.

```
measure_memory_usage() -> float
```
- Returns RSS in MB via `psutil`, or 0.0 if unavailable.

```
evaluate_boundary(true_boundary, inferred_boundary) -> Dict[str, Any]
```
- Returns True Offset, Inferred Offset, Error, Error %.

```
evaluate_clustering(labels_true, labels_pred, feature_matrix=None, exec_time=0.0, memory_mb=None) -> Dict[str, Any]
```
- **Core metrics**: Homogeneity (Eq. 18), Completeness (Eq. 19), V-Measure (Eq. 20), ARI, NMI, clustering accuracy.
- **Timing**: Execution Time (s).
- **Memory**: Peak memory in MB.
- **Internal metrics**: Silhouette, Davies-Bouldin (only if `feature_matrix` provided and ≥2 clusters).
- **Single-class guard**: Returns NaN for homogeneity/completeness/V-measure/ARI/NMI if `num_true_classes <= 1`.

```
evaluate_keyword_inference(candidates, true_keyword_offset) -> Dict[str, Any]
```
- Checks if top-ranked candidate offset matches true keyword offset.
- Supports: tuple/list of acceptable offsets, string `"start:end"`, or plain int.
- Returns: Correct (bool), Rank (1-based), Probability, Top-3 Candidates.

---

## Pipeline Orchestration (`RPKClust.fit`)

### `rpkclust/rpkclust.py` (142 lines)

```
RPKClust.fit(self, X, interaction_metadata=None, capture_start=None,
    capture_end=None, timezone_offset=0.0, direction_labels=None,
    t_len=1, l_len=1, validate_tlv=None, stage1_prior=0.1, top_k=None)
    -> "RPKClust"
```

**5-stage pipeline:**

| Stage | Component | Function | Output |
|---|---|---|---|
| 1 | Boundary | `SemanticRules.identify_boundary()` | `boundary_B: int`, `hit_offsets: Set[int]` |
| 2 | Regions | `SemanticRules.collect_semantic_regions()` | `semantic_regions: List[Dict]` |
| 3 | Candidates | `utils.extract_for_candidates()` + `extract_nfor_tlv_candidates()` | `candidates: List[Dict]` |
| 4 | Inference | `optimizer.rank_stage1_candidates()` → `score_candidate()` | `best_candidate`, per-candidate scores |
| 5 | Clustering | `_assign_clusters()` | `labels_: np.ndarray` |

**Key deviation from RPKClust-UI**: Uses **log-odds** for ranking in both Stage 1 and Stage 2 (while RPKClust-UI uses probability forms). Candidates are scored dictionaries (not `Candidate` dataclass), and clustering is done inline rather than via a separate `cluster.py` module.

### Candidate Scoring in `fit()`
After Stage 1 ranking and Stage 2 scoring, each candidate dict is augmented with:
- `stage1_prob` (sigmoid probability from log-odds)
- `p_bit` (bit-use constraint)
- `p_offset` (position constraint)
- `score` (final log-odds for ranking)
- `prob` (sigmoid(score))

---

## Dataset Catalog

### Supported Protocols
| Protocol | Transport | Direction Logic | GT Offset |
|---|---|---|---|
| dhcp | udp | `payload[0]` (1=req, 2=resp) | 242 |
| dnp3 | tcp | `payload[3] & 0x80` | 12 |
| icmp | icmp | ICMP type classification | 0 |
| modbus | tcp | Port 502 | 7 |
| ntp | udp | Bits 0-2 of byte 0 | 0 |
| smb | tcp | `payload[13] & 0x80` | 8 |
| smb2 | tcp | Flags bit 0 | 16 |
| tftp | udp | (filename-based) | 0 |
| zeroaccess | tcp | `payload[7]` | 4 |

### NetPlier Ground-Truth Offsets
```
_NETPLIER_GT_OFFSETS = {
    "dhcp":       (242, 243),
    "dnp3":       (12, 13),
    "icmp":       (0, 2),
    "modbus":     (7, 8),
    "ntp":        (0,),
    "smb":        (8,),
    "smb2":       (16, 18),
    "tftp":       (0, 2),
    "zeroaccess": (4, 8),
}
```

---

## Experiments & Benchmarking

### `experiments/compare_baselines.py`
```
run_baseline_comparison(messages, labels_true) -> pd.DataFrame
```
- Compares RPKClust against: K-Means, DBSCAN, GMM, Spectral Clustering.
- Returns DataFrame with ARI, NMI, V-Measure, Silhouette, Davies-Bouldin, Execution Time.

### `experiments/parameter_analysis.py`
- `analyze_sample_size_scalability()`: Tests RPKClust at 5 sample sizes [100, 250, 500, 1000, 2000].
- `analyze_offset_shift_impact()`: Demonstrates RPKClust's robustness to keyword offset shifts vs. K-Means failure.

---

## Known Issues & Limitations

Per `RESULTS_ANALYSIS_REPORT.md` — root-cause analysis with 9 identified issues (solutions proposed but not applied):

| # | Issue | Affected | Proposed Fix |
|---|---|---|---|
| 1 | Stage-1 saturation from binary `dimensional_constraint` | modbus, dnp3, dhcp | Grade the constraint instead of binary thresholding |
| 2 | `p_bit` rewards high-entropy fields over low-cardinality keywords | modbus, dhcp, dnp3 | De-circularize `message_similarity` |
| 3 | `p_offset` position prior favors low offsets | modbus, dhcp, dnp3 | Evaluate requests/responses separately |
| 4 | `candidate_lengths=(1,2,4)` not exposed in `fit()` | dhcp (460 candidates) | Expose `candidate_lengths` in `fit()`, default to `(1,)` |
| 5 | Direction baked into GT label space | smb, smb2 | Evaluate per-direction separately |
| 6 | No bit-level candidate extraction | ntp (mode = bits 5-7) | Add bit-level candidate extraction |
| 7 | Length-determined fields not excluded | dnp3 (offset 2) | Exclude length fields from FOR candidates |
| 8 | Boundary overshoot | dhcp (248 vs 240) | Fix boundary inference |
| 9 | O(n²) `message_similarity` + uncached checksums | dhcp (74s) | Vectorize + cache |

### Data Isolation
- `main.py` enforces strict ground-truth isolation: `project_metadata_for_rpkclust()` strips all `gt_*` fields from metadata before passing to `RPKClust.fit()`. This prevents the algorithm from accidentally learning from evaluation labels.

### Scratch Scripts (non-functional)
- `_scratch_check.py` and `_scratch_time.py` import `CsvDatasetLoader` from `datasets`, but `datasets/__init__.py` does not export it — these scripts will fail on import.
- `experiments/parameter_analysis.py` imports from `datasets.generate_data` which exists only as notebook cells and `.pyc` cache.

---

## Module Dependency Graph

```
model.py              (no deps)
config.py             (no deps)
dataset_loader.py     → (struct, json, collections, np — no internal deps)
csv_dataset_loader.py → (pandas, numpy)
semantic_rules.py     → (struct, itertools — no internal deps)
utils.py              → (semantic_rules for type defs)
constraints.py        → (np, scipy.stats)
metrics.py            → (np, scipy.stats)
optimizer.py          → constraints, model
rpkclust.py           → optimizer, semantic_rules, utils, metrics
main.py               → rpkclust, datasets, temp_evaluator
temp_evaluator.py     → rpkclust, metrics
experiments/*         → rpkclust, metrics, datasets
```
