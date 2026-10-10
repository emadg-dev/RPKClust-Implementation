# RPKClust Implementation State & NetPlier Benchmark Diagnostics Report

> **Purpose:** Snapshot of the current `Implementation/` state, the NetPlier evaluation
> dataset configuration, the benchmark results, and a root-cause analysis of why
> five of eight protocols produce below-paper clustering metrics — **with solutions
> only (no fixes applied)**.
>
> **Reference materials analyzed alongside this report:**
> - `RPKClust-e.pdf` — the paper (RPKClust: Region-Partitioned Keywords Inference for Binary Protocol Reverse, The Computer Journal).
> - `RPKClust-SPEC.md` — normative implementation specification of the paper.
> - `NetPlier_RPKClust_Dataset_Integration_Guide.md` — evaluation/benchmark harness guide.
> - NetPlier source: `netplier/processing.py`, `netplier/netplier.py`, `data/README.md` (v1.0.2).
> - Netzob 1.0.2 source: `PCAPImporter.py`, `Session.py`.

---

## 1. Project overview

RPKClust infers the keyword field (message-type field) of an unknown binary protocol
from captured network traffic, then clusters application-layer messages by that keyword.
The keyword is found by a two-stage Bayesian model over byte-aligned (FOR) and TLV
(NFOR) field candidates; it is never given the ground-truth label.

The current `Implementation/` directory is a **research-quality prototype** that runs
RPKClust end-to-end on the eight NetPlier benchmarking PCAPs and reports clustering
metrics against protocol-documentation ground truth.

**NetPlier is used only as an external benchmark** — its protocol-specific ground-truth
rules (keyword offset, direction, message-type values) are reproduced for *evaluation
only*; RPKClust never receives any ground-truth signal during clustering.

---

## 2. Implementation layout

```text
Implementation/
├── main.py                          # benchmark driver, dataset dispatch, GT isolation
├── temp_evaluator.py                # RPKClustEvaluator — per-dataset diagnostics + artifacts
├── rpkclust/
│   ├── rpkclust.py                  # facade: fit() → B, candidates, keyword, labels (149 lines)
│   ├── optimizer.py                 # two-stage Bayesian inference (337 lines)
│   ├── constraints.py               # 4 NetPlier-style constraints (397 lines)
│   ├── semantic_rules.py            # 6 semantic detectors + boundary/keyword inference
│   ├── candidates.py                # candidate generation dispatch (via utils.py)
│   ├── metrics.py                   # evaluation metrics (h, c, v, ARI, NMI, Acc, timing)
│   └── utils.py                     # extract_for_candidates, extract_nfor_tlv_candidates
├── datasets/
│   ├── dataset_loader.py            # PcapDatasetLoader — PCAP/PCAPNG parsing + NetPlier GT rules
│   └── downloads/                   # the 8 NetPlier PCAPs + README.md
└── results/
    ├── tables/                      # summary_metrics.md, boundary_keyword_evaluation.md,
    │   │                            #   candidates_<protocol>_100.md (top-10 candidate rankings)
    └── figures/                     # stage_inference_<protocol>_100.png, candidate probability plots,
                                     #   keyword_candidate_count.png, benchmark comparison plots
```

### Key code locations

| Component | File | Key functions |
|---|---|---|
| Loader | `datasets/dataset_loader.py` | `extract_payloads_with_metadata`, `_iter_pcap`, `_iter_pcapng`, `_extract_transport_payload`, `_format_gt_label` (line ~1858) |
| RPKClust facade | `rpkclust/rpkclust.py:17-30` | `fit(X, interaction_metadata, capture_start, capture_end, direction_labels, ...)` |
| Boundary + semantics | `rpkclust/semantic_rules.py` | `identify_boundary`, `collect_semantic_regions`, 6 rule detectors |
| FOR candidates | `rpkclust/utils.py:29-34` | `extract_for_candidates(X, boundary_B, semantic_regions, candidate_lengths=(1,2,4))` |
| Stage-1 constraints | `rpkclust/constraints.py` | `message_similarity` (O(n²) at line 31-77), `remote_coupling` (84-218), `structural_consistency` (220-359), `dimensional_constraint` (366-397) |
| Stage-2 inference | `rpkclust/optimizer.py` | `compute_p_bit` (145-200), `compute_p_offset` (202-231), `bayesian_update` (253), `infer_keyword` (260-337) |
| Evaluation | `temp_evaluator.py` | `run_diagnostics` (189-327), `export_summary_artifacts` (329-...) |
| Metrics | `rpkclust/metrics.py` | `evaluate_clustering`, `evaluate_boundary`, `evaluate_keyword_inference` |
| Main driver | `main.py` | `NETPLIER_DATASETS`, `project_metadata_for_rpkclust` (76-116), `evaluate_pcap` (173-255), `main` (258-377) |

---

## 3. Dataset configuration

### 3.1 The eight NetPlier benchmarking PCAPs

All live in `Implementation/datasets/downloads/`:

| File | Protocol |
|---|---|
| `dhcp_100.pcap` | dhcp |
| `dnp3_100.pcap` | dnp3 |
| `icmp_100.pcap` | icmp |
| `modbus_100.pcap` | modbus |
| `ntp_100.pcap` | ntp |
| `smb_100.pcap` | smb |
| `smb2_100.pcap` | smb2 |
| `tftp_100.pcap` | tftp |

These are 100-message subsets of the full 1000-message NetPlier captures.

### 3.2 Loading behavior (NetPlier fidelity)

The loader reproduces NetPlier's preprocessing faithfully:

- **Capture format:** 7 of 8 PCAPs are **pcapng** (magic `0a0d0d0a`); only
  `dhcp_100.pcap` is classic PCAP. The loader handles both
  (SHB/IDB/EPB/PB/SPB blocks, `if_tsresol`/`if_tsoffset` options, per-interface
  link types).
- **Message model:** NetPlier pins Netzob 1.0.2 and calls
  `PCAPImporter.readFile(filePath=..., importLayer=5)` with
  `mergePacketsInFlow=False` → **one message per TCP/UDP packet payload**
  (no TCP reassembly, no protocol framing). The loader respects this and makes
  reassembly opt-in via `reassemble_tcp=False`.
- **ICMP:** parsed at layer 3, IPv4 header stripped via IHL before GT extraction.
- **MBAP/Modbus:** `signed=True` (NetPlier convention).
- **SMB/SMB2:** filtered by signature `data[4:8] ∈ {"ff534d42", "fe534d42"}`;
  messages truncated to 500 bytes; the 4-byte NetBIOS header is preserved.
- **MAX_LEN = 500** applied uniformly to all protocols.

### 3.3 Ground-truth rules (evaluation only — never fed to RPKClust)

| Protocol | GT field extraction | Direction rule |
|---|---|---|
| dhcp | `data[242:243]` | `op` field: 1=request, 2=response |
| dnp3 | `data[12:13]` | bit 7 of `data[3]`: 1=master/request, 0=outstation/response |
| icmp | `data[0:2]` (after IHL strip) | type lists: requests `[8,13,15,17,10]`, responses `[0,3,4,5,11,12,14,16,18,9]` |
| modbus | `data[7:8]` | dst port 502=request, src port 502=response |
| ntp | `data[0] & 0x07` | mode lists: requests `[1,3,5]`, responses `[2,4,6]` |
| smb | `data[8]` | `data[13] & 0x80 == 0` → request |
| smb2 | `int.from_bytes(data[16:18], "little")` | `flags & 1 == 0` → request |
| tftp | `data[0:2]` | session-based (first source = request side) |

### 3.4 GT isolation enforcement

`main.py` enforces that RPKClust receives no ground-truth signal:

- `RPKCLUST_METADATA_FIELDS` (main.py:56-68) allow-lists the metadata fields handed
  to `fit()`: timestamp, protocol, transport, direction, source/dest IP+port,
  session_id, packet_index, payload_length.
- `GROUND_TRUTH_FIELDS` (main.py:72-80) names the forbidden fields
  (`gt_value`, `gt_label`, `gt_source`, `true_keyword`, `keyword_offset`).
- `project_metadata_for_rpkclust()` (main.py:83-116) projects each metadata record
  through the allow-list and asserts no GT field survives.

**Note:** the metadata handed to RPKClust **does** include `direction`
(`"request"` / `"response"`), which is needed for the remote-coupling constraint
and the address rule. Direction is NetPlier metadata, not ground-truth type
labels — this is consistent with the guide.

---

## 4. Current benchmark results

File: `Implementation/results/tables/rpkclust_summary_metrics.md`

| Dataset | Msgs | B (inferred) | h | c | v | Acc | Time (s) | FOR cnds | NFOR cnds | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| dhcp_100 | 100 | 248 | 0.5181 | 1.0000 | 0.6826 | 0.65 | 74.56 | — | — | — |
| dnp3_100 | 114 | 13 | 0.9067 | 0.7537 | 0.8231 | 0.8246 | 0.74 | — | — | — |
| icmp_100 | 100 | 37 | 1.0000 | 1.0000 | 1.0000 | 1.00 | 4.01 | — | — | — |
| modbus_100 | 100 | 8 | 0.1674 | 0.1391 | 0.1519 | 0.18 | 0.55 | — | — | — |
| ntp_100 | 100 | 48 | 1.0000 | 0.8247 | 0.9039 | 0.91 | 2.27 | — | — | — |
| smb_100 | 91 | 33 | 0.7389 | 1.0000 | 0.8499 | 0.5055 | 7.10 | — | — | — |
| smb2_100 | 100 | 70 | 0.7522 | 1.0000 | 0.8586 | 0.54 | 15.73 | — | — | — |
| tftp_100 | 100 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.00 | 8.47 | — | — | — |

### Candidate-count context (from per-dataset candidate tables)

| Dataset | Top-1 selected offset | Top-1 p_bit | True keyword offset | True keyword p_bit | Stage1_Prob top-1 |
|---|---|---|---|---|---|
| modbus | 0 (transaction ID) | 0.9203 | 7 (function code) | 0.6056 | 1.0 |
| dnp3 | 2 (link length byte) | 0.4240 | 12 (function code) | 0.5386 | 1.0 |
| dhcp | 0 (op byte) | 0.5247 | 242 (option 53) | not in top 10 | 1.0 |
| smb | 8 ✓ | 0.7827 | 8 ✓ | — | 1.0 |
| smb2 | 16 ✓ | 0.6797 | 16 ✓ | — | 1.0 |
| ntp | 0 (byte) | 0.6388 | bits 5–7 of byte 0 | unreachable | 1.0 |
| icmp | 0 ✓ | 0.2986 | 0 ✓ | — | 1.0 |
| tftp | 1 | 0.7121 | 0 (width 1) | 0.7121 (rank 2) | 1.0 |

Key observation: **every** candidate in every per-protocol top-10 table has
`Stage1_Prob = 1.0` (i.e., `p_f = 1.0`). Stage 1 provides zero discrimination;
Stage 2 (`p_bit` × `p_offset`) decides everything.

---

## 5. Paper conformance baseline

From the PDF and `RPKClust-SPEC.md` (§9 conformance targets):

**Table 2 — true boundaries (100-message subset):**
| Protocol | True B | 100-msg target | Inferred |
|---|---|---|---|
| DNP3 | 14 | 14 (+2 tol) | 13 ✓ |
| Modbus | 8 | 8 (exact) | 8 ✓ |
| NTP | 48 | 48 (exact) | 48 ✓ |
| DHCP | 240 | 245 (+5 tol) | 248 (over by 3) |
| SMB | 29 | 32 (+3 tol) | 33 ✓ |
| SMB2 | 64 | 68 (+4 tol) | 70 (over by 2) |
| TFTP | 1 | 2 (+1 tol) | 2 ✓ |

**Table 3/4 — keyword inference (100 messages):**
| Protocol | 1st-place true keyword? | Our result |
|---|---|---|
| DNP3 | ✗ (offset 3, 82.67%) | ✗ (offset 2) |
| Modbus | ✓ (offset 7, 93.46%) | ✗ (offset 0) |
| NTP | ✗ (bits 5–7) | ✗ (offset 0) |
| DHCP | ✓ (offset 242, 80.62%) | ✗ (offset 0) |
| SMB | ✗ (offset 16, 71.75%) | ✓ (offset 8) * |
| SMB2 | ✗ (offset 16, 71.56%) | ✓ (offset 16) |

\* SMB offset 8 is listed as the true keyword in SPEC §4.6 guide and NetPlier rules
(`data[8]`); the paper Table 3 lists offset 9/14 for SMB at 100 messages. The
guide's `data[8]` is treated here as authoritative for the benchmark.

**Table 5 — timing:** paper reports avg 1.64s (boundary) / 74.02s (keyword) on
i7-8700K, 6-thread. Our DHCP boundary pass alone is 74.56s — already at the
paper's *keyword* budget, before keyword inference even starts.

**Headline metric:** paper reports mean over 8 protocols of h=0.959, c=0.941,
v=0.949 (7/8 protocols at 1.0/1.0/1.0; NTP is the single exception at
≈0.67/0.53/0.59). Current implementation mean: h≈0.76, c≈0.87, v≈0.83 —
dragged down by modbus, dhcp, smb, smb2.

**Boundary/keyword conformance table currently all N/A** —
`main.py:evaluate_pcap` (line 250) calls `run_diagnostics(X, y, dataset_name=...,
fit_kwargs=...)` and never passes `true_boundary` or `true_keyword_offset`.
The `run_diagnostics` signature (temp_evaluator.py:189-197) accepts them but
they are not supplied. See `results/tables/boundary_keyword_evaluation.md`
(all rows show `N/A`).

---

## 6. Root-cause analysis: why five protocols have low stats

### 6.1 modbus_100 — worst performer (h=0.167, Acc=0.18)

**Symptom:** boundary B=8 is correct, but only 18% clustering accuracy.

**Evidence (candidates_modbus_100.md):**
- Rank 1 = `FOR_Offset_0_W1` (transaction ID), p_bit=0.9203, posterior=1.0
- True keyword = offset 7 (function code), ranks only #7, p_bit=0.6056
- 13 predicted clusters vs 4 GT classes

**Root cause (chain):**
1. **Stage-1 saturation.** `dimensional_constraint`
   (`constraints.py:366-397`) returns a binary 0.95 (if `distinct_ratio ≤ 0.5`
   and `single_ratio ≤ 0.5`) or 0.1 (otherwise). Nearly every candidate
   passes the lenient 0.5 thresholds → `c4=0.95` for almost all candidates →
   `p_f` (naive Bayes, `optimizer.py:117-143`) saturates to 1.0. The entire
   top-10 shows `Stage1_Prob=1.0`. Stage 1 provides zero ranking signal.
2. **p_bit rewards high-entropy fields.** The bit-use constraint (Eq. 7–10,
   `optimizer.py:145-200`) measures how well a field's MSB distribution
   matches a uniform-random model `P(k) = 1 − 1/2^{MSB+1−k}`. The transaction ID
   is near-unique → its MSB distribution spreads across the full range →
   p_bit=0.92 (high). The true function code has only 4 distinct values
   `{01, 02, 04, 0f}` → MSBs concentrated at bit 0 → p_bit=0.61 (low). The
   constraint is fooled by entropy, not keywordness.
3. **p_offset compounds the bias.** `compute_p_offset` (`optimizer.py:225-231`)
   gives FOR candidates `max(0.95 − 0.01*offset, 0.7)` — 0.95 at offset 0,
   0.88 at offset 7. The posterior
   `P(K=1) = p_bit·p_offset·p_f / (p_bit·p_offset·p_f + (1−p_bit)(1−p_offset)(1−p_f))`
   (`optimizer.py:233-254`) is dominated by p_bit and p_offset when p_f≈1.
   Offset 0's (0.92, 0.95) beats offset 7's (0.61, 0.88) decisively.
4. **No de-circularization of message_similarity.** `message_similarity`
   (`constraints.py:46-77`) rewards any partition where within-cluster byte
   similarity exceeds between-cluster — clustering by a near-unique field
   trivially maximizes this (each cluster has 1–2 identical messages).

### 6.2 dnp3_100 — h=0.907 / c=0.754

**Symptom:** boundary B=13 is correct (paper: 14 at 100 messages, tolerance +2);
keyword is wrong.

**Evidence (candidates_dnp3_100.md):**
- Rank 1 = `FOR_Offset_2_W1` (link-layer length byte), p_bit=0.424
- True keyword = offset 12 (application function code), ranks #8, p_bit=0.5386

**Root cause:** same Stage-1 saturation + p_bit/p_offset bias. The length byte
at offset 2 is correlated with message type (different functions produce
different response lengths) → it partially matches GT → high h, but it's not
the true keyword → low c. The Sequence/Length semantic rule
(`semantic_rules.py`) fails to classify offset 2 as a length-determined field
and exclude it from FOR candidates; a length-determined field should not be a
viable keyword candidate.

### 6.3 dhcp_100 — h=0.518, Time=74.56s (worst performance)

**Symptom:** keyword offset 0 selected; true keyword offset 242 (DHCP option 53)
not in top 10; boundary B=248 vs true 240.

**Root cause (multiple):**
1. **Candidate explosion.** `extract_for_candidates` (`utils.py:29-34`) defaults to
   `candidate_lengths=(1, 2, 4)`. The paper (SPEC §11) uses L=1 only, giving an
   average of 9.625 candidates per protocol. With L∈{1,2,4}, DHCP generates ~460
   candidates (multi-byte windows multiply the candidate count). `RPKClust.fit()`
   (`rpkclust.py:17-30`) does not expose `candidate_lengths` as a parameter, so
   the (1,2,4) default cannot be overridden to match the paper.
2. **p_bit penalizes the true keyword.** DHCP option 53 has only 4–8 distinct
   small values → concentrated MSB distribution → low p_bit. Meanwhile random
   high-entropy fields in the 236-byte fixed header (xid, chaddr, sname, file)
   score high p_bit and occupy low offsets, winning the posterior.
3. **Boundary overshoot.** B=248 vs true B=240 — the boundary detector overshot
   by 8 bytes into the DHCP options region. While offset 242 is technically still
   inside [0, 248), the FOR is inflated and the inflated NFOR/FOR split distorts
   candidate generation.
4. **Performance.** The 74.56s boundary pass equals the paper's entire keyword
   budget. Root: `message_similarity` (`constraints.py:46-77`) is a pure-Python
   O(n²) all-pairs loop, called once per candidate (≈460 times) — O(C·n²) total.
   Additionally, `is_checksum` (`semantic_rules.py`) recomputes CRC-16/XOR/sums
   per message per offset per width without caching.

### 6.4 smb_100 / smb2_100 — h≈0.74 / h≈0.75, Acc≈0.51 / 0.54

**Symptom:** keyword offsets are CORRECT (8 and 16, matching the NetPlier/paper
true keyword). Metrics are low despite correct keyword selection.

**Evidence:** c=1.0 for both (completeness perfect), h≈0.74 (homogeneity low),
Acc≈0.5. `candidates_smb_100.md` shows rank-1 = `FOR_Offset_8_W1` (p_bit=0.78,
correct). `candidates_smb2_100.md` shows rank-1 = `FOR_Offset_16_W1` (p_bit=0.68,
correct).

**Root cause — harness label-space mismatch (NOT an algorithm failure):**

The label space is `_format_gt_label`
(`dataset_loader.py:1858-1870`) returns `f"{direction}:{protocol}:{gt_value}"`:

```
"request:smb:0x25"   "response:smb:0x25"   etc.
```

This **doubles** the GT label space: 8 SMB type values × 2 directions = 16 classes;
12 SMB2 type values × 2 directions = 21+ classes (per the diagnostics).

However, **RPKClust clusters on message content only.** The same command byte
(SMB `data[8]`, SMB2 `data[16:18]`) appears identically in both request and
response messages for a given type. The clustering cannot distinguish direction
from the keyword field alone → each predicted cluster contains messages from
both directions → every cluster is impure → **homogeneity capped at ≈0.74**
and **accuracy ≈0.5** (random when direction is the only differentiator).

This is confirmed by c=1.0: every true (direction:type) class is fully
contained within a single predicted cluster (the content-based keyword), so
completeness is satisfied — but the cluster also contains the other direction's
messages, so homogeneity is violated.

**NetPlier's actual evaluation methodology** (guide section 7 + NetPlier source
`divide_msgs_by_directionlist`): requests and responses are clustered as
**two separate problems**, each against type-only GT. The current harness
violates this by merging both directions into one label space.

### 6.5 ntp_100 — h=1.0 / c=0.825 (known paper limitation)

**Symptom:** h=1.0 but c=0.825 — keyword selected is offset 0 (whole byte)
instead of the true bit-level keyword (bits 5–7 of byte 0, the mode field).

**Root cause:** candidate extraction is **byte-aligned only** (widths 1/2/4
bytes). The true keyword is a 3-bit subfield of byte 0. RPKClust can never select
it — it clusters on the whole byte 0, which is shared by leap-indicator (bits
0-1), version (bits 2-4), and mode (bits 5-7). The version and leap bits split
some type classes across clusters → completeness < 1.

The paper itself reports this as NTP's known failure (paper §5.2: "NTP is the
exception"). Paper reports NTP at h≈0.67/c≈0.53/v≈0.59; the current
implementation at h=1.0/c=0.825/v=0.90 actually **exceeds** the paper on NTP.
So NTP is not a regression — it is the paper's documented design limitation.

---

## 7. Summary of root causes (grouped)

| # | Root cause | Affected protocols | Evidence location |
|---|---|---|---|
| 1 | Stage-1 saturation: `dimensional_constraint` binary (0.95/0.1) → `p_f`=1.0 for nearly all candidates; Stage 1 provides no ranking signal; Stage 2 defaults to p_bit×p_offset | modbus, dnp3, dhcp (all except perfect ones) | `constraints.py:366-397`; all `candidates_*.md` tables show `Stage1_Prob=1.0` |
| 2 | `p_bit` rewards high-entropy fields (near-unique values spread MSB distribution) over true low-cardinality keywords | modbus (transaction ID), dhcp (xid/chaddr), dnp3 (length byte) | `optimizer.py:145-200`; candidate tables |
| 3 | `p_offset` position prior favors low offsets (0.95 at offset 0, decaying 0.01/byte) | modbus, dhcp, dnp3 | `optimizer.py:225-231` |
| 4 | Candidate-space mismatch: `candidate_lengths=(1,2,4)` not exposed in `fit()`, paper uses L=1 → candidate explosion | dhcp (460 vs paper 9.625 avg) | `utils.py:33`; `rpkclust.py:17-30` |
| 5 | Direction baked into GT label space (`direction:protocol:value`) while clustering is content-only | smb (16 classes), smb2 (21 classes) | `dataset_loader.py:1858-1870`; c=1.0 vs h≈0.74 |
| 6 | Byte-aligned candidates cannot express sub-byte keywords | ntp (mode = bits 5–7) | `utils.py` candidate extraction; paper §5.2 |
| 7 | Length-determined fields not excluded from FOR candidates | dnp3 (offset 2, length byte) | `semantic_rules.py` Sequence/Length rule |
| 8 | Boundary overshoot (B too large) | dhcp (248 vs 240), smb2 (70 vs 68) | boundary inference logic |
| 9 | Performance: O(n²) `message_similarity` + uncached checksum recomputation | dhcp (74.56s vs paper ~1.6s budget) | `constraints.py:46-77`; `semantic_rules.py` |

**Note on "the field the paper cares most about" (Table 3/4 keyword inference):**
modbus and dhcp are the **hardest regressions** — both select the wrong keyword
where the paper claims rank-1 correctness at 100 messages (modbus 93.46%,
dhcp 80.62%). dnp3 is also wrong (paper: wrong at 100, correct at 500+).

---

## 8. Proposed solutions (no changes applied)

### Solution 1 — Grade Stage-1 constraints instead of binary thresholding (CAUSE 1)

**Problem:** `dimensional_constraint` (`constraints.py:366-397`) returns hard
0.95/0.1 based on two fixed 0.5 thresholds. This saturates `p_f` to 1.0 for
almost every candidate, eliminating Stage 1's ranking power.

**Fix:** Replace the binary step with a graded score — e.g., a logistic function
over the combined `distinct_ratio` and `single_ratio`, or a proper likelihood
P(constraints | keyword) estimated from real protocol data. This lets `p_f`
discriminate between strong and weak candidates so Stage 1's ranking survives
into Stage 2.

**Impact:** Directly fixes the modbus/dnp3/dhcp keyword mis-selection. If `p_f`
properly down-weights the transaction-ID/length/xid candidates (whose
`message_similarity` scores are artifactually high due to over-segmentation),
the true keyword's Stage-1 signal can surface.

### Solution 2 — De-circularize `message_similarity` and penalize over-segmentation (CAUSE 1 + 2)

**Problem:** `message_similarity` (`constraints.py:31-77`) rewards any partition
with high intra-cluster similarity — trivially achieved by clustering on a
near-unique field (each cluster = 1–2 messages). This is why the transaction ID
scores p_bit=0.92.

**Fix:** Normalize the intra/inter similarity by the expected similarity under
a null model (e.g., compare against randomly partitioned baselines, or penalize
excessive cluster counts). Alternatively, cap or penalize the single-message-
cluster ratio. The SPEC already documents this as a known concern
(`constraints.py:276-278`).

### Solution 3 — Expose `candidate_lengths` in `RPKClust.fit()` and default to `(1,)` (CAUSE 4)

**Problem:** `fit()` (`rpkclust.py:17-30`) does not expose `candidate_lengths`;
`extract_for_candidates` (`utils.py:33`) defaults to `(1, 2, 4)`, generating
≈460 candidates for DHCP vs the paper's 9.625 average.

**Fix:** Add `candidate_lengths: Tuple[int, ...] = (1,)` to the `fit()` signature
and thread it through. The paper uses L=1 (byte-level) as the primary candidate
set (SPEC §11). This alone reduces DHCP's candidate count by ~3× and removes
multi-byte decoys (`FOR_Offset_0_W2`, `FOR_Offset_0_W4`) that currently win on
p_bit.

**Impact:** Fixes the DHCP candidate explosion and partially fixes modbus/smb2
(whose top candidates include width-2/4 entries).

### Solution 4 — Evaluate requests and responses as two separate runs (CAUSE 5)

**Problem:** `_format_gt_label` (`dataset_loader.py:1858-1870`) produces
`f"{direction}:{protocol}:{gt_value}"` → 16 classes (SMB) / 21 (SMB2). RPKClust
clusters on content, which cannot distinguish direction → homogeneity capped at
≈0.74, accuracy ≈0.5 for both protocols. NetPlier evaluates requests and
responses as **separate** clustering problems (`divide_msgs_by_directionlist`).

**Fix (two-run approach):** In `main.py`, split each dataset into request and
response subsets by the direction field, run RPKClust on each subset separately
with GT = type-only (no direction prefix), and report metrics per direction +
merged average. This mirrors NetPlier's actual evaluation.

**Fix (alternative — single run):** If a merged run is preferred for
comparability, drop direction from the GT label and document the deviation.
This would give h≈1.0 (content-based keyword correctly separates types) but
would conflate the paper's definition (which separates directions).

**Impact:** Should restore SMB/SMB2 to h≈1.0, c≈1.0, Acc≈1.0 (their keyword
selection is already correct).

### Solution 5 — Add bit-level candidate extraction (CAUSE 6)

**Problem:** Candidate extraction produces byte-aligned slices only (widths 1/2/4
bytes). The true NTP keyword is bits 5–7 of byte 0 — unreachable. This is the
paper's known limitation (§5.2, marked [GAP] in SPEC §11).

**Fix:** Extend the candidate schema to support `(bit_offset, bit_width)` and
extract sub-byte fields. The `p_bit` model (Eq. 7–10) already operates on
integer values, so it would naturally handle sub-byte fields once they can be
generated. Mark this as an enhancement; it does not affect the 7 protocols whose
keywords are byte-aligned.

### Solution 6 — Exclude length-determined fields from FOR candidates (CAUSE 7)

**Problem:** The DNP3 link-layer length byte (offset 2) is fully determined by
message length — it should not be a keyword candidate — yet it ranks #1 for
DNP3.

**Fix:** In `semantic_rules.py`, the Sequence/Length rule should detect when a
field's value is a deterministic function of `len(message)` and exclude it from
the FOR candidate seed set, or at minimum down-weight its Stage-2 posterior.
More generally, any field whose cardinality is bounded by message length
(carried-length fields, padding lengths) should be filtered.

### Solution 7 — Re-derive and pass `true_boundary` / `true_keyword_offset` (CAUSE: missing evaluation data)

**Problem:** `main.py:evaluate_pcap` (line 250) never passes
`true_boundary` or `true_keyword_offset` to `run_diagnostics`, so the
boundary/keyword conformance table (`results/tables/boundary_keyword_evaluation.md`)
is all `N/A`, and keyword correctness cannot be reported.

**Fix:**
1. Add a `GROUND_TRUTH_KEYWORDS` mapping in `main.py`:
   ```python
   {
       "dnp3":   {"boundary": 14, "keyword": (12, 1)},
       "modbus": {"boundary": 8,  "keyword": (7, 1)},
       "ntp":    {"boundary": 48, "keyword": (0, 3, 5)},  # bits 5–7 of byte 0
       "dhcp":   {"boundary": 240, "keyword": (242, 1)},
       "smb":    {"boundary": 29, "keyword": (8, 1)},
       "smb2":   {"boundary": 64, "keyword": (16, 2)},
       "tftp":   {"boundary": 1,  "keyword": (0, 1)},
       "icmp":   {"boundary": None, "keyword": None},  # no paper targets
   }
   ```
2. Pass these as `true_boundary=...` and `true_keyword_offset=...` to
   `run_diagnostics`. Note: DHCP's paper keyword offset source is BinaryInferno
   (not SMIA 2011 like the paper's other datasets) — confirm the NetPlier GT
   rule matches (`data[242:243]`), which the loader already does.
3. This populates the boundary error table and the keyword rank/correctness
   columns in the summary.

**Note:** ICMP has no paper targets (it's not in the paper's 8 evaluated
protocols). DHCP's paper keyword (242) and boundary (240) are transferable from
the guide's GT rules; its source dataset differs (BinaryInferno vs SMIA 2011)
but the GT extraction is well-defined.

### Solution 8 — Fix boundary overshoot (CAUSE 8)

**Problem:** DHCP B=248 (true 240) and SMB2 B=70 (true 68) are slightly
overshot. For DHCP, the 8-byte overshoot pushes B past the magic cookie
boundary (236+4=240), into the options region (240+).

**Fix:** The boundary detector (`semantic_rules.py:identify_boundary`) should
stop at the last **detected** semantic hit + 1, not extend further. If the
DHCP magic cookie (0x63825363 at bytes 236–239) is detected as a constant
field, B should be 240. Verify the constant rule fires on the magic cookie
(4-byte constant, all 100 messages). For SMB2, investigate which semantic
region extends to 70.

### Solution 9 — Vectorize `message_similarity` and cache checksum scans (CAUSE 9)

**Problem:** DHCP takes 74.56s for boundary detection (paper budget ~1.6s).
`message_similarity` (`constraints.py:46-77`) does O(n²) Python-level byte
comparisons, called once per candidate (≈460 times for DHCP → O(460·100²) =
4.6M all-pairs comparisons in pure Python).

**Fix:**
1. Replace the Python all-pairs loop with a numpy operation: build a
   `n × n` pairwise byte-similarity matrix using broadcasting
   (`(X[:, None, :] == X[None, :, :])` then mask by length), compute intra/inter
   means by label groups. This is O(n²) but in C-level numpy — ~100× faster.
2. Cache checksum computations per (offset, width, algorithm) so they are not
   recomputed for every candidate that spans the same bytes.
3. Reduce candidate count via Solution 3 (L=1 only) — fewer `message_similarity`
   calls per dataset.

**Impact:** Should bring DHCP from 74.56s to <5s, matching the paper's
performance target.

---

## 9. Prioritized fix order

| Priority | Solution | Fixes | Est. impact |
|---|---|---|---|
| P0 | #1 (grade dimensional_constraint) + #2 (de-circularize message_similarity) | modbus, dnp3, dhcp keyword selection | +0.5 to mean h |
| P0 | #4 (per-direction evaluation) | smb, smb2 h/Acc | +0.25 to mean h |
| P1 | #3 (expose candidate_lengths, default L=1) | dhcp candidate explosion, modbus/smb2 decoys | +0.1 to mean h, fixes perf |
| P1 | #7 (pass true_boundary/true_keyword) | boundary/keyword conformance tables | evaluation completeness |
| P2 | #5 (bit-level candidates) | ntp sub-byte keyword | ntp (already exceeds paper) |
| P2 | #6 (exclude length fields) | dnp3 offset 2 decoy | +0.03 to dnp3 c |
| P2 | #9 (vectorize similarity + cache) | dhcp 74.56s runtime | ~75s → <5s |

---

## 10. NetPlier methodology notes (for Claude's reference)

- **NetPlier evaluates requests and responses separately** via
  `divide_msgs_by_directionlist` (`netplier/netplier.py`). The current harness
  violates this by merging both directions into a single label space
  (`netplier/processing.py` line ~162-170 confirms the separate-split approach).
- **NetPlier pins Netzob 1.0.2** and calls `PCAPImporter.readFile` with
  `importLayer=5` and `mergePacketsInFlow=False` — no TCP reassembly. The
  loader's `reassemble_tcp` option (guide §9) is an enhancement, not a
  NetPlier fidelity requirement.
- **DHCP ground-truth note:** the paper uses BinaryInferno's DHCP dataset,
  while the local PCAP is from `SMIA-2011` per SPEC §9.1. The GT rule
  (`data[242:243]`) is the same per NetPlier; only the source differs. The
  keyword offset (242) and boundary (240) are transferable from the guide's
  rules but should be validated against the NetPlier source for the SMIA data.
- **ZeroAccess** is referenced by NetPlier but has no local PCAP
  (`datasets/downloads/` does not contain `zeroaccess_100.pcap`); it is optional
  and does not affect the 8-protocol benchmark.

---

*Generated from the current `Implementation/` state. See
`results/tables/rpkclust_summary_metrics.md` and
`results/tables/candidates_<protocol>_100.md` for the source data.*
