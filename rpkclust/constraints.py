import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class ClusteringConstraints:

    # Direction vocabularies accepted in interaction_metadata["direction"].
    # NetPlier's ground-truth extractor emits "request"/"response";
    # some harnesses use the client/server role naming.
    CLIENT_LABELS = ("client", "request")
    SERVER_LABELS = ("server", "response")

    # Number of server-label shuffles used for the chance baseline in
    # remote_coupling.
    COUPLING_PERMUTATIONS = 20

    # k/n at which dimensional_constraint is fully penalized (the old
    # step function's threshold, now an end point instead of a pass mark).
    DIM_DISTINCT_LIMIT = 0.5

    # ==============================================================
    #  1. Message Similarity Constraint
    # ==============================================================

    @staticmethod
    def _byte_similarity(a: bytes, b: bytes) -> float:
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0

        min_len = min(len(a), len(b))
        same = sum(1 for i in range(min_len) if a[i] == b[i])
        return same / max_len

    @staticmethod
    def message_similarity(
        labels: np.ndarray,
        X: List[bytes],
    ) -> float:
       
        n = len(labels)
        if len(X) != n:
            raise ValueError("labels and X must contain the same number of messages")
        if n < 2:
            return 0.0

        intra_scores: List[float] = []
        inter_scores: List[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                similarity = ClusteringConstraints._byte_similarity(
                    X[i], X[j]
                )

                if labels[i] == labels[j]:
                    intra_scores.append(similarity)
                else:
                    inter_scores.append(similarity)

        if not intra_scores:
            return 0.0
        if not inter_scores:
            # All messages in one cluster — high intra similarity.
            return float(np.mean(intra_scores))

        intra_arr = np.array(intra_scores)
        inter_arr = np.array(inter_scores)

        threshold = (np.mean(intra_arr) + np.mean(inter_arr)) / 2.0

        # False match rate: inter scores above threshold.
        false_match = np.mean(inter_arr >= threshold)
        # False non-match rate: intra scores below threshold.
        false_nonmatch = np.mean(intra_arr <= threshold)

        # Constraint probability = 1 - total error rate.
        error_rate = (false_match + false_nonmatch) / 2.0
        p_m = 1.0 - error_rate

        return float(np.clip(p_m, 0.0, 1.0))

    # ==============================================================
    #  2. Remote Coupling Constraint
    # ==============================================================

    @staticmethod
    def remote_coupling(
        labels: np.ndarray,
        X: List[bytes],
        interaction_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
       
        if len(X) != len(labels):
            raise ValueError("labels and X must contain the same number of messages")
        if interaction_metadata is None:
            warnings.warn(
                "Remote coupling constraint requires interaction_metadata "
                "(source/dest IP, ports, direction, session_id, timestamps). "
                "Returning neutral score 0.5 — NOT paper-accurate.",
                UserWarning,
                stacklevel=2,
            )
            return 0.5

        n = len(labels)
        if len(interaction_metadata) != n:
            raise ValueError("interaction_metadata must contain one entry per message")
        if n < 2:
            return 0.0

        # Extract direction and session info from metadata.
        directions: List[str] = []
        session_ids: List[Any] = []
        timestamps: List[float] = []

        for meta in interaction_metadata:
            directions.append(meta.get("direction", "unknown"))
            session_ids.append(meta.get("session_id"))
            timestamps.append(meta.get("timestamp", 0.0))

        # Check if we have enough metadata for pairing.
        has_sessions = any(s is not None for s in session_ids)
        has_directions = any(d != "unknown" for d in directions)

        if not has_directions:
            warnings.warn(
                "Remote coupling constraint requires 'direction' in "
                "interaction_metadata. Returning neutral 0.5.",
                UserWarning,
                stacklevel=2,
            )
            return 0.5

        # Separate messages by direction.
        client_indices = [
            i
            for i in range(n)
            if directions[i] in ClusteringConstraints.CLIENT_LABELS
        ]
        server_indices = [
            i
            for i in range(n)
            if directions[i] in ClusteringConstraints.SERVER_LABELS
        ]

        if not client_indices or not server_indices:
            # All messages from one direction — cannot check coupling.
            return 0.5

        # Build request-response pairs.
        # Strategy: within each session, pair client messages with the
        # nearest subsequent server message by timestamp.
        pairs: List[Tuple[int, int]] = []

        if has_sessions:
            # Group by session_id.
            client_by_session: Dict[Any, List[int]] = defaultdict(list)
            server_by_session: Dict[Any, List[int]] = defaultdict(list)

            for i in client_indices:
                client_by_session[session_ids[i]].append(i)
            for i in server_indices:
                server_by_session[session_ids[i]].append(i)

            for sess_id in client_by_session:
                if sess_id not in server_by_session:
                    continue
                c_msgs = sorted(
                    client_by_session[sess_id],
                    key=lambda i: timestamps[i],
                )
                s_msgs = sorted(
                    server_by_session[sess_id],
                    key=lambda i: timestamps[i],
                )

                # Pair each request with the nearest unused subsequent response.
                available_servers = list(s_msgs)
                for ci in c_msgs:
                    matching = [si for si in available_servers if timestamps[si] >= timestamps[ci]]
                    if matching:
                        best_si = min(matching, key=lambda si: timestamps[si] - timestamps[ci])
                        pairs.append((ci, best_si))
                        available_servers.remove(best_si)
        else:
            # No session info — pair by timestamp proximity globally.
            sorted_clients = sorted(client_indices, key=lambda i: timestamps[i])
            sorted_servers = sorted(server_indices, key=lambda i: timestamps[i])

            available_servers = list(sorted_servers)
            for ci in sorted_clients:
                matching = [si for si in available_servers if timestamps[si] >= timestamps[ci]]
                if matching:
                    best_si = min(matching, key=lambda si: timestamps[si] - timestamps[ci])
                    pairs.append((ci, best_si))
                    available_servers.remove(best_si)

        if not pairs:
            return 0.5

        # ---- Chance-corrected request->response cluster purity ----
        # The raw "dominant server cluster" ratio is 1.0 for singleton
        # clusters and for any field that is a pure function of direction
        # (e.g. DHCP op byte), so it rewards over-segmentation and
        # direction proxies.  Only client clusters holding >= 2 pairs carry
        # evidence; for those we subtract the purity expected when the
        # server-side labels are shuffled (fixed seed -> deterministic) and
        # rescale, so chance level maps to the neutral value 0.5.  The
        # result is shrunk towards 0.5 by the share of pairs that sit in
        # informative clusters.
        c_lab = np.array([int(labels[ci]) for ci, _ in pairs])
        s_lab = np.array([int(labels[si]) for _, si in pairs])

        uniq, cnts = np.unique(c_lab, return_counts=True)
        informative = set(uniq[cnts >= 2].tolist())
        keep = np.array([c in informative for c in c_lab])
        coverage = float(keep.sum()) / len(c_lab)
        if coverage == 0.0:
            return 0.5
        c_lab, s_lab = c_lab[keep], s_lab[keep]

        def _weighted_purity(c_arr: np.ndarray, s_arr: np.ndarray) -> float:
            total = 0
            for c in np.unique(c_arr):
                _, counts = np.unique(s_arr[c_arr == c], return_counts=True)
                total += counts.max()
            return total / len(c_arr)

        observed = _weighted_purity(c_lab, s_lab)
        rng = np.random.default_rng(0)
        baseline = float(np.mean([
            _weighted_purity(c_lab, rng.permutation(s_lab))
            for _ in range(ClusteringConstraints.COUPLING_PERMUTATIONS)
        ]))

        if baseline >= 1.0 - 1e-9:
            # Chance already yields perfect purity (e.g. one server
            # cluster, which is what a direction-proxy field produces).
            return 0.5

        corrected = (observed - baseline) / (1.0 - baseline)  # in [-inf, 1]
        corrected = float(np.clip(corrected, -1.0, 1.0))
        p = 0.5 + 0.5 * corrected
        return float(np.clip(coverage * p + (1.0 - coverage) * 0.5, 0.0, 1.0))

    @staticmethod
    def structural_consistency(
        labels: np.ndarray,
        candidate: Dict[str, Any],
        X: Optional[List[bytes]] = None,
    ) -> float:
       
        values = candidate["values"]
        cand_type = candidate.get("type", "FOR")
        if len(labels) != len(values):
            raise ValueError("labels and candidate values must contain the same number of messages")
        if X is not None and len(X) != len(values):
            raise ValueError("X and candidate values must contain the same number of messages")

        n_total = len(values)
        if n_total == 0:
            return 0.0

        valid_mask = [v is not None for v in values]
        presence = sum(valid_mask) / n_total

        if presence < 0.5:
            return 0.1

        # Build cluster -> indices mapping.
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        if X is not None:
     
            cluster_scores: List[float] = []

            for cluster_id, indices in clusters.items():
                if len(indices) < 2:
                    continue

                cluster_msgs = [X[i] for i in indices]

                # 1. Length consistency: messages in the same cluster
                #    should have similar lengths (low variance).
                lengths = [len(m) for m in cluster_msgs]
                mean_len = np.mean(lengths)
                if mean_len > 0:
                    len_cv = np.std(lengths) / mean_len  # coefficient of variation
                    len_score = 1.0 / (1.0 + len_cv)
                else:
                    len_score = 1.0

                cand_offset = candidate.get("offset", 0)
                cand_width = candidate.get("width", 1)

                # Build set of offsets occupied by the candidate field.
                cand_offsets = set()
                if cand_type == "FOR":
                    cand_offsets = set(range(cand_offset, cand_offset + cand_width))
           
                min_len = min(lengths)
                if min_len > 0:
                    agreement_scores = []
                    for pos in range(min_len):
                        if pos in cand_offsets:
                            continue  # Skip keyword field positions
                        col_vals = [m[pos] for m in cluster_msgs]
                        unique_count = len(set(col_vals))
                        if unique_count == 1:
                            agreement_scores.append(1.0)
                        else:
                            agreement_scores.append(1.0 / unique_count)

                    if agreement_scores:
                        byte_agreement = float(np.mean(agreement_scores))
                    else:
                        byte_agreement = 1.0
                else:
                    byte_agreement = 1.0

                # Combine length consistency and byte agreement.
                consistency = 0.4 * len_score + 0.6 * byte_agreement
                cluster_scores.append(consistency)

            # Singleton clusters carry no consistency evidence.  Without a
            # correction, an over-segmented partition is scored only on its
            # few multi-message clusters (or on `presence` alone), which
            # rewards near-unique fields.  Shrink towards neutral (0.5) in
            # proportion to the share of messages that sit in singletons.
            n_in_multi = sum(
                len(idx) for idx in clusters.values() if len(idx) >= 2
            )
            coverage = n_in_multi / n_total

            if not cluster_scores:
                return 0.5

            avg_consistency = float(np.mean(cluster_scores))
            p_s = 0.2 * presence + 0.8 * avg_consistency
            p_s = coverage * p_s + (1.0 - coverage) * 0.5

        else:
            # ---- Fallback: candidate-only checks (NOT paper-accurate) ----
            warnings.warn(
                "structural_consistency called without X (full messages). "
                "Using candidate-only fallback — NOT paper-accurate due to "
                "circularity (clustering by candidate values then checking "
                "candidate value consistency).",
                UserWarning,
                stacklevel=2,
            )

            cluster_scores: List[float] = []

            for cluster_id, indices in clusters.items():
                cluster_vals = [
                    values[i] for i in indices
                    if values[i] is not None
                ]

                if len(cluster_vals) < 2:
                    continue

                if cand_type == "NFOR":
                    # Check TLV structural validity.
                    patterns = candidate.get("patterns", [])
                    valid_count = 0
                    for p in patterns:
                        if not isinstance(p, dict):
                            continue
                        len_val = p.get("len_val", -1)
                        value_bytes = p.get("value_bytes", b"")
                        if len_val == len(value_bytes):
                            valid_count += 1
                    consistency = valid_count / len(patterns) if patterns else 0.5
                else:
                    # FOR: use byte-level uniqueness as a weak proxy.
                    unique_vals = set(
                        bytes(v) if isinstance(v, (bytes, bytearray))
                        else v
                        for v in cluster_vals
                    )
                    uniqueness_ratio = len(unique_vals) / len(cluster_vals)
                    consistency = 1.0 - uniqueness_ratio * 0.5

                cluster_scores.append(consistency)

            if not cluster_scores:
                return float(presence * 0.5)

            avg_consistency = float(np.mean(cluster_scores))
            p_s = 0.3 * presence + 0.7 * avg_consistency

        return float(np.clip(p_s, 0.0, 1.0))

    # ==============================================================
    #  4. Dimensional Constraint
    # ==============================================================

    @staticmethod
    def dimensional_constraint(labels: np.ndarray) -> float:
   
        n = len(labels)
        if n == 0:
            return 0.0

        k = len(np.unique(labels))

        if k <= 1:
            # Single cluster — degenerate case.
            return 0.1

        # Metric 1: distinct value ratio.
        r_distinct_value = k / n

        # Metric 2: single-message cluster ratio.
        cluster_sizes: Dict[int, int] = {}
        for label in labels:
            cluster_sizes[int(label)] = cluster_sizes.get(int(label), 0) + 1

        single_message_clusters = sum(
            1 for size in cluster_sizes.values() if size == 1
        )

        r_single = single_message_clusters / k if k > 0 else 1.0

        # Graded replacement for the old 0.95 / 0.1 step function (which
        # passed any partition with k/n <= 0.5, e.g. a Modbus transaction
        # ID shared by request/response pairs, or DHCP xid bytes).  Same end
        # points (0.95 when both ratios are ~0, 0.1 when fully penalized),
        # but k/n = DIM_DISTINCT_LIMIT (0.5) is now fully penalized and the
        # penalty grows quadratically towards it.
        d = min(1.0, r_distinct_value / ClusteringConstraints.DIM_DISTINCT_LIMIT)
        penalty = max(d * d, r_single)
        return float(0.95 - 0.85 * min(1.0, penalty))