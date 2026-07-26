"""
Netplier-style Clustering Constraints for RPKClust Stage 1.

RPKClust Section 3.6: "we use the four constraints proposed in
Netplier: message similarity constraint, remote coupling constraint,
structural consistency constraint, and dimensional constraint."

The RPKClust paper does not redefine these constraints — it references
Netplier [29] directly. This implementation follows Netplier's definitions
adapted for RPKClust's region-partitioned (non-MSA) pipeline.

Netplier reference: Ye et al., "NETPLIER: Probabilistic Network Protocol
Reverse Engineering from Message Traces," NDSS 2021.

NOTE: Because the paper does not provide exact formulas, these are labeled
as "Netplier-style approximations aligned with RPKClust's described inputs."
"""

import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class ClusteringConstraints:
    """
    Four clustering constraints from Netplier, adapted for RPKClust.

    Each constraint returns a float in [0, 1] representing the degree
    of constraint satisfaction for the current clustering.
    """

    # ==============================================================
    #  1. Message Similarity Constraint
    # ==============================================================

    @staticmethod
    def _byte_similarity(a: bytes, b: bytes) -> float:
        """
        Netplier byte-level similarity: number of matching bytes at the
        same position divided by the maximum message length.
        Extra bytes in the longer message count as mismatches.

        Netplier: s = (number of same bytes) / (total bytes compared).
        """
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
        """
        Netplier Message Similarity Constraint.

        Netplier: "messages in the same cluster should have higher
        similarity than messages in different clusters."

        Computes byte-level similarity (matching_bytes / max_len) for all
        message pairs. Separates into intra-cluster and inter-cluster
        score distributions. The constraint is satisfied when intra-
        cluster scores are consistently higher than inter-cluster scores.

        Returns the probability that the constraint is observed,
        based on the overlap (false match + false non-match) between
        the two distributions.
        """
        n = len(labels)
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

        # Netplier: compute false match and false non-match errors.
        # False match: inter-cluster score >= threshold
        #   (different types incorrectly grouped together).
        # False non-match: intra-cluster score <= threshold
        #   (same type incorrectly split apart).
        #
        # Use a threshold at the midpoint between the two means.
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
        """
        Netplier Remote Coupling Constraint.

        Netplier: "client and server clusters should have corresponding
        relationships." Uses interaction metadata (source/dest IP, ports,
        timestamps, direction, session_id) to pair request-response
        messages and check that clusters correspond across client/server
        sides.

        If interaction_metadata is None or lacks session/pairing info,
        this constraint cannot be computed. A neutral score (0.5) is
        returned with a warning.
        """
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
        client_indices = [i for i in range(n) if directions[i] == "client"]
        server_indices = [i for i in range(n) if directions[i] == "server"]

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

                # Pair each client message with nearest server message.
                for ci in c_msgs:
                    best_si = None
                    best_dt = float("inf")
                    for si in s_msgs:
                        dt = timestamps[si] - timestamps[ci]
                        if dt >= 0 and dt < best_dt:
                            best_dt = dt
                            best_si = si
                    if best_si is not None:
                        pairs.append((ci, best_si))
        else:
            # No session info — pair by timestamp proximity globally.
            sorted_clients = sorted(client_indices, key=lambda i: timestamps[i])
            sorted_servers = sorted(server_indices, key=lambda i: timestamps[i])

            for ci in sorted_clients:
                best_si = None
                best_dt = float("inf")
                for si in sorted_servers:
                    dt = timestamps[si] - timestamps[ci]
                    if dt >= 0 and dt < best_dt:
                        best_dt = dt
                        best_si = si
                if best_si is not None:
                    pairs.append((ci, best_si))

        if not pairs:
            return 0.5

        # Check cluster correspondence: for each client cluster,
        # what fraction of its paired server messages land in a
        # single dominant server cluster?
        # Build cluster -> paired cluster mapping.
        client_cluster_pairs: Dict[int, List[int]] = defaultdict(list)

        for ci, si in pairs:
            client_cluster_pairs[int(labels[ci])].append(int(labels[si]))

        cluster_correspondence_scores: List[float] = []

        for c_cluster, s_clusters in client_cluster_pairs.items():
            if not s_clusters:
                continue
            # Find dominant server cluster.
            s_counts: Dict[int, int] = defaultdict(int)
            for sc in s_clusters:
                s_counts[sc] += 1
            dominant_ratio = max(s_counts.values()) / len(s_clusters)
            cluster_correspondence_scores.append(dominant_ratio)

        if not cluster_correspondence_scores:
            return 0.5

        return float(np.mean(cluster_correspondence_scores))

    # ==============================================================
    #  3. Structural Consistency Constraint
    # ==============================================================

    @staticmethod
    def structural_consistency(
        labels: np.ndarray,
        candidate: Dict[str, Any],
        X: Optional[List[bytes]] = None,
    ) -> float:
        """
        Netplier Structure Coherence Constraint.

        Netplier: "messages of the same type share similar field
        structure." After clustering by candidate field, re-aligns
        messages within each cluster and computes alignment gap ratio.
        p_s = 1 - (avg_gaps / total_length).

        RPKClust adaptation: Since RPKClust does not use MSA, we adapt
        this constraint to check full-message structural consistency
        within each cluster using the message data X.

        Requires X (full message set) to avoid circularity — clustering
        by candidate values and then checking candidate value consistency
        would be tautological. We check whether full messages within
        each cluster share similar structure (byte-level consistency
        at non-keyword offsets, length consistency, TLV validity).

        Falls back to candidate-only checks if X is not provided
        (labeled as NOT paper-accurate).
        """
        values = candidate["values"]
        cand_type = candidate.get("type", "FOR")

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
            # ---- Full-message structural consistency (preferred) ----
            # Netplier: messages in the same cluster should share
            # similar field structure. Without MSA, we measure this as
            # intra-cluster message length consistency and byte-level
            # agreement at non-keyword offsets.
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

                # 2. Byte-level structural agreement at non-keyword
                #    positions: for each offset not covered by the
                #    candidate field, check how consistent the bytes
                #    are across messages in the cluster.
                cand_offset = candidate.get("offset", 0)
                cand_width = candidate.get("width", 1)

                # Build set of offsets occupied by the candidate field.
                cand_offsets = set()
                if cand_type == "FOR":
                    cand_offsets = set(range(cand_offset, cand_offset + cand_width))
                # For NFOR, candidate offsets vary per message, so
                # we skip offset-level exclusion and just use length.

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

            if not cluster_scores:
                return float(presence * 0.5)

            avg_consistency = float(np.mean(cluster_scores))
            p_s = 0.2 * presence + 0.8 * avg_consistency

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
        """
        Netplier Dimension Constraint.

        Netplier considers two metrics:
        1. r_distinct_value = (number of distinct field values) /
           (number of messages) — compared to threshold t_value = 0.5.
           If r > t_value, too many clusters → unlikely keyword.

        2. r_single = (number of single-message clusters) /
           (number of clusters) — compared to threshold t_single = 0.5.
           If r > t_single, too many singleton clusters → unlikely keyword.

        If both metrics are below their thresholds, p_d = 0.95 (high).
        Otherwise, p_d = 0.1 (low).

        The thresholds are conservatively set at 0.5 to avoid
        discarding true keywords.
        """
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

        # Netplier thresholds (conservatively set at 0.5).
        # Paper: "If both values are less than their thresholds, the
        # probability of the dimension constraint is high, e.g., 0.95.
        # Otherwise, it is set to a low probability, e.g., 0.1."
        # Use <= to be inclusive at the boundary (conservative).
        t_value = 0.5
        t_single = 0.5
        r_single = single_message_clusters / k if k > 0 else 1.0

        if r_distinct_value <= t_value and r_single <= t_single:
            return 0.95
        else:
            return 0.1