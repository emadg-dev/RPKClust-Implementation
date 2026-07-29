"""
RPKClust Evaluation Metrics (Section 4).

Paper Section 4.2 uses three primary metrics:
    - Homogeneity (Eq. 18): 1 - H(T|C) / H(T)
    - Completeness (Eq. 19): 1 - H(C|T) / H(C)
    - V-Measure (Eq. 20): 2 * h * c / (h + c)

Paper Section 4.5 also reports:
    - Execution time (seconds)
    - Memory overhead (MB)

Paper Section 4.3 reports:
    - FOR-NFOR boundary inference error (offset difference)

Supplementary metrics (NOT in paper, useful for analysis):
    - ARI, NMI, Silhouette, Davies-Bouldin, Clustering Accuracy
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
)
from scipy.optimize import linear_sum_assignment


def convert_bytes_to_feature_matrix(
    messages: List[bytes],
    max_len: Optional[int] = None,
) -> np.ndarray:
    """
    Converts a list of byte strings into a fixed-width numerical feature
    matrix suitable for internal clustering metrics (Silhouette, DB).
    Pads shorter messages with 0s and crops longer messages.
    """
    if max_len is None:
        max_len = max(len(m) for m in messages) if messages else 0

    if max_len == 0:
        return np.zeros((len(messages), 1), dtype=np.float64)

    matrix = np.zeros((len(messages), max_len), dtype=np.float64)
    for i, m in enumerate(messages):
        length = min(len(m), max_len)
        matrix[i, :length] = np.frombuffer(m[:length], dtype=np.uint8)
    return matrix


def clustering_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Clustering accuracy with optimal label matching using the Hungarian
    algorithm. Cluster IDs are arbitrary, so we find the optimal assignment
    between predicted and true labels that maximizes accuracy.

    Returns a float in [0, 1].
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise ValueError("labels_true and labels_pred must be one-dimensional")
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length")
    if len(labels_true) == 0:
        return 0.0

    # Build contingency matrix.
    true_classes = np.unique(labels_true)
    pred_classes = np.unique(labels_pred)

    # Map labels to indices.
    true_map = {label: i for i, label in enumerate(true_classes)}
    pred_map = {label: i for i, label in enumerate(pred_classes)}

    n_true = len(true_classes)
    n_pred = len(pred_classes)

    contingency = np.zeros((n_true, n_pred), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        contingency[true_map[t], pred_map[p]] += 1

    # Hungarian algorithm to find optimal assignment.
    # linear_sum_assignment minimizes cost, so negate.
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Accuracy = correctly assigned / total.
    correct = contingency[row_ind, col_ind].sum()
    accuracy = correct / len(labels_true)

    return float(accuracy)


def measure_memory_usage() -> float:
    """
    Returns current memory usage in MB.
    Uses psutil if available, otherwise returns 0.0.
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        return 0.0


def evaluate_boundary(
    true_boundary: int,
    inferred_boundary: int,
) -> Dict[str, Any]:
    """
    Paper Section 4.3: FOR-NFOR boundary inference evaluation.

    Returns dict with:
        - true_offset: ground truth boundary
        - inferred_offset: inferred boundary
        - error: difference (inferred - true)
        - error_percentage: |error| / true_offset * 100
    """
    error = inferred_boundary - true_boundary
    error_pct = abs(error) / true_boundary * 100 if true_boundary > 0 else 0.0

    return {
        "True Offset": true_boundary,
        "Inferred Offset": inferred_boundary,
        "Error": error,
        "Error (%)": round(error_pct, 2),
    }


def evaluate_clustering(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    feature_matrix: Optional[np.ndarray] = None,
    exec_time: float = 0.0,
    memory_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates clustering performance using paper metrics (Section 4.2)
    and supplementary analysis metrics.

    Paper metrics (primary):
        - Homogeneity (Eq. 18)
        - Completeness (Eq. 19)
        - V-Measure (Eq. 20)
        - Execution Time
        - Memory Overhead

    Supplementary metrics (NOT in paper):
        - ARI, NMI, Clustering Accuracy
        - Silhouette, Davies-Bouldin (require feature_matrix)

    Parameters
    ----------
    labels_true : array-like
        Ground truth message type labels.
    labels_pred : array-like
        Predicted cluster labels.
    feature_matrix : optional np.ndarray
        Numeric feature matrix for internal clustering metrics.
    exec_time : float
        Execution time in seconds.
    memory_mb : optional float
        Memory usage in MB. If None, measured automatically.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if labels_true.ndim != 1 or labels_pred.ndim != 1:
        raise ValueError("labels_true and labels_pred must be one-dimensional")
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length")
    if len(labels_true) == 0:
        raise ValueError("at least one label is required for clustering evaluation")
    if feature_matrix is not None and np.asarray(feature_matrix).shape[0] != len(labels_true):
        raise ValueError("feature_matrix must contain one row per label")

    if memory_mb is None:
        memory_mb = measure_memory_usage()

    # Mask for internal metrics (exclude noise points).
    valid_mask = labels_pred != -1
    num_valid = int(np.sum(valid_mask))
    num_clusters = (
        len(set(labels_pred[valid_mask])) if num_valid > 0 else 0
    )

    # ---- Paper Metrics (Primary) ----
    metrics: Dict[str, Any] = {
        # Paper Eq. 18: Homogeneity
        "Homogeneity": round(
            homogeneity_score(labels_true, labels_pred), 4
        ),
        # Paper Eq. 19: Completeness
        "Completeness": round(
            completeness_score(labels_true, labels_pred), 4
        ),
        # Paper Eq. 20: V-Measure
        "V-Measure": round(
            v_measure_score(labels_true, labels_pred), 4
        ),
        # Paper Section 4.5: Execution Time
        "Execution Time (s)": round(exec_time, 4),
        # Paper Section 4.5: Memory Overhead
        "Memory (MB)": memory_mb,
        # Paper Section 4.2: Clusters Found
        "Clusters Found": num_clusters,
    }

    # ---- Supplementary Metrics (NOT in paper) ----
    metrics["ARI"] = round(
        adjusted_rand_score(labels_true, labels_pred), 4
    )
    metrics["NMI"] = round(
        normalized_mutual_info_score(labels_true, labels_pred), 4
    )
    metrics["Clustering Accuracy"] = round(
        clustering_accuracy(labels_true, labels_pred), 4
    )

    # Internal metrics require feature matrix and >= 2 clusters.
    if (
        feature_matrix is not None
        and num_clusters >= 2
        and num_valid > num_clusters
    ):
        try:
            metrics["Silhouette"] = round(
                silhouette_score(
                    feature_matrix[valid_mask],
                    labels_pred[valid_mask],
                ),
                4,
            )
            metrics["Davies-Bouldin"] = round(
                davies_bouldin_score(
                    feature_matrix[valid_mask],
                    labels_pred[valid_mask],
                ),
                4,
            )
        except ValueError:
            metrics["Silhouette"] = np.nan
            metrics["Davies-Bouldin"] = np.nan
    else:
        metrics["Silhouette"] = np.nan
        metrics["Davies-Bouldin"] = np.nan

    return metrics


def evaluate_keyword_inference(
    candidates: List[Dict[str, Any]],
    true_keyword_offset: Any,
) -> Dict[str, Any]:
    """
    Paper Section 4.4: Keyword inference evaluation.

    Evaluates whether the real keyword field was correctly identified
    and ranked first.

    Parameters
    ----------
    candidates : list of candidate dicts
        Must be sorted by 'prob' descending (as output by RPKClust pipeline).
    true_keyword_offset : int or tuple
        The byte offset(s) of the real keyword field(s).
    """
    if not candidates:
        return {
            "True Keyword Offset": true_keyword_offset,
            "Inferred Keyword Offset": None,
            "Correct": False,
            "Rank": None,
            "Probability": None,
        }

    best = candidates[0]
    inferred_offset = best.get("offset")
    inferred_width = best.get("width", 1)
    inferred_end = inferred_offset + inferred_width - 1 if inferred_offset is not None else None

    # Handle various true_keyword_offset formats.
    if isinstance(true_keyword_offset, (tuple, list)):
        correct = inferred_offset in true_keyword_offset
    elif isinstance(true_keyword_offset, str) and ":" in true_keyword_offset:
        # e.g. "(16:18)" format.
        parts = true_keyword_offset.strip("()").split(":")
        t_start, t_end = int(parts[0]), int(parts[1])
        correct = (
            inferred_offset is not None
            and inferred_end is not None
            and inferred_offset >= t_start
            and inferred_end <= t_end
        )
    else:
        correct = inferred_offset == true_keyword_offset

    # Find rank of true keyword.
    # Compare spans (offset, offset+width-1) for multi-byte keywords.
    rank = None
    for i, cand in enumerate(candidates):
        cand_start = cand.get("offset")
        cand_end = cand_start + cand.get("width", 1) - 1 if cand_start is not None else None
        if isinstance(true_keyword_offset, (tuple, list)):
            # true_keyword_offset is a list/tuple of acceptable offsets.
            if cand_start in true_keyword_offset:
                rank = i + 1
                break
        elif isinstance(true_keyword_offset, str):
            # e.g. "(16:18)" format — parse start:end.
            if ":" in true_keyword_offset:
                parts = true_keyword_offset.strip("()").split(":")
                t_start, t_end = int(parts[0]), int(parts[1])
                if cand_start is not None and cand_end is not None:
                    if cand_start >= t_start and cand_end <= t_end:
                        rank = i + 1
                        break
            else:
                if cand_start == int(true_keyword_offset):
                    rank = i + 1
                    break
        else:
            if cand_start == true_keyword_offset:
                rank = i + 1
                break

    return {
        "True Keyword Offset": true_keyword_offset,
        "Inferred Keyword Offset": inferred_offset,
        "Correct": correct,
        "Rank": rank,
        "Probability": round(best.get("prob", 0.0), 4),
        "Top-3 Candidates": [
            {
                "offset": c.get("offset"),
                "prob": round(c.get("prob", 0.0), 4),
                "type": c.get("type"),
            }
            for c in candidates[:3]
        ],
    }
