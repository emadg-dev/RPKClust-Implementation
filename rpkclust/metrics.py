import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
    v_measure_score
)

def convert_bytes_to_feature_matrix(messages, max_len=None):
    """
    Converts a list of byte strings/arrays into a fixed-width numerical feature 
    matrix suitable for traditional tabular ML algorithms (K-Means, DBSCAN, GMM).
    Pads shorter messages with 0s and crops longer messages to max_len.
    """
    if max_len is None:
        max_len = max(len(m) for m in messages)
        
    matrix = np.zeros((len(messages), max_len), dtype=np.float64)
    for i, m in enumerate(messages):
        length = min(len(m), max_len)
        matrix[i, :length] = np.frombuffer(m[:length], dtype=np.uint8)
    return matrix

def evaluate_clustering(labels_true, labels_pred, feature_matrix=None, exec_time=0.0):
    """
    Evaluates clustering performance using both external ground-truth metrics 
    and internal structural metrics.
    """
    # Mask out unassigned/noise points (-1) if necessary for internal metrics
    valid_mask = labels_pred != -1
    num_clusters = len(set(labels_pred[valid_mask])) if np.sum(valid_mask) > 0 else 0
    
    metrics = {
        "Execution Time (s)": round(exec_time, 4),
        "ARI": round(adjusted_rand_score(labels_true, labels_pred), 4),
        "NMI": round(normalized_mutual_info_score(labels_true, labels_pred), 4),
        "V-Measure": round(v_measure_score(labels_true, labels_pred), 4),
        "Clusters Found": num_clusters
    }
    
    # Internal metrics require at least 2 distinct clusters and feature matrix
    if feature_matrix is not None and num_clusters >= 2 and np.sum(valid_mask) > num_clusters:
        try:
            metrics["Silhouette"] = round(silhouette_score(feature_matrix[valid_mask], labels_pred[valid_mask]), 4)
            metrics["Davies-Bouldin"] = round(davies_bouldin_score(feature_matrix[valid_mask], labels_pred[valid_mask]), 4)
        except Exception:
            metrics["Silhouette"] = np.nan
            metrics["Davies-Bouldin"] = np.nan
    else:
        metrics["Silhouette"] = np.nan
        metrics["Davies-Bouldin"] = np.nan
        
    return metrics