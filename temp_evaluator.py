"""
RPKClust Diagnostics & Evaluation Suite (evaluator.py)
Profiles boundary identification, candidate rankings, and clustering metrics.
"""

import time
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from rpkclust import RPKClust


def run_rpkclust_diagnostics(X, y_true, dataset_name="Dataset"):
    """
    Executes diagnostic profiling of RPKClust pipeline stages.
    """
    print(f"\n==================================================")
    print(f" RPKCLUST DIAGNOSTIC REPORT: {dataset_name}")
    print(f"==================================================")

    model = RPKClust()

    # 1. Boundary Identification
    t0 = time.time()
    model.fit(X)
    t_boundary = time.time() - t0

    print(f"\n[1] BOUNDARY IDENTIFICATION PROFILING")
    print(f"Computed FOR-NFOR Boundary (B): {model.boundary_B} bytes")
    print(f"Execution Time: {t_boundary:.4f} seconds")

    # 2. Candidate Generation Stats
    for_count = sum(1 for c in model.candidates if c['type'] == 'FOR')
    nfor_count = sum(1 for c in model.candidates if c['type'] == 'NFOR')

    print(f"\n[2] CANDIDATE GENERATION STATS")
    print(f"FOR Candidates Generated:  {for_count}")
    print(f"NFOR Candidates Generated: {nfor_count}")

    # 3. Top Candidate Rankings
    print(f"\n[3] TWO-STAGE INFERENCE RANKINGS (Top 3)")
    for rank, cand in enumerate(model.candidates[:3], 1):
        print(f" Rank {rank}: {cand['tag']} ({cand['type']})")
        print(f"   -> p_bit    = {cand['p_bit']:.4f}")
        print(f"   -> p_offset = {cand['p_offset']:.4f}")
        print(f"   -> Stage1   = {cand['stage1_prob']:.4f}")
        print(f"   -> Posterior P(K=1|D) = {cand['prob']:.4f}")

    # 4. Clustering Performance Metrics
    y_pred = model.labels_
    h = homogeneity_score(y_true, y_pred)
    c = completeness_score(y_true, y_pred)
    v = v_measure_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print(f"\n[4] CLUSTERING METRICS")
    print(f"Homogeneity:  {h:.4f}")
    print(f"Completeness: {c:.4f}")
    print(f"V-Measure:    {v:.4f}")
    print(f"ARI:          {ari:.4f}")
    print(f"NMI:          {nmi:.4f}")
    print(f"Clusters Found : {len(np.unique(y_pred))}")
    print(f"\nBest Candidate")
    print(model.best_candidate["tag"])
    print(model.best_candidate["type"])
    print(model.best_candidate["prob"])
    print(f"==================================================\n")