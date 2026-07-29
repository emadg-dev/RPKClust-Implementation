import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

from rpkclust import RPKClust
from rpkclust.metrics import convert_bytes_to_feature_matrix, evaluate_clustering

def run_baseline_comparison(messages, labels_true):
    """
    Runs RPKClust and standard baseline models on the exact same dataset,
    returning a performance table.
    """
    if len(messages) != len(labels_true):
        raise ValueError("messages and labels_true must have the same length")
    if len(messages) < 2:
        raise ValueError("at least two messages are required for baseline comparison")
    X_mat = convert_bytes_to_feature_matrix(messages)
    n_clusters = len(set(labels_true))
    if n_clusters < 2 or n_clusters > len(messages):
        raise ValueError("labels_true must contain between 2 and len(messages) clusters")

    results = []

    # 1. RPKClust
    t0 = time.time()
    rpk = RPKClust()
    rpk_labels = rpk.fit_predict(messages)
    t_rpk = time.time() - t0
    rpk_res = evaluate_clustering(labels_true, rpk_labels, X_mat, t_rpk)
    rpk_res["Model"] = "RPKClust (Ours)"
    results.append(rpk_res)

    # 2. K-Means
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_mat)
    t_km = time.time() - t0
    km_res = evaluate_clustering(labels_true, km_labels, X_mat, t_km)
    km_res["Model"] = "K-Means"
    results.append(km_res)

    # 3. DBSCAN
    t0 = time.time()
    db = DBSCAN(eps=150.0, min_samples=5)
    db_labels = db.fit_predict(X_mat)
    t_db = time.time() - t0
    db_res = evaluate_clustering(labels_true, db_labels, X_mat, t_db)
    db_res["Model"] = "DBSCAN"
    results.append(db_res)

    # 4. Gaussian Mixture Model (GMM)
    t0 = time.time()
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X_mat)
    t_gmm = time.time() - t0
    gmm_res = evaluate_clustering(labels_true, gmm_labels, X_mat, t_gmm)
    gmm_res["Model"] = "GMM"
    results.append(gmm_res)

    # 5. Spectral Clustering
    t0 = time.time()
    spec = SpectralClustering(
        n_clusters=n_clusters,
        n_neighbors=min(10, len(messages) - 1),
        random_state=42,
        assign_labels="kmeans",
    )
    spec_labels = spec.fit_predict(X_mat)
    t_spec = time.time() - t0
    spec_res = evaluate_clustering(labels_true, spec_labels, X_mat, t_spec)
    spec_res["Model"] = "Spectral"
    results.append(spec_res)

    df_res = pd.DataFrame(results)
    # Reorder columns for presentation
    cols = ["Model", "ARI", "NMI", "V-Measure", "Silhouette", "Davies-Bouldin", "Execution Time (s)"]
    return df_res[cols]
