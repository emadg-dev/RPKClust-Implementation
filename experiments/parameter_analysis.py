import time
import pandas as pd
import matplotlib.pyplot as plt
from rpkclust import RPKClust
from datasets.generate_data import generate_simple_for
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from rpkclust.metrics import convert_bytes_to_feature_matrix
from datasets.generate_data import generate_nfor_tlv

def analyze_sample_size_scalability(sample_sizes=[100, 250, 500, 1000, 2000]):
    """
    Evaluates how RPKClust execution time scales with the number of messages N.
    """
    records = []
    for n in sample_sizes:
        messages, labels = generate_simple_for(n=n)
        
        t0 = time.time()
        model = RPKClust()
        model.fit(messages)
        t_elapsed = time.time() - t0
        
        records.append({
            "Sample Size (N)": n,
            "Execution Time (s)": t_elapsed,
            "Boundary Found (B)": model.boundary_B,
            "Best Candidate Prob": round(model.best_candidate['prob'], 4) if model.best_candidate else 0.0
        })
        
    return pd.DataFrame(records)

def analyze_offset_shift_impact(max_pad_lengths=[0, 5, 10, 20, 50, 100]):
    """
    Demonstrates the exact problem RPKClust solves: Traditional algorithms fail 
    when the keyword offset shifts.
    """
    print("\nRunning NFOR Offset Shift Impact Analysis...")
    records = []
    
    for pad in max_pad_lengths:
        # Generate data with increasing NFOR padding variance
        # We temporarily patch generate_nfor_tlv to accept max_pad_len in this loop
        import numpy as np
        np.random.seed(42)
        m, l_true = generate_nfor_tlv(n=400) 
        
        # Manually introduce extreme padding variance for this test
        m_shifted = []
        for msg in m:
            pad_len = np.random.randint(0, pad + 1) if pad > 0 else 0
            m_shifted.append(msg[:20] + np.random.bytes(pad_len) + msg[20:])
            
        # 1. RPKClust
        rpk = RPKClust()
        l_rpk = rpk.fit_predict(m_shifted)
        ari_rpk = adjusted_rand_score(l_true, l_rpk)
        
        # 2. K-Means
        X_mat = convert_bytes_to_feature_matrix(m_shifted)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        l_km = km.fit_predict(X_mat)
        ari_km = adjusted_rand_score(l_true, l_km)
        
        records.append({
            "Max Shift (Bytes)": pad,
            "RPKClust ARI": ari_rpk,
            "K-Means ARI": ari_km
        })
        
    df_impact = pd.DataFrame(records)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(df_impact["Max Shift (Bytes)"], df_impact["RPKClust ARI"], marker='o', label='RPKClust (Ours)', color='blue', linewidth=2)
    plt.plot(df_impact["Max Shift (Bytes)"], df_impact["K-Means ARI"], marker='x', label='K-Means', color='red', linestyle='--', linewidth=2)
    plt.title("Impact of NFOR Variable Offsets on Clustering Accuracy")
    plt.xlabel("Maximum Variable Padding (Bytes)")
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/offset_shift_impact.png", dpi=300)
    plt.close()
    
    return df_impact