import os

import matplotlib.pyplot as plt
import numpy as np

# from datasets import generate_generic_for_dataset, generate_generic_nfor_dataset
from experiments.compare_baselines import run_baseline_comparison
from rpkclust.metrics import convert_bytes_to_feature_matrix

try:  # Support both `python -m experiments.run_rpkclust` and direct execution.
    from .parameter_analysis import (
        analyze_offset_shift_impact,
        analyze_sample_size_scalability,
    )
except ImportError:
    from parameter_analysis import (
        analyze_offset_shift_impact,
        analyze_sample_size_scalability,
    )

def run_all_experiments():
    """Executes the complete experimental suite and exports visual artifacts."""
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("==================================================")
    print("       RUNNING RPKCLUST BENCHMARK SUITE          ")
    print("==================================================")

    # # 1. Benchmark on Dataset 1: Simple FOR
    # print("\n[1/3] Benchmarking Dataset 1: Simple FOR (OpCode)...")
    # m1, l1 = generate_generic_for_dataset()
    # res_df1 = run_baseline_comparison(m1, l1)
    # res_df1.to_csv("results/tables/dataset1_for_results.csv", index=False)
    # print(res_df1.to_string(index=False))

    # generate_benchmark_barchart(res_df1, "Performance Comparison: Simple FOR", "benchmark_for_barchart.png")

    # # 2. Benchmark on Dataset 2: NFOR TLV
    # print("\n[2/3] Benchmarking Dataset 2: NFOR TLV (DHCP-style)...")
    # m2, l2 = generate_generic_nfor_dataset()
    # res_df2 = run_baseline_comparison(m2, l2)
    # res_df2.to_csv("results/tables/dataset2_nfor_results.csv", index=False)
    # print(res_df2.to_string(index=False))

    # generate_benchmark_barchart(res_df2, "Performance Comparison: NFOR TLV", "benchmark_nfor_barchart.png")



    # # Scalability Plot
    # df_scale = analyze_sample_size_scalability()
    # df_scale.to_csv("results/tables/scalability_results.csv", index=False)

    # plt.figure(figsize=(7, 4))
    # plt.plot(df_scale["Sample Size (N)"], df_scale["Execution Time (s)"], marker='o', color='crimson', linewidth=2)
    # plt.title("RPKClust Execution Time vs. Message Count (N)")
    # plt.xlabel("Number of Messages (N)")
    # plt.ylabel("Execution Time (seconds)")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.savefig("results/figures/rpkclust_scalability.png", dpi=300)
    # plt.close()

    # # Run Variable Offset Stress Test for Question 1 & 5
    # print("\n[+] Running Offset Shift Stress Test...")
    # df_impact = analyze_offset_shift_impact()
    # df_impact.to_csv("results/tables/offset_shift_impact.csv", index=False)

    # print("\nAll experiments finished successfully! Results saved to 'results/'.")

def generate_benchmark_barchart(df_res, title, filename):
    """Generates a grouped bar chart for slide presentations."""
    models = df_res["Model"].tolist()
    ari = df_res["ARI"].tolist()
    nmi = df_res["NMI"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, ari, width, label='ARI (Accuracy)', color='#2ca02c')
    ax.bar(x + width/2, nmi, width, label='NMI (Information Info)', color='#1f77b4')

    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_all_experiments()
