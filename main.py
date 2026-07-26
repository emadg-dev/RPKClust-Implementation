# import sys
# import os

# # Ensure the local packages can be imported correctly
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from experiments.run_rpkclust import run_all_experiments

# if __name__ == "__main__":
#     print("Initializing RPKClust Project Environment...")
#     run_all_experiments()
"""
RPKClust Main Execution Entry Point (main.py)
Runs paper-faithful benchmark evaluation and stress-testing on synthetic datasets.
"""

import os
import sys
from datasets.stress_generator import BinaryProtocolStressGenerator

# Ensure local packages and modules resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.generate_data import (
    generate_generic_for_dataset,
    generate_generic_nfor_dataset,
)
from experiments.run_rpkclust import run_all_experiments
from temp_evaluator import run_rpkclust_diagnostics


def main():
    print("Initializing RPKClust Paper-Faithful Environment...\n")

    # 1. Generic Fixed-Offset Region (FOR) Dataset
    print("=== Running Generic FOR Diagnostics ===")
    X_for, y_for = generate_generic_for_dataset(num_messages=1000, seed=54761161)
    run_rpkclust_diagnostics(X_for, y_for, dataset_name="Generic Fixed-Offset (FOR)")

    # 2. Generic Non-Fixed-Offset Region (NFOR TLV) Dataset
    print("\n=== Running Generic NFOR TLV Diagnostics ===")
    X_nfor, y_nfor = generate_generic_nfor_dataset(
        num_messages=1000, seed=841561854
    )
    run_rpkclust_diagnostics(
        X_nfor, y_nfor, dataset_name="Generic Non-Fixed-Offset (NFOR TLV)"
    )

    # 3. Binary Stress Test Dataset
    print("\n=== Running Binary Protocol Stress Test ===")
    stress_gen = BinaryProtocolStressGenerator(num_messages=1000, noise_level=0.15, seed=4151684516)
    X_stress, y_stress = stress_gen.generate()

    run_rpkclust_diagnostics(
        X_stress, y_stress, dataset_name="Binary Protocol Stress Dataset"
    )

    # # 4. Full Experiment Suite
    # print("\n=== Running Full RPKClust Experiments ===")
    # run_all_experiments()

    # print("\nRPKClust pipeline completed successfully.")


if __name__ == "__main__":
    main()