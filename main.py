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
Runs paper-faithful benchmark evaluation on generic binary datasets.
"""

from datasets.generate_data import (
    generate_generic_for_dataset,
    generate_generic_nfor_dataset,
)
from temp_evaluator import run_rpkclust_diagnostics


def main():
    print("Initializing RPKClust Paper-Faithful Environment...\n")

    # Generate Generic FOR Dataset
    X_for, y_for = generate_generic_for_dataset(num_messages=200, seed=42)
    run_rpkclust_diagnostics(X_for, y_for, dataset_name="Generic Fixed-Offset (FOR)")

    # Generate Generic NFOR TLV Dataset
    X_nfor, y_nfor = generate_generic_nfor_dataset(num_messages=200, seed=42)
    run_rpkclust_diagnostics(X_nfor, y_nfor, dataset_name="Generic Non-Fixed-Offset (NFOR TLV)")


if __name__ == "__main__":
    main()