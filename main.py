import os
import sys

# Ensure local packages and modules resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.generate_data import (
    generate_generic_for_dataset,
    generate_generic_nfor_dataset,
)
from datasets.stress_generator import BinaryProtocolStressGenerator
from temp_evaluator import RPKClustEvaluator


def main():
    print("==========================================================")
    print(" Initializing RPKClust Paper-Faithful Evaluation Suite")
    print("==========================================================\n")

    evaluator = RPKClustEvaluator(output_dir="results", fig_format="png", dpi=300)

    # 1. Generic Fixed-Offset Region (FOR) Dataset
    X_for, y_for = generate_generic_for_dataset(num_messages=1000, seed=54761161)
    evaluator.run_diagnostics(
        X_for,
        y_for,
        dataset_name="Generic FOR",
        true_boundary=8,
        true_keyword_offset=2,
    )

    # 2. Generic Non-Fixed-Offset Region (NFOR TLV) Dataset
    X_nfor, y_nfor = generate_generic_nfor_dataset(num_messages=1000, seed=841561854)
    evaluator.run_diagnostics(
        X_nfor,
        y_nfor,
        dataset_name="Generic NFOR TLV",
        true_boundary=0,
    )

    # 3. Binary Stress Test Dataset
    stress_gen = BinaryProtocolStressGenerator(num_messages=1000, noise_level=0.15, seed=4151684516)
    X_stress, y_stress = stress_gen.generate()
    evaluator.run_diagnostics(
        X_stress,
        y_stress,
        dataset_name="Binary Protocol Stress",
        true_boundary=4,
    )

    # Export Aggregate Tables and Comparative Visualizations
    print("\n" + "=" * 65)
    print(" GENERATING FINAL PAPER ARTIFACTS")
    print("=" * 65)
    evaluator.export_summary_artifacts()
    print("Evaluation Complete. All artifacts saved to results/tables/ and results/figures/\n")


if __name__ == "__main__":
    main()