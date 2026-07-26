import os
import sys

# Ensure local packages and modules resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.generate_data import *
from datasets.stress_generator import BinaryProtocolStressGenerator
from datasets.dataset_loader import PcapDatasetLoader
from temp_evaluator import RPKClustEvaluator


def main():
    print("==========================================================")
    print(" Initializing RPKClust Paper-Faithful Evaluation Suite")
    print("==========================================================\n")

    evaluator = RPKClustEvaluator(output_dir="results", fig_format="png", dpi=300)

    # 1. Generic Fixed-Offset Region (FOR) Dataset
    X_for, y_for, metadata = GenericDatasetGenerator.generate_for_dataset(num_messages=1000, seed=54761161)
    evaluator.run_diagnostics(
        X_for,
        y_for,
        internal_metadata=metadata,
        dataset_name="Generic FOR",
        true_boundary=metadata["true_boundary_B"],
        true_keyword_offset=metadata["keyword_tag"],
    )

    # 2. Generic Non-Fixed-Offset Region (NFOR TLV) Dataset
    X_nfor, y_nfor = GenericDatasetGenerator.generate_nfor_dataset(num_messages=1000, seed=841561854)
    evaluator.run_diagnostics(
        X_nfor,
        y_nfor,
        internal_metadata=metadata,
        dataset_name="Generic NFOR TLV",
        true_boundary=metadata["true_boundary_B"],
    )

    # 3. Binary Stress Test Dataset
    stress_gen = BinaryProtocolStressGenerator(num_messages=1000, noise_level=0.15, seed=4151684516)
    X_stress, y_stress = stress_gen.generate()
    evaluator.run_diagnostics(
        X_stress,
        y_stress,
        internal_metadata=metadata,
        dataset_name="Binary Protocol Stress",
        true_boundary=metadata["true_boundary_B"],
        true_keyword_offset=metadata["keyword_tag"],
    )

    # 4. Real-World PCAP Traffic Dataset (Fetched via PcapDatasetLoader)
    print("\n" + "=" * 65)
    print(" FETCHING & PROCESSING EXTERNAL REAL PCAP DATASET")
    print("=" * 65)
    pcap_loader = PcapDatasetLoader()
    pcap_file = pcap_loader.download_pcap()
    X_pcap, y_pcap = pcap_loader.extract_payloads(pcap_file)

    evaluator.run_diagnostics(
        X_pcap,
        y_pcap,
        dataset_name="Real Network PCAP Dataset",
        true_boundary=4,
    )

    # Export Aggregate Tables and Comparative Visualizations
    print("\n" + "=" * 65)
    print(" GENERATING FINAL PAPER ARTIFACTS")
    print("=" * 65)
    evaluator.export_summary_artifacts()
    print("\nExecution complete! Check results/tables/ and results/figures/ for output.")


if __name__ == "__main__":
    main()