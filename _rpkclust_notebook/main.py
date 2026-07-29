import os
import sys

# Ensure local packages and modules resolve correctly
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.generate_data import *
from datasets.stress_generator import BinaryProtocolStressGenerator
from datasets.dataset_loader import PcapDatasetLoader
from temp_evaluator import RPKClustEvaluator
from datasets.generate_data import GenericDatasetGenerator
from rpkclust import RPKClust

print("==========================================================")
print(" Initializing RPKClust Paper-Faithful Evaluation Suite")
print("==========================================================\n")

evaluator = RPKClustEvaluator(output_dir="results", fig_format="png", dpi=300)
generator = GenericDatasetGenerator(seed=897845900173)
# 1. Generic Fixed-Offset Region (FOR) Dataset
X_for, y_for, metadata = generator.generate_for_dataset(num_messages=2000)
evaluator.run_diagnostics(
    X_for,
    y_for,
    dataset_name="Generic FOR",
    true_boundary=metadata["true_boundary_B"],
    true_keyword_offset=metadata["true_keyword_offset"],
    fit_kwargs={"interaction_metadata": metadata["interaction_metadata"]},
)

# 2. Generic Non-Fixed-Offset Region (NFOR TLV) Dataset
X_nfor, y_nfor, nfor_metadata = generator.generate_nfor_dataset(num_messages=2000)
evaluator.run_diagnostics(
    X_nfor,
    y_nfor,
    dataset_name="Generic NFOR TLV",
    true_boundary=nfor_metadata["true_boundary_B"],
    fit_kwargs={"interaction_metadata": nfor_metadata["interaction_metadata"]},
)

# 3. Binary Stress Test Dataset
stress_gen = BinaryProtocolStressGenerator(num_messages=1000, noise_level=0.15, seed=456782546)
X_stress, y_stress, stress_metadata = stress_gen.generate_with_metadata()
evaluator.run_diagnostics(
    X_stress,
    y_stress,
    dataset_name="Binary Protocol Stress",
    true_boundary=12,
    true_keyword_offset=3,
    fit_kwargs={"interaction_metadata": stress_metadata["interaction_metadata"]},
)

# 4. Real-World PCAP Traffic Dataset (Fetched via PcapDatasetLoader)
print("\n" + "=" * 65)
print(" FETCHING & PROCESSING EXTERNAL REAL PCAP DATASET")
print("=" * 65)
pcap_loader = PcapDatasetLoader()
pcap_file = pcap_loader.download_pcap()
X_pcap, y_pcap, pcap_metadata = pcap_loader.extract_payloads_with_metadata(pcap_file)

evaluator.run_diagnostics(
    X_pcap,
    y_pcap,
    dataset_name="Real Network PCAP Dataset",
    fit_kwargs={"interaction_metadata": pcap_metadata} if pcap_metadata else None,
)

# Export Aggregate Tables and Comparative Visualizations
print("\n" + "=" * 65)
print(" GENERATING FINAL PAPER ARTIFACTS")
print("=" * 65)
evaluator.export_summary_artifacts()
print("\nExecution complete! Check results/tables/ and results/figures/ for output.")


