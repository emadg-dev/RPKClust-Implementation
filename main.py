import os
import sys
from pathlib import Path

# Ensure local packages and modules resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import PcapDatasetLoader
from temp_evaluator import RPKClustEvaluator


def evaluate_pcap(loader, evaluator, pcap_file: Path):
    dataset_name = pcap_file.stem

    print(f"\n{'=' * 65}")
    print(f"Evaluating {dataset_name}")
    print(f"{'=' * 65}")

    X, y, metadata = loader.extract_payloads_with_metadata(str(pcap_file))

    evaluator.run_diagnostics(
        X,
        y,
        dataset_name=dataset_name,
        fit_kwargs={
            "interaction_metadata": metadata
        } if metadata else None,
    )


def main():
    print("=" * 65)
    print("Initializing RPKClust Evaluation")
    print("=" * 65)

    evaluator = RPKClustEvaluator(
        output_dir="results",
        fig_format="png",
        dpi=300,
    )

    pcap_loader = PcapDatasetLoader()

    downloads_dir = Path("datasets/downloads")

    pcap_files = sorted(downloads_dir.glob("*.pcap"))

    if not pcap_files:
        raise FileNotFoundError(
            f"No .pcap files found in '{downloads_dir}'."
        )

    print(f"Found {len(pcap_files)} datasets.\n")

    for pcap_file in pcap_files:
        evaluate_pcap(
            loader=pcap_loader,
            evaluator=evaluator,
            pcap_file=pcap_file,
        )

    evaluator.export_summary_artifacts()


if __name__ == "__main__":
    main()