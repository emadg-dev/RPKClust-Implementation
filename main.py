import os
import sys

# Ensure local packages and modules resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import PcapDatasetLoader
from datasets.generate_data import *
from temp_evaluator import RPKClustEvaluator

def main():
    print("=" * 65)
    print(" Initializing RPKClust Evaluation")
    print("=" * 65)

    evaluator = RPKClustEvaluator(output_dir="results", fig_format="png", dpi=300)
    # generator = GenericDatasetGenerator(random_seed=3457456834)


    # X, y, metadata = generator.generate_realistic_dataset(
    # num_sessions=10,
    # min_messages_per_session=5,
    # max_messages_per_session=15,
    # protocol_style="request_response",
    # random_seed=42
    # )

    # evaluator.run_diagnostics(
    #     X,
    #     y,
    #     dataset_name="Generic Data",
    #     fit_kwargs={"interaction_metadata": metadata} if metadata else None,
    # )
    # evaluator.export_summary_artifacts()

    # print("=" * 65)

    pcap_loader = PcapDatasetLoader()
    pcap_file = pcap_loader.download_pcap(filename="dhcp.pcap")
    X_pcap, y_pcap, pcap_metadata = pcap_loader.extract_payloads_with_metadata(pcap_file)
    # X_pcap, y_pcap, pcap_metadata = pcap_loader.load_protocol_dataset("datasets/ICS-pcap/DNP3")

    evaluator.run_diagnostics(
        X_pcap,
        y_pcap,
        dataset_name="DHCP - ",
        fit_kwargs={"interaction_metadata": pcap_metadata} if pcap_metadata else None,
    )
    evaluator.export_summary_artifacts()


if __name__ == "__main__":
    main()