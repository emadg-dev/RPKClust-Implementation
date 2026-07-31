
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

from rpkclust import RPKClust
from rpkclust.metrics import (
    evaluate_clustering,
    evaluate_boundary,
    evaluate_keyword_inference,
    convert_bytes_to_feature_matrix,
    measure_memory_usage,
)


def _to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        headers = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
        header_row = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
        separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        body = ["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |" for row in rows]
        return "\n".join([header_row, separator, *body])


class RPKClustEvaluator:
    def __init__(
        self,
        output_dir: str = "results",
        fig_format: str = "png",
        dpi: int = 300,
    ):
        self.output_dir = output_dir
        self.tables_dir = os.path.join(output_dir, "tables")
        self.figures_dir = os.path.join(output_dir, "figures")
        self.fig_format = fig_format
        self.dpi = dpi

        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        # Storage for multi-dataset comparative reports
        self.summary_records: List[Dict[str, Any]] = []

        # Configure matplotlib style for publication-grade plots
        plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        })

    def _export_boundary_keyword_table(self):
        """
        Exports boundary inference comparison table.
        """

        rows = []

        for record in self.summary_records:
            true_boundary = record.get("True_Boundary", None)
            inferred_boundary = record.get("Boundary_Inferred", 0)

            if true_boundary is not None:
                error = inferred_boundary - true_boundary
                err_percent = (
                    abs(error) / true_boundary * 100
                    if true_boundary > 0 else 0
                )
            else:
                error = None
                err_percent = None

            rows.append({
                "Sample": record.get("Dataset"),
                "True Offset": true_boundary,
                "Infer Offset": inferred_boundary,
                "Err. Per.": (
                    f"{err_percent:.1f}%"
                    if err_percent is not None else "N/A"
                )
            })

        df = pd.DataFrame(rows)

        path = os.path.join(
            self.tables_dir,
            "boundary_keyword_evaluation.md"
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Boundary Identification Evaluation\n\n")
            f.write(_to_markdown(df))
            f.write("\n")

        print(
            f"[Artifact Export] Boundary table saved:\n  - {path}"
        )

        return path

    def _generate_keyword_candidate_count_figure(
        self,
        df_summary: pd.DataFrame
        ):
        """
        Generates keyword candidate count comparison.
        """

        datasets = df_summary["Dataset"].tolist()

        for_candidates = df_summary["FOR_Candidates"].values
        nfor_candidates = df_summary["NFOR_Candidates"].values

        x = np.arange(len(datasets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.bar(
            x - width/2,
            for_candidates,
            width,
            label="FOR Candidates"
        )

        ax.bar(
            x + width/2,
            nfor_candidates,
            width,
            label="NFOR Candidates"
        )

        ax.set_ylabel("Number of Candidates")
        ax.set_title(
            "RPKClust Keyword Candidate Generation"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            datasets,
            rotation=20,
            ha="right"
        )

        ax.legend()
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5
        )

        plt.tight_layout()

        out_file = os.path.join(
            self.figures_dir,
            f"keyword_candidate_count.{self.fig_format}"
        )

        plt.savefig(
            out_file,
            dpi=self.dpi
        )

        plt.close()

        print(
            f"[Artifact Export] Candidate count plot saved:\n  - {out_file}"
        )

        return out_file

    def run_diagnostics(
        self,
        X: List[bytes],
        y_true: np.ndarray,
        dataset_name: str = "Dataset",
        true_boundary: Optional[int] = None,
        true_keyword_offset: Optional[Any] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single-dataset diagnostic pass, prints terminal diagnostics,
        and saves dataset-level tables and figures.
        """
        fit_kwargs = fit_kwargs or {}
        if len(X) != len(y_true):
            raise ValueError("X and y_true must contain the same number of messages")
        if not X:
            raise ValueError("at least one message is required for diagnostics")

        print("\n" + "=" * 65)
        print(f"  RPKCLUST PAPER DIAGNOSTIC REPORT: {dataset_name}")
        print("=" * 25)

        # Memory baseline before run
        mem_before = measure_memory_usage()

        # Fit Model and measure execution time
        model = RPKClust()
        t0 = time.perf_counter()
        model.fit(X, **fit_kwargs)
        exec_time = time.perf_counter() - t0
        mem_after = measure_memory_usage()
        memory_mb = max(0.0, round(mem_after - mem_before, 2))

        # ---- 1. Boundary Identification Evaluation (Paper Section 4.3)
        print("\n[1] FOR-NFOR BOUNDARY IDENTIFICATION (Algorithm 1)")
        print(f"  Inferred FOR-NFOR Boundary (B) : {model.boundary_B} bytes")
        print(f"  Execution Time               : {exec_time:.4f} seconds")

        boundary_eval = {}
        if true_boundary is not None:
            boundary_eval = evaluate_boundary(true_boundary, model.boundary_B)
            print(f"  Ground Truth Boundary        : {true_boundary} bytes")
            print(f"  Boundary Absolute Error      : {boundary_eval['Error']} bytes")
            print(f"  Boundary Relative Error      : {boundary_eval['Error (%)']:.2f}%")

        # ---- 2. Candidate Generation Breakdown (Paper Algorithms 2 & 3)
        for_cands = [c for c in model.candidates if c.get("type") == "FOR"]
        nfor_cands = [c for c in model.candidates if c.get("type") == "NFOR"]

        print("\n[2] REGION-PARTITIONED CANDIDATE GENERATION")
        print(f"  FOR Candidates Extracted     : {len(for_cands)}")
        print(f"  NFOR Candidates Extracted    : {len(nfor_cands)}")
        print(f"  Total Candidates Evaluated  : {len(model.candidates)}")

        # ---- 3. Two-Stage Bayesian Inference Breakdown (Paper Section 3.6 & 4.4)
        print("\n[3] TWO-STAGE BAYESIAN INFERENCE RANKINGS (Top Candidates)")
        print("-" * 65)
        print(f"{'Rank':<5}{'Tag':<22}{'Type':<6}{'p_bit':<8}{'p_offset':<10}{'Stage1':<8}{'Posterior P':<10}")
        print("-" * 65)

        top_candidates = model.candidates[:5]
        for rank, cand in enumerate(top_candidates, 1):
            c_tag = str(cand.get("tag", "N/A"))[:20]
            c_type = str(cand.get("type", "FOR"))
            p_bit = cand.get("p_bit", 0.0)
            p_offset = cand.get("p_offset", 0.0)
            p_stage1 = cand.get("stage1_prob", 0.0)
            p_final = cand.get("prob", 0.0)

            print(f"{rank:<5}{c_tag:<22}{c_type:<6}{p_bit:<8.4f}{p_offset:<10.4f}{p_stage1:<8.4f}{p_final:<10.4f}")
        print("-" * 65)

        keyword_eval = {}
        if true_keyword_offset is not None:
            keyword_eval = evaluate_keyword_inference(model.candidates, true_keyword_offset)
            print(f"  Ground Truth Keyword Offset  : {true_keyword_offset}")
            print(f"  Top Inferred Keyword Offset  : {keyword_eval['Inferred Keyword Offset']}")
            print(f"  Keyword Correctly Ranked #1  : {'YES' if keyword_eval['Correct'] else 'NO'}")
            print(f"  Keyword Rank                 : {keyword_eval['Rank']}")

        # ---- 4. Clustering Performance Metrics (Paper Section 4.2, Eq. 18-20)
        y_pred = model.labels_ if model.labels_ is not None else np.zeros(len(X), dtype=int)
        feature_matrix = convert_bytes_to_feature_matrix(X)

        cluster_metrics = evaluate_clustering(
            labels_true=y_true,
            labels_pred=y_pred,
            feature_matrix=feature_matrix,
            exec_time=exec_time,
            memory_mb=memory_mb,
        )

        print("\n[4] CLUSTERING PERFORMANCE METRICS (Section 4.2)")
        print(f"  Homogeneity (Eq. 18)         : {cluster_metrics['Homogeneity']:.4f}")
        print(f"  Completeness (Eq. 19)        : {cluster_metrics['Completeness']:.4f}")
        print(f"  V-Measure (Eq. 20)           : {cluster_metrics['V-Measure']:.4f}")
        print(f"  Adjusted Rand Index (ARI)    : {cluster_metrics['ARI']:.4f}")
        print(f"  Normalized Mutual Info (NMI) : {cluster_metrics['NMI']:.4f}")
        print(f"  Clustering Accuracy (Hungarian): {cluster_metrics['Clustering Accuracy']:.4f}")
        print(f"  Clusters Identified          : {cluster_metrics['Clusters Found']} (True: {len(np.unique(y_true))})")
        print(f"  Memory Overhead              : {cluster_metrics['Memory (MB)']} MB")

        if model.best_candidate:
            print(f"\n  Selected Keyword Tag        : {model.best_candidate.get('tag')}")
            print(f"  Selected Keyword Type       : {model.best_candidate.get('type')}")
            print(f"  Selected Keyword Posterior  : {model.best_candidate.get('prob', 0.0):.4f}")
        print("=" * 25 + "\n")

        # ---- Aggregate Summary Record
        summary_record = {
            "Dataset": dataset_name,
            "Messages": len(X),
            "Boundary_Inferred": model.boundary_B,
            "Boundary_True": true_boundary if true_boundary is not None else "N/A",
            "Boundary_Error": boundary_eval.get("Error", "N/A"),
            "FOR_Candidates": len(for_cands),
            "NFOR_Candidates": len(nfor_cands),
            "Total_Candidates": len(model.candidates),
            "Keyword_Correct": keyword_eval.get("Correct", "N/A"),
            "Keyword_Rank": keyword_eval.get("Rank", "N/A"),
            "Homogeneity": cluster_metrics["Homogeneity"],
            "Completeness": cluster_metrics["Completeness"],
            "V_Measure": cluster_metrics["V-Measure"],
            "ARI": cluster_metrics["ARI"],
            "NMI": cluster_metrics["NMI"],
            "Accuracy": cluster_metrics["Clustering Accuracy"],
            "Clusters_Found": cluster_metrics["Clusters Found"],
            "Time_s": cluster_metrics["Execution Time (s)"],
            "Memory_MB": cluster_metrics["Memory (MB)"],
        }
        self.summary_records.append(summary_record)

        # Export dataset-specific plots and candidate ranking table
        clean_ds_name = re.sub(r"[^a-z0-9_-]+", "_", dataset_name.lower()).strip("_-") or "dataset"
        self._export_candidate_table(model.candidates, clean_ds_name)
        self._generate_candidate_probability_figure(model.candidates, dataset_name, clean_ds_name)

        return summary_record

    def export_summary_artifacts(self) -> Tuple[str, str]:
        """
        Exports aggregate results across all evaluated datasets to Markdown tables
        and comparative figures.
        """
        if not self.summary_records:
            print("No evaluation records available to export.")
            return "", ""

        df_summary = pd.DataFrame(self.summary_records)

        # 1. Export Aggregate Paper Table (Markdown & CSV)
        csv_path = os.path.join(self.tables_dir, "rpkclust_summary_metrics.csv")
        md_path = os.path.join(self.tables_dir, "rpkclust_summary_metrics.md")

        df_summary.to_csv(csv_path, index=False)

        # Paper-formatted Markdown Table
        paper_table_cols = [
            "Dataset", "Messages", "Boundary_Inferred", "Homogeneity", 
            "Completeness", "V_Measure",  "Accuracy", "Time_s"
        ]
        df_paper = df_summary[paper_table_cols].copy()
        df_paper.columns = [
            "Dataset", "Messages", "Boundary (B)", "Homogeneity (h)", 
            "Completeness (c)", "V-Measure (v)",  "Acc", "Time (s)"
        ]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# RPKClust Paper Evaluation Summary Table\n\n")
            f.write(_to_markdown(df_paper))
            f.write("\n")

        print(f"[Artifact Export] Aggregated summary table written to:\n  - {md_path}\n  - {csv_path}")

        # 2. Generate Comparative Figure Across Datasets
        fig_path = self._generate_comparative_metrics_figure(df_summary)

        candidate_fig_path = self._generate_keyword_candidate_count_figure(
            df_summary
        )

        boundary_table_path = self._export_boundary_keyword_table()

        return (
            md_path,
            fig_path,
            candidate_fig_path,
            boundary_table_path
        )

    def _export_candidate_table(self, candidates: List[Dict[str, Any]], dataset_key: str):
        """Exports top candidate ranking breakdown to Markdown."""
        rows = []
        for rank, c in enumerate(candidates[:10], 1):
            rows.append({
                "Rank": rank,
                "Tag": c.get("tag", "N/A"),
                "Type": c.get("type", "N/A"),
                "Offset": c.get("offset", "N/A"),
                "p_bit": round(c.get("p_bit", 0.0), 4),
                "p_offset": round(c.get("p_offset", 0.0), 4),
                "Stage1_Prob": round(c.get("stage1_prob", 0.0), 4),
                "Posterior_P": round(c.get("prob", 0.0), 4),
            })

        df_cands = pd.DataFrame(rows)
        md_path = os.path.join(self.tables_dir, f"candidates_{dataset_key}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Top Candidate Keyword Rankings: {dataset_key}\n\n")
            f.write(_to_markdown(df_cands))
            f.write("\n")

    def _generate_candidate_probability_figure(
        self,
        candidates: List[Dict[str, Any]],
        dataset_name: str,
        dataset_key: str,
    ):
        """Generates a publication-grade bar chart comparing Stage 1 vs Stage 2 probabilities."""
        if not candidates:
            return

        top_cands = candidates[:8]
        tags = [c.get("tag", f"Cand {i}")[:15] for i, c in enumerate(top_cands)]
        p_stage1 = [c.get("stage1_prob", 0.0) for c in top_cands]
        p_posterior = [c.get("prob", 0.0) for c in top_cands]

        x = np.arange(len(tags))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4.5))
        rects1 = ax.bar(x - width/2, p_stage1, width, label="Stage 1 (p_f)", color="#4C72B0")
        rects2 = ax.bar(x + width/2, p_posterior, width, label="Stage 2 Posterior P(K=1|D)", color="#55A868")

        ax.set_ylabel("Probability Score")
        ax.set_title(f"RPKClust Inference Stage Comparison — {dataset_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_file = os.path.join(self.figures_dir, f"stage_inference_{dataset_key}.{self.fig_format}")
        plt.savefig(out_file, dpi=self.dpi)
        plt.close()

    def _generate_comparative_metrics_figure(self, df_summary: pd.DataFrame) -> str:
        """Generates a multi-metric comparative bar chart across all evaluated datasets."""
        datasets = df_summary["Dataset"].tolist()
        metrics = ["Homogeneity", "Completeness", "V_Measure"]

        x = np.arange(len(datasets))
        width = 0.18

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for i, metric in enumerate(metrics):
            scores = df_summary[metric].values
            ax.bar(x + (i - 1.5) * width, scores, width, label=metric, color=colors[i])

        ax.set_ylabel("Score [0.0 - 1.0]")
        ax.set_title("RPKClust Clustering Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_file = os.path.join(self.figures_dir, f"rpkclust_benchmark_comparison.{self.fig_format}")
        plt.savefig(out_file, dpi=self.dpi)
        plt.close()

        print(f"[Artifact Export] Comparative benchmark plot saved to:\n  - {out_file}")
        return out_file