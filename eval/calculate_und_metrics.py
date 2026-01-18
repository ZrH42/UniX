import pandas as pd
import os
import sys
import csv
from datetime import datetime
from tqdm import tqdm

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Import metric modules
from und_metrics.nlp_metrics import calculate_nlp_metrics
from und_metrics.chexbert_metrics import calculate_chexbert_metrics
from und_metrics.radgraph_metrics import calculate_radgraph_metrics


def process_csv_file(csv_file_path,
                     metrics_file,
                     summary_file,
                     chexbert_path,
                     radgraph_level="partial",
                     model_type="radgraph"):
    print(f"Reading CSV file from: {csv_file_path}")

    # Read data
    with open(csv_file_path, 'r', encoding='utf-8', errors='replace') as f:
        df = pd.read_csv(f)

    required_columns = ["Report Impression", "Ground Truth"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file missing columns: {', '.join(missing_columns)}")

    all_ground_truths = []
    all_predictions = []

    # Data preprocessing
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing data"):
        report = str(row["Report Impression"]).replace('\ufffd', '?').strip()
        truth = str(row["Ground Truth"]).replace('\ufffd', '?').strip()

        if not report or not truth:
            continue

        all_ground_truths.append(truth)
        all_predictions.append(report)

    if not all_ground_truths or not all_predictions:
        raise ValueError("No valid data for metrics calculation")

    # 1. NLP metrics (BLEU, ROUGE)
    nlp_metrics = calculate_nlp_metrics(all_predictions, all_ground_truths)
    avg_metrics = nlp_metrics.copy()

    # 2. CheXbert metrics
    clinical_metrics = calculate_chexbert_metrics(all_predictions,
                                                  all_ground_truths,
                                                  chexbert_path)
    avg_metrics.update(clinical_metrics)

    # 3. RadGraph metrics
    radgraph_metrics = calculate_radgraph_metrics(all_predictions,
                                                  all_ground_truths,
                                                  radgraph_level,
                                                  model_type)
    avg_metrics.update(radgraph_metrics)

    # ========== Save complete metrics to metrics.csv ==========
    metrics_exists = os.path.exists(metrics_file)
    with open(metrics_file, mode="a", newline="", encoding="utf-8") as f:
        fieldnames = ["Timestamp"] + list(avg_metrics.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not metrics_exists:
            writer.writeheader()
        row_data = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        row_data.update(avg_metrics)
        writer.writerow(row_data)

    # ========== Save summary metrics to metrics_summary.csv ==========
    # Define field order
    summary_fields = [
        "uncertain_as_negative_micro_f1",
        "uncertain_as_negative_f1_top5_micro",
        "uncertain_as_negative_macro_f1",
        "uncertain_as_negative_f1_top5_macro",
        "uncertain_as_positive_micro_f1",
        "uncertain_as_positive_f1_top5_micro",
        "uncertain_as_positive_macro_f1",
        "uncertain_as_positive_f1_top5_macro",
        "f1-radgraph",
        "BLEU1",
        "BLEU4",
        "ROUGE_L"
    ]
    summary_exists = os.path.exists(summary_file)
    with open(summary_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(["Timestamp"] + summary_fields)
        row_data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + \
                   [avg_metrics.get(k, "") for k in summary_fields]
        writer.writerow(row_data)

    return avg_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate understanding metrics")
    parser.add_argument("--input_file", type=str,
                        default=os.path.join(RESULTS_DIR, "understanding_results.csv"),
                        help="Path to input CSV file")
    parser.add_argument("--chexbert_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "chexbert", "chexbert.pth"))
    parser.add_argument("--radgraph_level", type=str, default="partial",
                        choices=["simple", "partial", "complete", "all"])
    parser.add_argument("--model_type", type=str, default="radgraph",
                        choices=["radgraph", "radgraph-xl", "modern-radgraph-xl", "echograph"])

    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_file = os.path.join(RESULTS_DIR, "metrics.csv")
    summary_file = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    csv_file = args.input_file

    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist")
        sys.exit(1)

    metrics = process_csv_file(csv_file,
                               metrics_file,
                               summary_file,
                               args.chexbert_path,
                               args.radgraph_level,
                               args.model_type)

    print("\n===== Evaluation Results =====")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nMetrics appended to {metrics_file}")
    print(f"Summary metrics appended to {summary_file}")


if __name__ == "__main__":
    main()