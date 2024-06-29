# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


def read_result_csv(result_path: Path) -> pd.DataFrame:
    """Reads the csv of results.

    Args:
        result_path (Path): Path of the csv.
    Returns:
        results (pd.DataFrame): A pandas dataframe of the csv.
    """
    results = pd.read_csv(result_path)
    return results


def extract_metrics(
    result_path: Path,
    groupby_feats: list[str] = ["object_category"],
    metric_names: list[str] = ["accuracy", "precision", "recall", "f1_score", "fpr"],
) -> dict:
    """Extract confusion matrix and metrics from csv result.

    Args:
        result_path (Path): Path of the csv.
        groupby_feats (list[str], optional): Feature(s) to group outputs by (default: "object_category").
        metric_names (list[str], optional): Name of a specific metric(s) to compute (default: all metrics).
    Returns:
        results_summary (dict): Dict. of confusion matrices and metrics.
    """
    # Read the CSV file
    results = read_result_csv(result_path)

    # Group by selected feature(s)
    grouped_results = results.groupby(groupby_feats)

    results_summary = {}
    for category, group in grouped_results:
        
        category_str = ", ".join(category) if isinstance(category, tuple) else category
        y_true = group["label_targets"]
        y_pred = group["label_preds"]

        # Compute metrics from confusion matrix
        try:
            metrics_dict = compute_metrics(y_true, y_pred, metric_names)
        except:
            print(f"Error computing metrics for category: {category_str}")
        results_summary[category_str] = metrics_dict

    return results_summary


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric_names: list[str] = ["accuracy", "precision", "recall", "f1_score", "fpr"],
) -> dict:
    """Compute confusion matrix and other metrics.

    Args:
        y_true (pd.Series): Labels (targets).
        y_pred (pd.Series): Predictions.
        metric_names (list[str], optional): Name of a specific metric(s) to compute (default: all metrics).
    Returns:
        (dict): Dict. of confusion matrices and metrics.
    """
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    metrics_dict = {}

    # Compute various binary classification metrics
    if "accuracy" in metric_names:
        metrics_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    if "precision" in metric_names:
        metrics_dict["precision"] = metrics.precision_score(
            y_true, y_pred, average="binary"
        )
    if "recall" in metric_names:
        metrics_dict["recall"] = metrics.recall_score(y_true, y_pred, average="binary")
    if "f1_score" in metric_names:
        metrics_dict["f1_score"] = metrics.f1_score(y_true, y_pred, average="binary")
    if "fpr" in metric_names:
        metrics_dict["fpr"] = cm[0][1] / (cm[0][1] + cm[0][0])

    return {"confusion_matrix": cm, "metrics": metrics_dict}


def compute_average_metrics(metrics_dict: dict) -> dict[str, dict[str, float]]:
    """Compute mean and std metrics.

    Args:
        metrics_dict: dict of metrics for all classes.
    Returns:
        avg_metrics: dict. of average metrics.
    """
    # Initialize dictionaries to store the metrics
    avg_metrics: dict[str, dict[str, float]] = {"mean": {}, "std": {}}

    # Initialize the metric names
    metric_names = ["accuracy", "precision", "recall", "f1_score", "fpr"]

    # Compute mean and standard deviation for each metric
    for metric in metric_names:
        metric_values = [
            class_data["metrics"][metric] for class_data in metrics_dict.values()
        ]

        # Calculate mean and standard deviation
        avg_metrics["mean"][metric] = np.mean(metric_values)
        avg_metrics["std"][metric] = np.std(metric_values)

    return avg_metrics


def generate_markdown_table(results_summary: dict, avg_metrics: dict) -> str:
    """Generate markdown summary of results.

    Args:
        results_summary (dict): Dict. with confusion matrices and metrics.
        average_metrics (dict): Dict with mean and std of metrics.
    Returns:
        (str): Markdown table with the results.
    """
    # Header for the markdown table
    markdown_table = "Category | Accuracy | Precision | Recall | F1 Score | FPR\n"
    markdown_table += "--- | --- | --- | --- | --- | ---\n"  # Column format

    # Iterating through each category and its metrics
    for category, data in results_summary.items():
        metrics = data["metrics"]
        markdown_table += f'{category} | {metrics["accuracy"]:.3f} | {metrics["precision"]:.3f} | {metrics["recall"]:.3f} | {metrics["f1_score"]:.3f} | {metrics["fpr"]:.3f}\n'

    # Adding the average metrics row
    markdown_table += "Mean +/- std | "
    for metric in ["accuracy", "precision", "recall", "f1_score", "fpr"]:
        mean = avg_metrics["mean"][metric]
        std = avg_metrics["std"][metric]
        markdown_table += f"{mean:.3f} +/- {std:.3f} | "

    # Remove the extra trailing pipe '|'
    markdown_table = markdown_table.rstrip(" | ")

    return markdown_table


def filter_img_duplicates(input_file_path: Path):
    """Generates a new csv results file after removing duplicate sample IDs.

    Args:
        input_file_path (Path): Input csv results file path.
    """
    # Generate output csv file path.
    base_name = input_file_path.stem
    new_file_name = f"{base_name}_filter{input_file_path.suffix}"
    output_file_path = input_file_path.parent / new_file_name

    df = pd.read_csv(input_file_path)
    # Drop duplicates based on the first column (sample ID)
    df = df.drop_duplicates(subset=df.columns[0], keep="first")
    df.to_csv(output_file_path, index=False)
