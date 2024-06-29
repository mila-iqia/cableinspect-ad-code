#!/usr/bin/env python3

"""Gather the results of an unsupervised kfold experiment."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping

import numpy as np
import pandas as pd
import yaml
from scipy.stats import betaprime

sys.path.append(os.path.join(os.path.dirname(__file__), "../notebooks"))
from utils import compute_metrics  # noqa: E402


def parse_args() -> Namespace:
    """Parser for the command line arguments.

    Returns:
        arguments (Namespace): The arguments.
    """
    parser = ArgumentParser(description="Generate and gather the results of an unsupervised kfold experiment.")

    parser.add_argument("--experiment-directory", type=str, help="Experiment results directory.")

    parser.add_argument("--data-directory", type=str, default="", help="Dataset directory, only for HQ cable dataset.")

    parser.add_argument("--percentile-threshold", type=float, default=0.95, help="Percentile for threshold.")
    arguments = parser.parse_args()
    return arguments


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    """Utility function to flatten a nested directory."""
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


def _get_run_metrics(metrics_fname: str) -> pd.DataFrame:
    """Load and post-process a metrics file log."""
    metrics = pd.read_csv(metrics_fname)
    # Only include test metrics
    test_metrics = [metric for metric in metrics.columns.tolist() if metric.startswith("test_image_")]
    metrics = metrics[test_metrics].dropna(axis=0)
    metrics.reset_index(drop=True, inplace=True)
    return metrics


def _get_run_hparam(hparam_fname: str) -> pd.DataFrame:
    """Load and post-process an hyper-parameter file log."""
    with open(hparam_fname, encoding="utf8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    hparam_dict = {k: [str(v)] for k, v in flatten_dict(cfg).items()}
    hparam = pd.DataFrame.from_dict(hparam_dict, orient="columns")
    return hparam


def _get_run_scores_stats(run_directory: str, labels: pd.DataFrame | None = None) -> dict:
    """Load and post-process normalization stats and image test predictions file logs."""
    # Validation scores stats
    stats_fname = os.path.join(run_directory, "normalization_stats.csv")
    stats = pd.read_csv(stats_fname).add_prefix("validation_")
    # Test image predictions (scores)
    test_scores_fname = os.path.join(run_directory, "test_image_predictions.csv")
    test_scores = pd.read_csv(test_scores_fname)
    # Add image test min/max scores stats
    stats["test_image_min"] = test_scores["anomaly_score"].min()
    stats["test_image_max"] = test_scores["anomaly_score"].max()
    # Isolate nominal pred from anomalous pred
    test_scores_nominal = test_scores[test_scores["target"] == 0]["anomaly_score"]
    test_scores_anomalous = test_scores[test_scores["target"] == 1]["anomaly_score"]
    # Add image test min/max scores stats stratified per nominal and anomalous images
    stats["test_nominal_image_min"] = test_scores_nominal.min()
    stats["test_nominal_image_max"] = test_scores_nominal.max()
    stats["test_anomalous_image_min"] = test_scores_anomalous.min()
    stats["test_anomalous_image_max"] = test_scores_anomalous.max()
    # Percentage of nominal test images with anomaly scores less than max validation score
    stats["pct_test_nominal_in_val"] = (
        (test_scores_nominal < stats["validation_image_max"][0]).sum() / len(test_scores_nominal) * 100
    )
    # Percentage of anomalous test images with anomaly scores less than max validation score
    stats["pct_test_anomalous_in_test_nominal"] = (
        (test_scores_anomalous < stats["test_nominal_image_max"][0]).sum() / len(test_scores_anomalous) * 100
    )
    if isinstance(labels, pd.DataFrame) and "anomaly_type_grade" in labels:
        # Add labels to anomalous test predictions
        test_predictions = test_scores[test_scores["target"] == 1][["image_path", "anomaly_score"]]
        tmp_labels = labels[["image_path", "anomaly_type_grade"]]
        test_predictions = test_predictions.merge(tmp_labels, on="image_path", how="left")
        anomaly_type_grades = test_predictions["anomaly_type_grade"].unique().tolist()
        for anomaly_type_grade in anomaly_type_grades:
            key = f"pct_test_{anomaly_type_grade.replace(' ', '_')}_in_test_nominal"
            tmp_pred = test_predictions[test_predictions["anomaly_type_grade"] == anomaly_type_grade]["anomaly_score"]
            value = (tmp_pred < stats["test_nominal_image_max"][0]).sum() / len(tmp_pred) * 100
            stats[key] = value
    return stats


def generate_and_save_thresholded_prediction_stats(
    percentile: float, run_directory: str, predictions_fname: str, threshold_types: list[str], log_directory: str
) -> None:
    """Generate prediction stats and save them using different thresholding techniques.

    Args:
        percentile (float): Percentile for the threshold.
        run_directory (str): Path to run directory.
        predictions_fname (str): Path to predictions file.
        threshold_types (list[str]): List of names of threshold types.
        log_directory (str): Path to metrics files.
    """
    # Load and pre-process predictions
    val_predictions_fname = predictions_fname.replace("test_image_predictions", "validation_image_predictions")
    val_predictions = pd.read_csv(val_predictions_fname)
    thresholds = {}

    if "max" in threshold_types:
        thresholds["max"] = max(val_predictions["anomaly_score"].to_list())

    # Fit a beta prime continuous random variable and get threshold.
    if "beta_prime" in threshold_types:
        a, b, loc, scale = betaprime.fit(val_predictions["anomaly_score"].to_list())
        beta_prime_threshold = betaprime.ppf(percentile, a, b, loc, scale)
        thresholds["beta_prime"] = beta_prime_threshold

    # Get threshold using a percentile of the empirical distribution.
    if "empirical" in threshold_types:
        emprirical_threshold = val_predictions.anomaly_score.quantile(percentile)
        thresholds["empirical"] = emprirical_threshold

    # Whisker threshold 1.5 * IQR.
    if "whisker" in threshold_types:
        _, bp = val_predictions.boxplot(column="anomaly_score", grid=False, return_type="both")
        whiskers = [whiskers.get_ydata().tolist() for whiskers in bp["whiskers"]]
        whisker_threshold = whiskers[-1][-1]
        thresholds["whisker"] = whisker_threshold

    log_directory = log_directory.split("*/")[-1]
    # Load run image threshold
    normalization_stats_fname = os.path.join(run_directory, "normalization_stats.csv")
    normalization_stats = pd.read_csv(normalization_stats_fname)
    predictions = pd.read_csv(predictions_fname)
    metrics_lst = ["F1Score", "Precision", "Recall", "AUPR", "FPR"]
    for threshold_name, value in thresholds.items():
        normalization_stats[threshold_name] = value
        save_dir = os.path.join(run_directory, log_directory)
        save_new_predictions(predictions, metrics_lst, threshold_name, value, save_dir)
    normalization_stats.to_csv(os.path.join(run_directory, "normalization_stats.csv"), index=False)


def save_new_predictions(
    predictions: pd.DataFrame, metrics_lst: list[str], threshold_name: str, threshold_value: float, save_directory: str
):
    """Threshold predictions and save new scores.

    Args:
        predictions (pd.DataFrame): Anomaly score predictions.
        metrics_lst (list[str]): List of metrics to save..
        threshold_name (str): Name of the thresholding technique used.
        threshold_value (float): Value of the threshold.
        save_directory (str): Path to save metrics files.
    """
    metrics_dict = {}
    metrics_lst_appended = [f"test_image_{threshold_name}_{metric}" for metric in metrics_lst]
    scores = compute_metrics(predictions["target"], predictions["anomaly_score"], threshold_value, metrics_lst)
    for metric_name, score in zip(metrics_lst_appended, scores):
        metrics_dict[metric_name] = [score.round(6)]
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_fname = os.path.join(save_directory, f"{threshold_name}_metrics.csv")
    metrics_df.to_csv(metrics_fname, index=False)


def generate_and_save_anomaly_ids_level_predictions(
    labels: pd.DataFrame, run_directory: str, predictions_fname: str
) -> None:
    """Generate and save anomaly IDs level predictions.

    Args:
        labels (pd.DataFrame): Data labels.
        run_directory (str): Path to run directory.
        predictions_fname (str): Path to predictions file.
    """
    # Load and pre-process predictions
    predictions = pd.read_csv(predictions_fname)
    predictions["image_path"] = predictions["image_path"].str.split("tight_crop/").str[-1]
    # Overwrite predictions
    predictions.to_csv(predictions_fname, index=False)

    # Add labels to predictions
    predictions = predictions.merge(labels, on="image_path", how="left")
    predictions["identification"] = predictions["identification"].fillna("good")
    # Remove duplicates if any
    predictions = predictions[["image_path", "identification", "anomaly_score"]].drop_duplicates()

    # Generate predictions at the anomaly IDs level
    col_names = ["identification", "anomaly_score"]
    # Predictions for nominal images remain the same
    nominal_cond = predictions["identification"] == "good"
    nominal_predictions = predictions[nominal_cond][col_names]
    # Predictions for individual anomalies
    # An anomaly is considered well predicted if found in at least one frame
    anomalies_cond = predictions["identification"] != "good"
    anomalies_predictions = predictions[anomalies_cond][col_names]
    anomalies_predictions_grouped = anomalies_predictions.groupby(["identification"])
    anomalies_predictions = anomalies_predictions_grouped[["anomaly_score"]].max().reset_index()
    # Merge nominal and anomalies predictions
    predictions_ids_level = pd.concat([anomalies_predictions, nominal_predictions], axis=0)

    # Load run image threshold
    normalization_stats_fname = os.path.join(run_directory, "normalization_stats.csv")
    normalization_stats = pd.read_csv(normalization_stats_fname)
    image_threshold = round(normalization_stats["image_threshold"].values[0], 6)
    # Add binary predictions to predictions
    predictions_ids_level["predictions"] = np.where(predictions_ids_level["anomaly_score"] >= image_threshold, 1, 0)

    # Build target
    predictions_ids_level["target"] = (predictions_ids_level["identification"] != "good").astype(int)
    # Save predictions to run directory
    predictions_ids_level.to_csv(os.path.join(run_directory, "test_identification_predictions.csv"), index=False)


def remove_broken_runs(metrics_fnames: list, broken_runs: list) -> list:
    """Remove broken runs from metric fname list.

    Args:
        metrics_fnames (list): Metrics file name.
        broken_runs (list): List of broken runs.

    Returns:
        clean_metrics_fnames (list): Cleaned list of metric file names.
    """
    clean_metrics_fnames = metrics_fnames
    for broken_run in broken_runs:
        for metrics_fname in metrics_fnames:
            if broken_run in metrics_fname:
                clean_metrics_fnames.remove(metrics_fname)
    return clean_metrics_fnames


if __name__ == "__main__":
    args = parse_args()
    data_directory = args.data_directory
    experiment_directory = args.experiment_directory
    percentile_threshold = args.percentile_threshold

    # Gather metrics and hyper-parameters logs for the experiment
    print(f"Gather results for experiment: {experiment_directory} ...", end=" ")
    start_time = time.time()

    # Metrics
    log_directory = "/*/logs/lightning_logs/version_0/"
    threshold_types = ["max", "beta_prime", "empirical", "whisker"]
    broken_runs = []

    if data_directory:
        print(f"Using the dataset stored in: {data_directory}")
        # Load data labels
        labels = pd.read_csv(os.path.join(data_directory, "labels.csv"))

        # List all runs directories
        runs_directories = glob.glob(f"{experiment_directory}/*/")
        # Generate runs predictions anomaly IDs level
        for run_directory in runs_directories:
            # Generate anomaly IDs level predictions
            predictions_fname = os.path.join(run_directory, "test_image_predictions.csv")
            if os.path.isfile(predictions_fname):
                generate_and_save_thresholded_prediction_stats(
                    percentile_threshold, run_directory, predictions_fname, threshold_types, log_directory
                )
            else:
                print(f"Broken run: {run_directory}")
                broken_runs.append(run_directory.split("/")[-2])
    else:
        # No available labels
        labels = None
        print(f"--data-directory {data_directory} arg will be ignored.")

    # Gather the results of the experiment
    # Metrics
    metrics_fnames = glob.glob(experiment_directory + log_directory + "metrics.csv")
    metrics_fnames = remove_broken_runs(metrics_fnames, broken_runs)
    metrics_fname = metrics_fnames[0]
    metrics = _get_run_metrics(metrics_fname)

    # Hyper-parameters
    hparam_fname = metrics_fnames[0].replace("metrics.csv", "hparams.yaml")
    hparam = _get_run_hparam(hparam_fname)

    # Scores stats
    run_directory = metrics_fnames[0].split("/logs")[0]
    stats = _get_run_scores_stats(run_directory, labels)

    results = pd.concat([metrics, stats, hparam], axis=1)
    for threshold_type in threshold_types:
        thresholded_metrics_fname = metrics_fname.replace("metrics.csv", f"{threshold_type}_metrics.csv")
        new_metrics = _get_run_metrics(thresholded_metrics_fname)
        results = pd.concat([results, new_metrics], axis=1)

    for metrics_fname in metrics_fnames[1:]:
        # Metrics
        metrics = _get_run_metrics(metrics_fname)

        # Hyper-parameters
        hparam_fname = metrics_fname.replace("metrics.csv", "hparams.yaml")
        hparam = _get_run_hparam(hparam_fname)

        # Scores stats
        run_directory = metrics_fname.split("/logs")[0]
        stats = _get_run_scores_stats(run_directory, labels)

        temp_results = pd.concat([metrics, stats, hparam], axis=1)

        for threshold_type in threshold_types:
            thresholded_metrics_fname = metrics_fname.replace("metrics.csv", f"{threshold_type}_metrics.csv")
            new_metrics = _get_run_metrics(thresholded_metrics_fname)
            temp_results = pd.concat([temp_results, new_metrics], axis=1)

        results = pd.concat([results, temp_results], ignore_index=True)

    # Save aggregated results to experiment directory
    results.to_csv(os.path.join(experiment_directory, "aggregated_results.csv"), index=False)

    time_elapsed = time.time() - start_time
    print(f"completed in {time_elapsed}s")
