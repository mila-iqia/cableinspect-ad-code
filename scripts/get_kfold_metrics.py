import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd
from src.anomaly_detector.evaluation_utils import compute_metrics, filter_img_duplicates


def extract_info_from_filename(filename: str) -> dict | None:
    """Extract cable_id and anomaly_id from the kfold labels filenames.

    Args:
        filename (str): Prompts yaml file.
    Returns:
        (dict | None): Extracted cable_id and anomaly_id information if matching structure.
    """
    match = re.search(r"cable-(C\d+)_anomaly_id-(\d+)", filename)
    if match:
        return {
            "cable_id": match.group(1),
            "anomaly_id": int(match.group(2)),
        }
    else:
        # If no filename match is found for a file
        return None


def aggregate_and_save_kfold_metrics(input_csv: Path) -> None:
    """Aggregate kfold metrics with mean and std.

    Args:
        input_csv (Path): Path to csv of VLM kfold metrics.
    """
    df = pd.read_csv(input_csv)

    # Group by cable id and aggregate both mean and std for each metric
    grouped_stats = df.groupby("cable_id").agg(
        {
            "accuracy": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1_score": ["mean", "std"],
            "fpr": ["mean", "std"],
        }
    )

    grouped_stats.columns = [
        "_".join(col).rstrip() for col in grouped_stats.columns.values
    ]
    grouped_stats_reset = grouped_stats.reset_index()

    # Save aggregated metrics into new csv file
    new_path = input_csv.with_name(input_csv.stem + "_aggr" + input_csv.suffix)
    grouped_stats_reset.to_csv(new_path, index=False)
    print(f"File of aggregated metrics saved as: {new_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Get the test sets kfold metrics for VLM inference results."
    )
    parser.add_argument(
        "--vlm-csv", type=str, required=True, help="the path to the vlm csv of results"
    )
    parser.add_argument(
        "--kfold-dir",
        type=str,
        required=True,
        help="the directory that contains the csv files of generated kfold labels",
    )
    parser.add_argument(
        "--output-csv-filename",
        type=str,
        required=False,
        default="vlm_kfold_metrics.csv",
        help="the filename for the output CSV to save metrics per test fold",
    )
    parser.add_argument(
        "--filter-duplicates",
        type=bool,
        required=False,
        default=True,
        help="whether to remove image row duplicates in the input vlm csv file",
    )

    args = parser.parse_args()

    vlm_csv = Path(args.vlm_csv)
    kfold_dir = Path(args.kfold_dir)
    output_csv_path = args.output_csv_filename
    filter_duplicates = args.filter_duplicates

    kfold_dir_str = str(kfold_dir)
    pattern = kfold_dir_str + "/**/*.csv"

    if filter_duplicates:
        filter_img_duplicates(vlm_csv)
        base_name = vlm_csv.stem
        new_file_name = f"{base_name}_filter{vlm_csv.suffix}"
        filtered_csv_path = vlm_csv.parent / new_file_name
        df_vlm = pd.read_csv(filtered_csv_path)
    else:
        df_vlm = pd.read_csv(vlm_csv)

    results = []

    for kfold_file_path in glob.glob(pattern, recursive=True):
        # Extract cable and anomaly info. from kfold labels file
        info = extract_info_from_filename(Path(kfold_file_path).name)
        if not info:
            continue

        # Process each kfold labels file
        if os.path.isfile(kfold_file_path) and kfold_file_path.endswith(".csv"):
            print(f"Processing kfold labels csv file: {Path(kfold_file_path).name}")

            df_kfold = pd.read_csv(kfold_file_path)

            # Filter rows labeled as test
            df_kfold_test = df_kfold[df_kfold["split"] == "test"]

            # Find the image_path values in vlm csv and select the required columns
            df_filtered = df_vlm[
                df_vlm["image_path"].isin(df_kfold_test["image_path"])
            ][["image_path", "object_category", "label_targets", "label_preds"]]

            # Compute the main metrics for that given test fold
            all_metrics = compute_metrics(
                df_filtered["label_targets"], df_filtered["label_preds"]
            )

            # Append metrics into results list
            results.append({**info, **all_metrics["metrics"]})

    # Convert results list to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    # Aggregate the kfolds by cable
    aggregate_and_save_kfold_metrics(Path(output_csv_path))


if __name__ == "__main__":
    main()
