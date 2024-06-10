#!/usr/bin/env python3

"""Generate labels for the CableInspect-AD dataset."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: CC-BY-4.0

from __future__ import annotations

import copy
import json
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
from pandas import DataFrame


def parse_args() -> Namespace:
    """Parser for the command line arguments.

    Returns:
        arguments (Namespace): The arguments.
    """
    parser = ArgumentParser(description="Generate labels.csv in data folder.")

    parser.add_argument("--data-folder", type=str, required=True, help="Data folder.")

    parser.add_argument(
        "--annotations-json-files", nargs="+", required=True, help="A space-separated list of annotations json files."
    )

    arguments = parser.parse_args()
    return arguments


def build_data_rows_from_data_dict(data_dict: dict) -> list[list]:
    """Build data rows from data dict.

    Args:
        data_dict (dict): Data dictionary.

    Returns:
        data_rows (list[list]): List of list containing data rows.
    """
    data_rows = []
    images = data_dict["images"]
    annotations = data_dict["annotations"]
    categories_map = {cat["id"]: cat["name"] for cat in data_dict["categories"]}
    attributes_names = ["rotation", "anomaly_id", "gradation"]
    for img in images:
        img_id = img["id"]
        img_path = img["file_name"]
        fname_info = img_path.split("/")[-1].split("_")
        cable_id = fname_info[0]
        side_id = fname_info[1][0]
        pass_id = fname_info[1][1:]
        frame_id = fname_info[-1][:-4]
        img_info = [img_path, cable_id, side_id, pass_id, frame_id]
        # Retrieve anomalies annotations corresponding to the image if any.
        annotation = [annot for annot in annotations if annot["image_id"] == img_id]
        if annotation:
            for annot in annotation:
                row = copy.copy(img_info)
                category_id = annot["category_id"]
                row += [category_id, categories_map[category_id]]
                row += [annot["area"]]
                row += annot["bbox"]
                attributes = annot["attributes"]
                for attr in attributes_names:
                    row += [attributes[attr]] if attr in attributes else [np.nan]
                data_rows.append(row)
        else:
            data_rows.append(img_info + [np.nan] * 10)

    return data_rows


def build_labels_dataframe_from_data_rows(data_rows: list[list]) -> DataFrame:
    """Build labels dataframe from data_rows.

    Args:
        data_rows (list[list]): List of list containing data rows.

    Returns:
        dataframe (DataFrame): Dataframe containing the labels.
    """
    col_names = [
        "image_path",
        "cable_id",
        "side_id",
        "pass_id",
        "frame_id",
        "anomaly_type_id",
        "anomaly_type",
        "bbox_area",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "bbox_rotation",
        "anomaly_id",
        "anomaly_grade",
    ]
    dataframe = DataFrame(np.array(data_rows), columns=col_names)
    column_types = {
        "image_path": "string",
        "cable_id": "string",
        "side_id": "string",
        "pass_id": "string",
        "frame_id": "string",
        "anomaly_type_id": "float32",
        "anomaly_type": "string",
        "bbox_area": "float32",
        "bbox_x": "float32",
        "bbox_y": "float32",
        "bbox_width": "float32",
        "bbox_height": "float32",
        "bbox_rotation": "float32",
        "anomaly_id": "string",
        "anomaly_grade": "string",
    }
    dataframe = dataframe.astype(column_types)

    # Create label indexes
    dataframe["label_index"] = ~dataframe["anomaly_id"].str.contains("nan") * 1

    # Add mask paths
    dataframe["mask_path"] = ""
    dataframe.loc[dataframe.label_index == 1, "mask_path"] = dataframe["image_path"].str.replace("images", "masks")

    return dataframe


def build_labels_dataframe_from_annotations(json_files: list[str]) -> DataFrame:
    """Build labels dataframe from annotation zip files.

    Args:
        json_files (list[str]): List of annotation .json files.

    Returns:
        dataframe (DataFrame): Dataframe containing the anomalies.
    """
    data_rows = []
    for json_file in json_files:
        with open(json_file) as f:
            data_dict = json.load(f)
        data_rows += build_data_rows_from_data_dict(data_dict)

    dataframe = build_labels_dataframe_from_data_rows(data_rows)
    return dataframe


if __name__ == "__main__":
    args = parse_args()
    data_folder = args.data_folder
    annotations_json_files = args.annotations_json_files
    output_path = os.path.join(data_folder, "labels.csv")

    print(f"Generating labels and save them to: {output_path} ...", end=" ")
    start_time = time.time()
    labels = build_labels_dataframe_from_annotations(annotations_json_files)
    labels.to_csv(output_path, index=False)
    time_elapsed = time.time() - start_time
    print(f"completed in {time_elapsed}s")
