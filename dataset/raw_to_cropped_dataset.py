#!/usr/bin/env python3

"""Preprocess CableInspect-AD images and generate masks."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

import os
import time
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw


BBOX_COLUMNS = ["bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_rotation"]
ANOMALY_COLUMNS = [
    "anomaly_type",
    "anomaly_type_id",
    "anomaly_grade",
    "primary_anomaly_id",
    "secondary_anomaly_id",
    "anomaly_id",
    "bbox_area",
    "mask_path",
]
CROPPED_CONFIG = {
    "height_margin": 200,
    "window_size": 5,
    "threshold": 0.6,
    "final_width": 1120,
    "final_height": 224,
}


def parse_args() -> Namespace:
    """Parser for the command line arguments.

    Returns:
        arguments (Namespace): The arguments.
    """
    parser = ArgumentParser(description="Preprocess HQ dataset.")

    parser.add_argument("--data-folder", type=str, help="Data folder.")

    arguments = parser.parse_args()
    return arguments


def read_image(path: str | Path, image_size: int | tuple[int, int] | None = None) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_image("test_image.jpg")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size:
        # This part is optional, where the user wants to quickly resize the image
        # with a one-liner code. This would particularly be useful especially when
        # prototyping new ideas.
        height, width = get_image_height_and_width(image_size)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

    return image


def get_border(image: np.array, idx_k: int, g_threshold: int, is_top: bool) -> int:
    """Returns the border between the background and the cable.

    This method finds the largest rectange that is completly inside the cable.
    Since the background is green, given a window, the value of green must be higher than
    the value of red and blue. With a threshold, we can decide if the window is mainly green.
    Since the cable is angled sometimes, we check for a window containing only the cable on right and
    left side of the image and find the lowest index for top and highest index for bottom to crop only the cable.

    Args:
        image (np.array): The image to be cropped.
        idx_k (int): The size of the window to inspect the green colour intensity.
        g_threshold (int): Threshold to decide whether the window is green.
        is_top (bool): Flag to indicate whether to find the top border or the bottom border.

    Returns:
        border_index (int): The index of the border.
    """
    index_range = range(0, image.shape[0]) if is_top else reversed(range(0, image.shape[0]))
    border_index = 0 if is_top else image.shape[0]
    for i in index_range:
        left_mean_r, left_mean_g, left_mean_b = image[i][:idx_k].mean(axis=0)
        if left_mean_g < g_threshold * (left_mean_r + left_mean_b):
            right_mean_r, right_mean_g, right_mean_b = image[i][-idx_k:].mean(axis=0)
            if right_mean_g < g_threshold * (right_mean_r + right_mean_b):
                return i
    return border_index


def crop_image(image: np.array, img_cfg: dict) -> tuple:
    """Load and crop an image if possible. Return the image and image config.

    Args:
        image (np.array): Image.
        img_cfg (dict): Image config. Modified in place.

    Returns:
        tuple: Image, image config.
    """
    final_height = img_cfg["final_height"]
    final_width = img_cfg["final_width"]

    img_cfg["original_height"] = image.shape[0]
    img_cfg["original_width"] = image.shape[1]
    img_cfg["crop_left"] = 0
    if img_cfg["original_width"] >= final_width:
        left = int((img_cfg["original_width"] - final_width) / 2)
        image = image[:, left : left + final_width]
        img_cfg["crop_left"] += left

    # Crop height margin
    height_margin = img_cfg.pop("height_margin")
    image = image[height_margin:-height_margin, :]
    img_cfg["crop_top"] = height_margin

    # Refine crop to get the tight crop
    window_size = img_cfg.pop("window_size")
    threshold = img_cfg.pop("threshold")
    top = get_border(image, window_size, threshold, True)
    bottom = get_border(image, window_size, threshold, False)
    image = image[top:bottom, :]
    img_cfg["crop_top"] += top

    height = image.shape[0]
    if height >= final_height:
        top = int((height - final_height) / 2)
        image = image[top : top + final_height, :]
        # Update crop top to correct the mask later on
        img_cfg["crop_top"] += top
    else:
        image = None
        img_cfg = {}

    return (image, img_cfg)


def rotate(point: tuple, pivot_point: tuple, angle: float) -> tuple:
    """2D clockwise rotation.

    2D clockwise rotation of a point (x, y) around a pivot point (x_piv, y_piv)
    given a specified angle in degrees.

    Args:
        point (tuple): Point (x, y) coordinates.
        pivot_point (tuple): Pivot point (x_piv, y_piv) coordinates.
        angle (float): Angle in degrees of the clockwise rotation.

    Returns:
       tuple: Rotated coordinates (x_rot, y_rot).
    """
    x, y = point  # pylint: disable=C0103
    x_piv, y_piv = pivot_point
    theta = np.radians(angle)
    xrot = np.cos(theta) * (x - x_piv) - np.sin(theta) * (y - y_piv) + x_piv
    yrot = np.sin(theta) * (x - x_piv) + np.cos(theta) * (y - y_piv) + y_piv
    xrot = round(xrot.item(), 2)
    yrot = round(yrot.item(), 2)
    return (xrot, yrot)


def compute_polygon_coordinates(bbox_coordinates: tuple) -> list:
    """Compute polygon coordinates from bounding box coordinates.

    Args:
        bbox_coordinates (tuple): Bounding box coordinates.
            Under the form: (x bottom left corner,
                             y bottom left corner,
                             width,
                             height,
                             clockwise rotation angle in degrees).

    Returns:
        polygon_coordinates (list): Ordered coordinates of the polygon.
            Each tuple in the list represents the (x, y) coordinates of, in order,
            bottom left, bottom right, upper right and upper left polygon corner.
    """
    x, y, width, height, rotation = bbox_coordinates  # pylint: disable=C0103
    x_piv = x + width / 2
    y_piv = y + height / 2
    polygon_coordinates = [
        rotate((x, y), (x_piv, y_piv), rotation),  # bottom left
        rotate((x + width, y), (x_piv, y_piv), rotation),  # bottom right
        rotate((x + width, y + height), (x_piv, y_piv), rotation),  # upper right
        rotate((x, y + height), (x_piv, y_piv), rotation),  # upper left
    ]
    return polygon_coordinates


def generate_masks(bbox_coordinates: pd.DataFrame, image_size: tuple) -> list:
    """Generate the masks from bbox coordinates. Return a list of masks.

    Args:
        bbox_coordinates (pd.DataFrame): Bounding box coordinates.
            Under the form: (x bottom left corner,
                             y bottom left corner,
                             width,
                             height,
                             clockwise rotation angle in degrees).
        image_size (tuple): Image size (width, height).

    Returns:
        masks (list): List of masks.
    """
    masks = []
    for _, coordinates in bbox_coordinates.iterrows():
        polygon = compute_polygon_coordinates(tuple(coordinates))
        image = Image.new("L", image_size, 0)
        ImageDraw.Draw(image).polygon(polygon, outline=255, fill=255)
        image = np.array(image)
        masks.append(image)
    return masks


if __name__ == "__main__":
    args = parse_args()
    data_folder = args.data_folder
    cfg = CROPPED_CONFIG

    labels_fname = os.path.join(data_folder, "labels.csv")
    df = pd.read_csv(labels_fname)

    img_paths = df["image_path"].unique()
    abnormal_img_paths = df.dropna(subset=["anomaly_id"])["image_path"].unique()
    for img_path in img_paths:
        start_time = time.time()
        print(f"Pre-process image and labels for: {img_path}...", end=" ")

        path = os.path.join(data_folder, img_path)
        img = read_image(path)
        img, img_cfg = crop_image(img, cfg.copy())

        mask_path = path.replace("images", "masks")

        if not img_cfg:
            # Remove image from dataset
            # AND skip last part of the loop if no tight crop fot this image
            img_rows = df["image_path"] == img_path
            df = df[~img_rows]
            os.remove(path)
            print(f"WARNING: {img_path} has been removed from dataset.")
            # Remove mask if exist
            if os.path.exists(mask_path):
                os.remove(mask_path)
            continue

        else:
            # Overwrite image with cropped version
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # Crop segmentation mask if exist
            if os.path.exists(mask_path):
                crop_top = img_cfg["crop_top"]
                final_height = img_cfg["final_height"]
                crop_left = img_cfg["crop_left"]
                final_width = img_cfg["final_width"]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask[crop_top : crop_top + final_height, crop_left : crop_left + final_width]
                mask = Image.fromarray(mask.astype(np.uint8))
                mask.save(mask_path)

        if img_path in abnormal_img_paths:
            # Retrieve bounding boxes coordinates and generate anomaly masks
            mask_bbox_coordinates = df.loc[df["image_path"] == img_path, BBOX_COLUMNS]
            original_width = img_cfg["original_width"]
            original_height = img_cfg["original_height"]
            masks = generate_masks(mask_bbox_coordinates, (original_width, original_height))

            # Crop masks
            crop_top = img_cfg["crop_top"]
            final_height = img_cfg["final_height"]
            crop_left = img_cfg["crop_left"]
            final_width = img_cfg["final_width"]
            img_pixels = final_height * final_width
            bbox_areas = []
            for idx, i in enumerate(masks):
                mask = masks[idx][crop_top : crop_top + final_height, crop_left : crop_left + final_width]
                area = round(mask.sum() / 255 / img_pixels, 3)
                bbox_areas.append(area)

            mask_path = os.path.join(data_folder, img_path.replace("images", "masks"))

            # Update labels
            df.loc[df["image_path"] == img_path, "bbox_area"] = bbox_areas

            # Correct labels based on polygon areas
            if all(area == 0 for area in bbox_areas):
                df.loc[df["image_path"] == img_path, ANOMALY_COLUMNS] = np.nan
                df.loc[df["image_path"] == img_path, "label_index"] = 0
            elif any(area == 0 for area in bbox_areas):
                cond = (df["image_path"] == img_path) & (df["bbox_area"] == 0.0)
                df.drop(df[cond].index, inplace=True)

        time_elapsed = time.time() - start_time
        print(f"completed in {time_elapsed}s")

    # Remove "original" bounding boxes columns
    df.drop(columns=BBOX_COLUMNS, inplace=True)
    # Remove duplicated rows if any (it can happen when an image becomes normal)
    df.drop_duplicates(inplace=True)

    # Overwrite labels
    df.to_csv(labels_fname, index=False)