#!/usr/bin/env python3

"""Preprocess the HQ dataset for tight crop and tight crop imagette."""

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

from anomalib.data.utils import read_image

BBOX_COLUMNS = ["bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_rotation"]
ANOMALY_COLUMNS = [
    "anomaly_type",
    "anomaly_type_id",
    "anomaly_grade",
    "primary_identification",
    "secondary_identification",
    "identification",
    "bbox_area",
    "mask_path",
]


def parse_args() -> Namespace:
    """Parser for the command line arguments.

    Returns:
        arguments (Namespace): The arguments.
    """
    parser = ArgumentParser(description="Preprocess HQ dataset.")

    parser.add_argument("--data-folder", type=str, help="Data folder.")
    parser.add_argument("--config", type=str, default="config/tight_crop.yaml", help="Path to dataset config.")

    arguments = parser.parse_args()
    return arguments


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

    Note: If imagette in config, create imagettes from crop image if possible.
    Return the imagettes and imagettes config instead.

    Args:
        image (np.array): Image.
        img_cfg (dict): Image config. Modified in place.

    Returns:
        tuple: Image or list of imagettes, image config.
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

    if "imagette" in img_cfg:
        imagette_cfg = img_cfg["imagette"]
        imagettes, imagette_cfg = create_imagette(image, imagette_cfg)
        if not imagette_cfg:
            img_cfg = {}
        return (imagettes, img_cfg)

    else:
        return (image, img_cfg)


def create_imagette(image: np.array, imagette_cfg: dict) -> tuple:
    """Create imagettes for image/mask if possible.

    Args:
        image (np.array): Image or mask.
        imagette_cfg (dict): Imagette config. Modified in place.

    Returns:
        tuple: List of imagette, imagette config.
    """
    imagettes = []
    height, width, _ = image.shape
    final_width, final_height = imagette_cfg["width"], imagette_cfg["height"]
    n_imagette = imagette_cfg["n_imagette"]
    # If image is too small to be splitted into imagettes just delete the image
    if width >= final_width * n_imagette and height >= final_height:
        left = int((width - final_width * n_imagette) / 2)
        top = int((height - final_height) / 2)
        img_crop = image[top : top + final_height, left : left + final_width * n_imagette]
        for i in range(n_imagette):
            imagette = img_crop[:, final_width * i : final_width * (i + 1)]
            imagettes.append(imagette)
    else:
        imagette_cfg = {}
    return (imagettes, imagette_cfg)


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
    with open(args.config, encoding="utf8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    labels_fname = os.path.join(data_folder, "labels.csv")
    df = pd.read_csv(labels_fname)

    if "imagette" in cfg:
        df_imagette = pd.DataFrame(columns=df.columns.tolist())
        n_imagette = cfg["imagette"]["n_imagette"]
        imagette_width = cfg["imagette"]["width"]
        imagette_height = cfg["imagette"]["height"]
        imagette_pixels = imagette_width * imagette_height

    # Create masks sub-folders
    for cable in ["Cable_1", "Cable_2", "Cable_3"]:
        for pass_id in ["01", "02", "03"]:
            dir_path = os.path.join(data_folder, f"{cable}/masks/{pass_id}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    img_paths = df["image_path"].unique()
    abnormal_img_paths = df.dropna(subset=["identification"])["image_path"].unique()
    for img_path in img_paths:
        start_time = time.time()
        print(f"Pre-process image and labels for: {img_path}...", end=" ")

        path = os.path.join(data_folder, img_path)
        img = read_image(path)
        img, img_cfg = crop_image(img, cfg.copy())

        if not img_cfg:
            # Remove image from dataset
            # AND skip last part of the loop if no tight crop fot this image
            img_rows = df["image_path"] == img_path
            df = df[~img_rows]
            os.remove(path)
            print(f"WARNING: {img_path} has been removed from dataset.")
            continue

        elif "imagette" in img_cfg:
            for idx, imagette in enumerate(img):
                # Extract imagette labels to correct them
                df_imagette_ = df.loc[(df["image_path"] == img_path), :].copy()
                # Update image and mask paths
                df_imagette_.loc[:, "image_path"] = img_path.replace(".png", f"_{idx}.png")
                cond = df_imagette_["mask_path"].notna()
                mask_path = img_path.replace("images", "masks")
                df_imagette_.loc[cond, "mask_path"] = mask_path.replace(".png", f"_{idx}.png")
                df_imagette = pd.concat([df_imagette, df_imagette_], ignore_index=True)
                # Save imagette
                cv2.imwrite(path.replace(".png", f"_{idx}.png"), cv2.cvtColor(imagette, cv2.COLOR_RGB2BGR))
            # Delete original image
            os.remove(path)

        else:
            # Overwrite image with cropped version
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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
                masks[idx] = mask

            mask_path = os.path.join(data_folder, img_path.replace("images", "masks"))

            if "imagette" in img_cfg:
                # Get imagettes masks
                imagettes_masks = defaultdict(list)
                for mask in masks:
                    imagettes, _ = create_imagette(np.expand_dims(mask, axis=2), img_cfg["imagette"].copy())
                    for idx, imagette in enumerate(imagettes):
                        imagettes_masks[idx].append(imagette)

                # For each imagette mask
                for idx, imagettes in imagettes_masks.items():
                    bbox_areas = []
                    for imagette in imagettes:
                        area = round(imagette.sum() / 255 / imagette_pixels, 3)
                        bbox_areas.append(area)

                    # Merge and save mask
                    imagette_mask = np.array(imagettes).sum(axis=0)
                    if imagette_mask.sum() != 0:
                        # Save mask if not empty
                        imagette_mask = Image.fromarray(imagette_mask[:, :, 0].astype(np.uint8), "L")
                        imagette_mask_path = mask_path.replace(".png", f"_{idx}.png")
                        imagette_mask.save(imagette_mask_path)

                    # Update labels
                    imagette_path = img_path.replace(".png", f"_{idx}.png")
                    df_imagette.loc[(df_imagette["image_path"] == imagette_path), "bbox_area"] = bbox_areas

                    # Correct labels based on polygon areas
                    if all(area == 0 for area in bbox_areas):
                        df_imagette.loc[df_imagette["image_path"] == imagette_path, ANOMALY_COLUMNS] = np.nan
                        df_imagette.loc[df_imagette["image_path"] == imagette_path, "label_index"] = 0
                    elif any(area == 0 for area in bbox_areas):
                        cond = (df_imagette["image_path"] == imagette_path) & (df_imagette["bbox_area"] == 0.0)
                        df_imagette.drop(df_imagette[cond].index, inplace=True)

            else:
                # Merge and save mask
                mask = np.array(masks).sum(axis=0)
                if mask.sum() != 0:
                    # Save mask if not empty
                    mask = Image.fromarray(mask.astype(np.uint8))
                    mask.save(mask_path)

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

    if "imagette" in cfg:
        df = df_imagette

    # Remove "original" bounding boxes columns
    df.drop(columns=BBOX_COLUMNS, inplace=True)
    # Remove duplicated rows if any (it can happen when an image becomes normal)
    df.drop_duplicates(inplace=True)

    # Overwrite labels
    df.to_csv(labels_fname, index=False)