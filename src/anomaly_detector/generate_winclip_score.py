# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: CC-BY-4.0

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from anomalib.models.image.winclip import WinClipModel

# This script works only with the latest version of anomaliib


# Function to load, resize, and convert images to tensors
def load_resize_convert_images(image_paths, size=(240, 240)):
    images = []
    transform = Compose([Resize(size), ToTensor()])  # Resize the image  # Convert the image to a tensor
    for path in image_paths:
        pil_image = Image.open(path).convert("RGB")
        tensor_image = transform(pil_image)  # Shape: [C, H, W]
        tensor_image = tensor_image.unsqueeze(0)  # Shape: [1, C, H, W]
        images.append(tensor_image)
    return torch.cat(images)  # Shape: [batch_size, C, H, W]


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    labels_file = os.path.join(dataset_path, "labels.csv")
    df = pd.read_csv(labels_file)
    image_files = df["image_path"].unique().tolist()
    output_path = args.output_path
    all_scores = []
    print("Loading model")
    model = WinClipModel("cable")
    print("Model loaded")
    for image_file in tqdm(image_files):
        image = load_resize_convert_images([os.path.join(dataset_path, image_file)])
        score, pixels = model(image)
        all_scores.append(score)
        dir_path = os.path.dirname(image_file)
        os.makedirs(os.path.join(output_path, dir_path), exist_ok=True)
        torch.save(pixels, os.path.join(output_path, image_file.replace(".png", ".pt")))
    pred_df = pd.DataFrame({"image_path": image_files, "score": all_scores})
    pred_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)
    print("Saved predictions")


def parse_args() -> argparse.Namespace:
    description = "Script to run WinCLIP."
    arg_parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--dataset-path",
        default="CableInspect-AD",
        type=str,
        help="dataset path",
    )

    arg_parser.add_argument(
        "--output-path",
        default="winclip_results",
        type=str,
        help="output path",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
