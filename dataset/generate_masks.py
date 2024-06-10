#!/usr/bin/env python3

"""Generate labels for the CableInspect-AD dataset."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: CC-BY-4.0

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import tqdm
from PIL import Image
from pycocotools.coco import COCO

def parse_args() -> Namespace:
    """Parser for the command line arguments.

    Returns:
        arguments (Namespace): The arguments.
    """
    parser = ArgumentParser(description="Preprocess CableInspect-AD images and generate masks.")

    parser.add_argument("--data-folder", type=str, help="Data folder.")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    data_folder = args.data_folder

    # Create masks folder
    cables = ["cable_1", "cable_2", "cable_3"]
    for cable in cables:
        masks_dir = os.path.join(data_folder, f"{cable}/masks/01")
        Path(masks_dir).mkdir(parents=True, exist_ok=True)

    for cable in cables:
        print(f"Building masks for {cable}")
        coco = COCO(os.path.join(data_folder, f"{cable}_seg.json"))
        cat_ids = coco.getCatIds()
        img_ids = coco.getImgIds()

        for img_id in tqdm.tqdm(img_ids):
            img_info = coco.imgs[img_id]
            anns_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            if anns:
                # Construct the binary mask
                mask = coco.annToMask(anns[0])>0
                for i in range(len(anns)):
                    mask += coco.annToMask(anns[i])>0
                # Save mask
                mask_path = img_info["file_name"].replace("images", "masks")
                mask = Image.fromarray(mask)
                mask.save(os.path.join(data_folder, mask_path))
