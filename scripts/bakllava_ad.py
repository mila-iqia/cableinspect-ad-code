from pathlib import Path
import requests
import argparse
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.anomaly_detector import (
    LLaVA_AD,
)

def main() -> None:
    args = parse_args()
    test_csv= args.test_csv
    data_path = Path(args.data_path)
    batch_size = args.batch_size
    generate_scores = args.generate_scores
    out_csv =  args.out_csv

    model_config = {
        "model_name": "LLaVA",
        "model_id": "llava-hf/bakLlava-v1-hf",
    }
    scoring_config = {
        "scoring_prompt": "Does this figure show an anomalous or defective cable? Please answer Yes or No.",
        "scoring_type": "vqa",
        "answer_template": "Yes",
    }
    prompt_type = "zero_shot"
    prompt_file = "scripts/prompts.yaml"
    ad_model = LLaVA_AD(Path(prompt_file), prompt_type, model_config, scoring_config)
    test_df = pd.read_csv(test_csv)

    anomaly_score_list = []
    labels_targets = []
    object_categories =[]
    output_list = []

    for i in tqdm(range(len(test_df))):

            image_path = data_path / test_df["image_path"].iloc[i]
            target_value = test_df["label_index"].iloc[i]
            object_category = test_df["cable_id"].iloc[i]

            labels_targets.append(target_value)
            object_categories.append(object_category)
            image = Image.open(image_path).convert("RGB")
            
            if i % batch_size == 0:
                images = [image]
                image_path_list = [Path(image_path)] 
            else:
                images.append(image)
                image_path_list.append(Path(image_path))

            if len(images) < batch_size and i != len(test_df) - 1:
                # wait until the batch is full and the last image is processed
                continue

            if generate_scores:
                scores = ad_model.generate_score(images)
                anomaly_score_list.extend(scores)
            else:
                batch_outputs = ad_model.run_model(image_path_list ,images)
                output_list.extend(batch_outputs)

    test_df["object_category"] = pd.Series(object_categories)
    test_df["label_targets"] = pd.Series(labels_targets)
    if generate_scores:
        test_df["anomaly_score"] = pd.Series(anomaly_score_list)
    else:
        test_df["output"] = pd.Series(output_list)
    test_df.to_csv(out_csv, index=False)

def parse_args() -> argparse.Namespace:
    description = "Zero-shot AD using BakLLaVA for a series of images."
    arg_parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="data root path",
    )
    arg_parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="path to the csv file of image paths to test",
    )
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="size for parallel batched inference",
    )
    arg_parser.add_argument(
        "--generate-scores",
        type=bool,
        default=False,
        help="whether to generate anomaly scores (vqascore)",
    )
    arg_parser.add_argument(
        "--out-csv",
        default="cables_bakllava_zero_shot_vqascore.csv",
        type=str,
        help="path to the output csv file to save results",
    )
  

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()