# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import requests
from PIL import Image
from src.anomaly_detector import (
    CogVLM_AD,
    LLaVA_AD,
)


def test_cogvlm() -> None:
    model_config = {
        "model_name": "CogVLM",
        "model_id": "THUDM/cogvlm-chat-hf",
        "tokenizer_id": "lmsys/vicuna-7b-v1.5",
    }
    scoring_config = {
        "scoring_prompt": "Does this figure show an anomalous or defective cable? Please answer Yes or No.",
        "scoring_type": "vqa",
        "answer_template": "Yes",
    }
    prompt_type = "zero_shot"
    prompt_file = "tests/sample/prompts.yaml"
    ad_model = CogVLM_AD(Path(prompt_file), prompt_type, model_config, scoring_config)

    assert isinstance(ad_model, CogVLM_AD)
    assert ad_model.scoring_enabled
    image_url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(image_url, stream=True).raw)
    output = ad_model.run_model([Path("test.png")] * 2, [image] * 2)
    scores = ad_model.generate_score([image] * 2)

    assert len(output) == 2
    assert isinstance(output[0], str)

    assert len(scores) == 2
    assert isinstance(scores[0], float)


def test_llava() -> None:
    model_config = {
        "model_name": "LLaVA",
        "model_id": "llava-hf/llava-1.5-7b-hf",
    }
    scoring_config = {
        "scoring_prompt": "Does this figure show an anomalous or defective cable? Please answer Yes or No.",
        "scoring_type": "vqa",
        "answer_template": "Yes",
    }
    prompt_type = "zero_shot"
    prompt_file = "tests/sample/prompts.yaml"
    ad_model = LLaVA_AD(Path(prompt_file), prompt_type, model_config, scoring_config)

    assert isinstance(ad_model, LLaVA_AD)
    assert ad_model.scoring_enabled
    image_url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(image_url, stream=True).raw)
    output = ad_model.run_model([Path("test.png")] * 2, [image] * 2)
    scores = ad_model.generate_score([image] * 2)

    assert len(output) == 2
    assert isinstance(output[0], str)

    assert len(scores) == 2
    assert isinstance(scores[0], float)
