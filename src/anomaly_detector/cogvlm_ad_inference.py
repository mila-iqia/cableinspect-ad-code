# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from src.anomaly_detector.base_anomaly_detector_inference import (
    BaseAnomalyDetector,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


class CogVLM_AD(BaseAnomalyDetector):
    def __init__(
        self,
        prompt_file: Path,
        prompt_type: str,
        model_config: dict,
        scoring_config: Optional[dict] = None,
    ):
        super().__init__(prompt_file, prompt_type, model_config, scoring_config)

        if "template_version" in model_config:
            self.template_version = model_config["template_version"]
        else:
            self.template_version = "vqa"

        tokenizer_id = model_config["tokenizer_id"]
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, add_bos_token=False
        )

        # Load the model
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        ).eval()

    def inputs_collate_fn(self, features: list) -> dict:
        """
        This function takes a list of input features and pads them to ensure that all inputs
        in the batch have the same length, making them suitable for model inference.

        Args:
            features (list): A list of input features to be collated and padded.

        Returns:
            dict: A dictionary containing the padded input features ready for model inference.
        """
        images = [feature.pop("images") for feature in features]
        self.tokenizer.padding_side = "left"
        padded_features = self.tokenizer.pad(features)
        inputs = {**padded_features, "images": images}
        return inputs

    def prepare_inputs(self, queries: List[str], images: List[Image.Image]) -> dict:
        """
        Args:
            queries (List[str]): The list of input text prompts.
            images (List[Image.Image]): The list of images to be processed.

        Returns:
            inputs (dict): The processed input dict for inference.
                input_ids (torch.tensor): batch x seq_len
                token_type_ids (torch.tensor): batch x seq_len
                attention_mask (torch.tensor): batch x seq_len
                images (List[List[torch.Tensor]] ):
                     len(images)=batch
                     the tensor has a shape of (ch,width,height)
        """
        assert len(queries) == len(images)

        inputs_list = []
        for query, image in zip(queries, images):
            inputs_sample = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=[],
                images=[image],
                template_version=self.template_version,
            )
            inputs_sample = {
                "input_ids": inputs_sample["input_ids"].to(self.device),
                "token_type_ids": inputs_sample["token_type_ids"].to(self.device),
                "attention_mask": inputs_sample["attention_mask"].to(self.device),
                "images": [
                    inputs_sample["images"][0].to(self.device).to(torch.bfloat16)
                ],
            }
            inputs_list.extend([inputs_sample])

        inputs = self.inputs_collate_fn(inputs_list)

        inputs = self.recur_move_to(
            inputs, self.device, lambda x: isinstance(x, torch.Tensor)
        )
        inputs = self.recur_move_to(
            inputs,
            torch.bfloat16,
            lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
        )
        return inputs

    def run_model(
        self, image_paths: List[Path], images: List[Image.Image]
    ) -> List[str]:
        """
        Args:
            image_path (List[Path]): The list of input image paths.
            images (List[Image.Image]): The list of images to be processed.
        Returns:
            generated_tokens (list[str]): The list of generated tokens input dict for inference.
        """
        assert self.text_prompt is not None, "Prompt is not set."

        text_prompts = [self.add_class_info((image_path)) for image_path in image_paths]
        # print(f"Input prompt: {text_prompts[0]}")

        inputs = self.prepare_inputs(queries=text_prompts, images=images)

        gen_kwargs = {"max_new_tokens": 2048, "do_sample": False}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        generated_tokens = self.tokenizer.batch_decode(generated_ids)

        return generated_tokens
