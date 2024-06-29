# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from src.anomaly_detector.base_anomaly_detector_inference import (
    BaseAnomalyDetector,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)


class LLaVA_AD(BaseAnomalyDetector):
    def __init__(
        self,
        prompt_file: Path,
        prompt_type: str,
        model_config: dict,
        scoring_config: Optional[dict] = None,
    ):
        super().__init__(prompt_file, prompt_type, model_config, scoring_config)

        # Load the model
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = (
            LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2",
            )
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

    def reformat_prompt(self, in_prompt: str) -> str:
        """Reformats the text prompt in "USER: xxx\nASSISTANT:"

        Args:
            in_prompt (str): Input text prompt.
        Returns:
            (str): prompt compatible with LLaVA variants.
        """
        return "USER: <image>\n" + in_prompt + "\nASSISTANT:"

    def prepare_inputs(self, queries: List[str], images: List[Image.Image]):
        """
        Args:
            queries (List[str]): The list of input text prompts.
            images (List[Image.Image]): The list of images to be processed.

        Returns:

        inputs (dict): The processed input dict for inference.
                input_ids (torch.tensor): batch x seq_len
                attention_mask (torch.tensor): batch x seq_len
                pixel_values (torch.Tensor): batch x ch x width x height):

        """
        assert len(queries) == len(images)

        queries = [self.reformat_prompt(query) for query in queries]
        inputs = self.processor(text=queries, images=images, return_tensors="pt").to(
            self.device
        )

        return inputs

    def run_model(
        self, image_paths: List[Path], images: List[Image.Image]
    ) -> List[str]:
        assert self.text_prompt is not None, "Prompt is not set."

        text_prompts = [self.add_class_info((image_path)) for image_path in image_paths]

        inputs = self.prepare_inputs(queries=text_prompts, images=images)

        gen_kwargs = {"max_new_tokens": 1024, "do_sample": False}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated_ids)
