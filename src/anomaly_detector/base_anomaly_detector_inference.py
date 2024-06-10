import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
import yaml
from PIL import Image


def read_prompt_from_yaml(yaml_file: Path, prompt_type: str) -> str:
    """Get the right prompt from the prompts file.

    Args:
        yaml_file (Path): Prompts yaml file.
        prompt_type (str): The type of prompt you want.
    Returns:
        text_prompt (str): The parsed prompt.
    """
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Get all prompt types from prompt file
    all_prompt_keys = list(yaml_data.keys())

    # Read the right prompt
    if prompt_type not in all_prompt_keys:
        available_keys = ", ".join(all_prompt_keys)
        raise ValueError(
            f"Invalid prompt type: {prompt_type}. Please choose from the available prompt types: {available_keys}."
        )

    text_prompt = yaml_data[prompt_type]

    return text_prompt


def add_class_name(img_path: Path, in_prompt: str) -> str:
    """Add object type in the text prompt as context.
    Note: For the classes "leather" and "wood", we do not add a determinant before the name.

    Args:
        image_path (Path): Image path.
        in_prompt (str): The input prompt to modify.
    Returns:
        out_prompt (str): The updated prompt with object type context.
    """

    class_name = str(img_path).split(os.sep)[-4]
    # Replaces <class> with the right class type
    if class_name in ["leather", "wood"]:
        out_prompt = in_prompt.replace("<class>", class_name)
    else:
        out_prompt = in_prompt.replace("<class>", f"a {class_name}")
    return out_prompt


def add_class_description(
    img_path: Path,
    in_prompt: str,
    yaml_path: str = "../prompts/mvtec_descriptions.yaml",
) -> str:
    """Add normal object description in the text prompt as context.

    Args:
        image_path (Path): Image path.
        in_prompt (str): The input prompt to modify.
        yaml_path (str): Description of normal classes.
    Returns:
        out_prompt (str): The updated prompt with object description.
    """

    with open(yaml_path, "r") as f:
        mvtec_descriptions = yaml.safe_load(f)

    class_name = str(img_path).split(os.sep)[-4]
    # Replaces <class_description> with the right class description
    out_prompt = in_prompt.replace(
        "<class_description>", mvtec_descriptions[class_name]
    )
    return out_prompt


# Base Class for Anomaly Detection
class BaseAnomalyDetector(ABC):
    def __init__(
        self,
        prompt_file: Path,
        prompt_type: str,
        model_config: dict,
        scoring_config: Optional[dict] = None,
    ):
        self.prompt_file = Path(prompt_file).expanduser().resolve()
        self.model_id = model_config["model_id"]
        self.text_prompt = read_prompt_from_yaml(prompt_file, prompt_type)
        self.prompt_type = prompt_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any
        self.tokenizer: Any

        if scoring_config is not None:
            self.answer_template = scoring_config["answer_template"]
            self.scoring_type = scoring_config["scoring_type"]
            self.scoring_prompt = scoring_config["scoring_prompt"]
            self.scoring_enabled = True
        else:
            self.scoring_enabled = False

    def add_class_info(self, image_path: Path):
        if self.prompt_type == "zero_shot_class_context":
            self.text_prompt = add_class_name(image_path, self.text_prompt)
        elif self.prompt_type == "zero_shot_class_description":
            self.text_prompt = add_class_description(image_path, self.text_prompt)
        return self.text_prompt

    def recur_move_to(
        self, item: Any, tgt: torch.device, criterion_func: Callable
    ) -> Any:
        """
        This function moves the specified item and its nested elements to a target device,
        such as a GPU or CPU, only if the provided criterion function returns True for those items
        Args:
            item (Any): The item to be moved. This can be a single item or a collection
                        (e.g., list, tuple, dict) containing nested items.
            tgt (torch.device): The target device to which the items should be moved.
            criterion_func (callable): A function that takes an item as input and returns
                                       True if the item should be moved to the target device,
                                       and False otherwise.

        Returns:
            Any: The item or collection of items after attempting to move them to the target device.

        """
        if criterion_func(item):
            device_copy = item.to(tgt)
            return device_copy
        elif isinstance(item, list):
            return [self.recur_move_to(v, tgt, criterion_func) for v in item]
        elif isinstance(item, tuple):
            return tuple([self.recur_move_to(v, tgt, criterion_func) for v in item])
        elif isinstance(item, dict):
            return {
                k: self.recur_move_to(v, tgt, criterion_func) for k, v in item.items()
            }
        else:
            return item

    @abstractmethod
    def prepare_inputs(self, queries: List[str], images: List[Image.Image]):
        pass

    @abstractmethod
    def run_model(self, image_path: List[Path], images: List[Image.Image]) -> List[str]:
        pass

    def update_prompt(self, new_prompt: str) -> None:
        self.text_prompt = new_prompt

    def generate_score(self, images: List[Image.Image]) -> List[float]:
        """
        Generates score for anomalies.
        Args:
            images: List of images to be scored.
        Returns:
            score: The score for the anomalies.
        """
        assert self.scoring_enabled, "Scoring is not enabled."

        queries = [self.scoring_prompt] * len(images)
        inputs = self.prepare_inputs(queries, images)
        tokens_of_ans = self.tokenizer.encode(
            self.answer_template, add_special_tokens=False
        )

        # TODO: Implement multiple tokens in the answer template
        # This requires extending inputs to include multiple tokens
        # at the end, which would require adding an append_tokens method.

        assert (
            len(tokens_of_ans) == 1
        ), "Only one token is allowed in the answer template"

        tokens_of_ans_list = [tokens_of_ans] * len(images)

        with torch.no_grad():
            output = self.model(**inputs, return_dict=True)
            logits = output.logits.detach()

        # # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        # get the logits of the last token
        last_logits = logits[:, -1, :]  # (batch, seq_len)
        labels_at_targets = torch.tensor(tokens_of_ans_list)  # (batch, 1)
        labels_at_targets = labels_at_targets.to(last_logits.device)

        scores = []
        for k in range(len(tokens_of_ans_list)):
            scores.append(
                (-loss_fct(last_logits[k].unsqueeze(0), labels_at_targets[k]))
                .exp()
                .item()
            )

        return scores
