"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import IntTensor, Tensor, nn  # pylint: disable=E0611

from anomalib.data.dinov2_backbones import DINOV2_BACKBONES
from anomalib.data.simple_backbones import SIMPLE_BACKBONES
from anomalib.data.vit_backbones import VIT_BACKBONES
from anomalib.models.components import (
    DinoV2FeatureExtractor,
    DynamicBufferModule,
    FeatureExtractor,
    HuggingFaceFeatureExtractor,
    KCenterGreedy,
)
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

logger = logging.getLogger(__name__)


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        input_size: tuple[int, int],
        layers: list[str],
        backbone_config: dict | None = None,
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        feature_pooler: dict | None = None,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors
        self.backbone_config = backbone_config

        if self.backbone in DINOV2_BACKBONES:
            self.feature_extractor = DinoV2FeatureExtractor(backbone=self.backbone)
            # Note that here the full architecture is printed but in fact the norm and head component are not used
            # to extract the features.
            logger.info("\n\nBackbone: %s\nArchitecture: %s\n", self.backbone, self.feature_extractor)
        elif self.backbone in VIT_BACKBONES:
            self.feature_extractor = HuggingFaceFeatureExtractor(backbone=self.backbone)
            logger.info("\n\nBackbone: %s\nArchitecture: %s\n", self.backbone, self.feature_extractor)
        elif self.backbone in SIMPLE_BACKBONES:
            backbone_info = "\n\n:No backbone: using raw patches" if self.backbone == "raw" else "\n\nBackbone: PCA"
            logger.info("\n\n%s\n Config: %s\n", backbone_info, backbone_config)
        else:
            self.feature_extractor = FeatureExtractor(
                backbone=self.backbone, pre_trained=pre_trained, layers=self.layers
            )
            logger.info(
                "\n\nBackbone: %s\nLayers: %s\nArchitecture: %s\n", self.backbone, self.layers, self.feature_extractor
            )

            if feature_pooler is None:
                feature_pooler = {"name": "avg", "count_include_pad": True}
            if feature_pooler["name"] == "avg":
                self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1, count_include_pad=feature_pooler["count_include_pad"])
            else:
                self.feature_pooler = torch.nn.MaxPool2d(3, 1, 1)

        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size, num_neighbors=self.num_neighbors)

        self.register_buffer("embedding_width", IntTensor())
        self.embedding_width: IntTensor
        self.register_buffer("embedding_height", IntTensor())
        self.embedding_height: IntTensor
        self.register_buffer("memory_bank_positions", IntTensor())
        self.memory_bank_positions: IntTensor
        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor
        self.register_buffer("embedding", Tensor())
        self.embedding: Tensor
        self.emb_range_for_train_paths: dict[Any, tuple[Any, Any]] = {}

    def forward(
        self, input_tensor: Tensor, input_paths: list[str] | None = None, mask_neighbors: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor.
            input_paths (list[str] | None): List of paths of all the inputs. Defaults to None.
            mask_neighbors (bool): Flag to indicate whether to mask the neighbors in fully_supervised setting.
                                    Defaults to False.

        Returns:
            Tensor | tuple[Tensor, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        if self.backbone not in SIMPLE_BACKBONES:
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)

        vit_and_simple_backbones = DINOV2_BACKBONES + VIT_BACKBONES + SIMPLE_BACKBONES
        if self.backbone in vit_and_simple_backbones:
            if self.backbone in DINOV2_BACKBONES or self.backbone in VIT_BACKBONES:
                embedding = features
            elif self.backbone in SIMPLE_BACKBONES:
                # No pretrained backbone is used in this case.
                embedding = input_tensor
            if self.tiler:
                raise NotImplementedError("Tiler is not supported for PatchCore with ViT or simple backbones.")
            batch_size, width_height, _ = embedding.shape
            width = height = math.floor(width_height**0.5)
            embedding = self.reshape_embedding_vit_or_simple(embedding)

        else:
            features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
            embedding = self.generate_embedding(features)

            if self.tiler:
                embedding = self.tiler.untile(embedding)

            batch_size, _, width, height = embedding.shape
            embedding = self.reshape_embedding(embedding)

        self.embedding_width = IntTensor([width]).to(device=embedding.device)
        self.embedding_height = IntTensor([height]).to(device=embedding.device)

        if self.training:
            output = embedding
        else:
            input_indices_in_membank = []
            if mask_neighbors and input_paths is not None:
                patch_scores_lst = []
                locations_lst = []
                for i in range(0, batch_size):
                    # Patch scores has min distance between embedding of patches in img and memory bank
                    start = width * height * (i)
                    end = width * height * (i + 1)
                    patch_score, location, found_ind = self.nearest_neighbors_masked(
                        input_emb=embedding[start:end],
                        given_input_path=input_paths[i],
                        n_neighbors=1,
                        input_ind_in_membank=None,
                    )
                    patch_scores_lst.append(patch_score)
                    locations_lst.append(location)
                    input_indices_in_membank.append(found_ind)
                patch_scores = torch.cat(patch_scores_lst)
                locations = torch.cat(locations_lst)

            else:
                # apply nearest neighbor search
                patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)

            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
            # compute anomaly score
            anomaly_score = self.compute_anomaly_score(
                patch_scores, locations, embedding, mask_neighbors, input_indices_in_membank
            )
            # reshape to w, h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)

            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest-exact")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    @staticmethod
    def reshape_embedding_vit_or_simple(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Patch*Patch, Embedding] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from ViT or simple features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(-1)
        embedding = embedding.reshape(-1, embedding_size)
        return embedding

    def subsample_embedding(self, embedding: Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        if sampling_ratio < 1.0:
            # Coreset Subsampling
            sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
            idxs, coreset = sampler.sample_coreset()
            self.memory_bank = coreset
            self.memory_bank_positions = IntTensor(idxs).to(device=coreset.device)
        else:
            # No coreset subsampling
            self.memory_bank = embedding
            # All embeddings go in memory bank
            self.memory_bank_positions = torch.arange(len(embedding))
        self.embedding = embedding

    def generate_embedding_index(self, train_input_paths: list):
        """Generate the mapping between input image path and the index in the embedding tensor that it correspoonds to.

        Args:
            train_input_paths (list): Path of the training input.
        """
        emb_range_for_path = {}
        for i, input_path in enumerate(train_input_paths):
            embeddings_ind_start = i * self.embedding_width * self.embedding_height
            embeddings_ind_end = (i + 1) * self.embedding_width * self.embedding_height
            emb_range_for_path[input_path] = (embeddings_ind_start, embeddings_ind_end)
        self.emb_range_for_train_paths = emb_range_for_path

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def nearest_neighbors_masked(
        self, input_emb: Tensor, given_input_path: str, n_neighbors: int, input_ind_in_membank: list | None
    ) -> tuple[Tensor, Tensor, list]:
        """Nearest Neighbours with mask to mask own input patches when generating anomaly scores on the train set.

        Args:
            input_emb (Tensor): Features to compare the distance with the memory bank.
            given_input_path (str): input path (used as ID here) of the given input.
            n_neighbors (int): Number of neighbors to look at
            input_ind_in_membank (list | None): indices in the memory bank that has patches from given input.
                                                Defaults to None.

        Returns:
            tuple[Tensor, Tensor, list]: Tuple of Patch scores, locations of the nearest neighbor(s) and
                                        indices in the memory bank that has patches from given input.
        """
        memory_bank_positions = self.memory_bank_positions.tolist()
        distance = torch.cdist(input_emb, self.memory_bank, p=2.0)  # euclidean norm

        if input_ind_in_membank is None:
            embeddings_ind_start, embeddings_ind_end = self.emb_range_for_train_paths[given_input_path]
            input_ind_in_membank = []
            embeddings_ind_range = [i for i in range(embeddings_ind_start, embeddings_ind_end)]
            # memory_bank_positions store the indices from embedding list, indicating the indices that are in membank
            # By finding the intersection between the index of the given input in the embeddings tensor and
            # memory bank positions, we are locating the index of the given image patches in membank
            overlap_with_membank = list(set(embeddings_ind_range) & set(memory_bank_positions))

            for j in overlap_with_membank:
                input_ind_in_membank.append(memory_bank_positions.index(j))

        mask = torch.zeros_like(distance)
        # We increase the distance to a high number at these indices
        # So that the nearest neighbor does not pick these indices up.
        mask[:, input_ind_in_membank] = float("inf")
        distance = distance + mask

        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distance.min(1)
        else:
            patch_scores, locations = distance.topk(k=n_neighbors, largest=False, dim=1, sorted=True)
        return patch_scores, locations, input_ind_in_membank

    def compute_anomaly_score(
        self,
        patch_scores: Tensor,
        locations: Tensor,
        embedding: Tensor,
        mask_neighbors,
        input_ind_in_membank: list | None,
    ) -> Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores.
            input_ind_in_membank (list | None): indices in the memory bank that has patches from given input.
                                                Defaults to None.

        Returns:
            Tensor: Image-level anomaly scores
        """

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper

        if mask_neighbors and input_ind_in_membank is not None:
            support_samples = []
            for i in range(0, batch_size):
                # Patch scores has min distance between embedding of patches in img and memory bank
                _, support_sample, _ = self.nearest_neighbors_masked(
                    nn_sample[i].unsqueeze(0),
                    nn_index[i],
                    n_neighbors=self.num_neighbors,
                    input_ind_in_membank=input_ind_in_membank[i],
                )
                support_samples.append(support_sample)
            support_samples = torch.cat(support_samples)
        else:
            # indices of N_b(m^*) in the paper
            _, support_samples = self.nearest_neighbors(nn_sample, n_neighbors=self.num_neighbors)
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples], p=2.0)
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score
