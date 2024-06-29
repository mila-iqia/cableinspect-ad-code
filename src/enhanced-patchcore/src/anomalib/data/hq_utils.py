"""Utility to build the HQ datasets splits."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import combinations

import networkx
import pandas as pd

from anomalib.data.utils import LabelName

COLUMN_NAMES = [
    "cable_id",
    "side_id",
    "pass_id",
    "image_path",
    "mask_path",
    "label_index",
    "frame_id",
    "anomaly_id",
]


def check_df_cable_side_pass(cbl_side_labels: pd.DataFrame) -> tuple:
    """Check dataframe contains only frames from the same cable side pass.

    Args:
        cbl_side_labels (pd.DataFrame): Cable side dataframe labels.

    Returns:
        tuple: Tuple containing cable and side ID.

    Raises:
        Value Error
            If multiple cables or multiple cable sides exist in the dataframe.
    """
    cbl_id = cbl_side_labels["cable_id"].to_numpy()
    if (cbl_id[0] == cbl_id).all():
        cbl_id = cbl_id[0]
    else:
        raise ValueError("Multiple cable labels given.")

    side_id = cbl_side_labels["side_id"].to_numpy()
    if (side_id[0] == side_id).all():
        side_id = side_id[0]
    else:
        raise ValueError("Multiple cable side labels given.")

    return cbl_id, side_id


def get_anomaly_ids(
    cbl_side_abnormal_img_labels: pd.DataFrame,
    connected_anomalies: list,
    split_ratio: float,
    old_selected_anomalies: set,
    kfold: bool = False,
    train_anomaly_id: set | None = None,
) -> set:
    """Get IDs of anomalies that fit into the splits based on the ratios.

    This function takes abnormal labels from a given cable side.

    Args:
        cbl_side_abnormal_img_labels (pd.DataFrame): Abnormal image cable side labels.
        connected_anomalies (list): Anomalies that are connected.
        split_ratio (float): Ratio for the splits.
        old_selected_anomalies (set): Anomalies that have already been selected for a different split.
        kfold (bool, optional): Flag to indicate whether the split is kfold. Defaults to False.
        train_anomaly_id (set | None, optional): Set of anomalies occurring in the train set in kfold setting.
            Defaults to None.

    Returns:
        selected_anomalies (set): Anomalies that are selected.
    """
    cbl_id, side_id = check_df_cable_side_pass(cbl_side_abnormal_img_labels)

    count = 0
    selected_anomaly = set()
    ids = sorted(cbl_side_abnormal_img_labels["anomaly_id"].unique())
    # Remove the A-side anomaly ids that appear on B-side.
    # This is done because A-side ids do not match B-side ids
    # and we use the ascending order of ids on side B to separate the frames.
    # Note: we might have a small leak between sides as we consider them independent, but they are not completely.
    if cbl_id == "C03" and side_id == "B":
        ids = [i for i in ids if i not in ["084_00", "087_00", "092_00"]]
    elif cbl_id == "C02" and side_id == "B":
        ids = [i for i in ids if i not in ["047_00", "057_00"]]

    if kfold:
        for anomaly in ids:
            cond1 = anomaly in train_anomaly_id  # type: ignore
            cond2 = anomaly not in selected_anomaly
            if cond1 and cond2:
                anomaly_is_connected = [i for i in connected_anomalies if anomaly in i]
                if anomaly_is_connected:
                    selected_anomaly.update(anomaly_is_connected[0])
                else:
                    selected_anomaly.update({anomaly})
        return selected_anomaly

    # Number of anomalies we want to select for the first split
    selected_anomaly_count = int(len(ids) * split_ratio)
    for anomaly in ids:
        cond1 = count < selected_anomaly_count
        cond2 = anomaly not in selected_anomaly
        cond3 = anomaly not in old_selected_anomalies
        if cond1 and cond2 and cond3:
            anomaly_is_connected = [i for i in connected_anomalies if anomaly in i]
            if anomaly_is_connected:
                selected_anomaly.update(anomaly_is_connected[0])
                count += len(anomaly_is_connected[0])
            else:
                selected_anomaly.update({anomaly})
                count += 1
    return selected_anomaly


def identify_connected_anomalies(cbl_side_abnormal_img_labels: pd.DataFrame) -> list:
    """Get unique anomalies and identify their connections using a graph.

    This function takes abnormal labels from a given cable side.

    Args:
        cbl_side_abnormal_img_labels (pd.DataFrame): Abnormal image cable side labels.

    Returns:
        connected_anomalies (list): Connected anomalies.
    """
    _, _ = check_df_cable_side_pass(cbl_side_abnormal_img_labels)

    lists = []
    abnormal_img_paths = cbl_side_abnormal_img_labels["image_path"].unique()
    for img in sorted(abnormal_img_paths):
        img_info = cbl_side_abnormal_img_labels[cbl_side_abnormal_img_labels["image_path"] == img]
        anomaly_id = img_info["anomaly_id"].tolist()
        lists.append(anomaly_id)
    # A graph is used to connect the anomalies that appear in a single image.
    # That way, we make sure that we have no leak between the splits.
    anomaly_graph = networkx.Graph()
    for sub_list in lists:
        for edge in combinations(sub_list, r=2):
            anomaly_graph.add_edge(*edge)
    connected_anomalies = list(networkx.connected_components(anomaly_graph))
    return connected_anomalies


def find_next_overlapping_anomaly(selected_anomaly_df: pd.DataFrame, cable_labels: pd.DataFrame) -> int | None:
    """Find the next overlapping anomaly ID on a given cable.

    Args:
        selected_anomaly_df (pd.DataFrame): Selected anomaly information.
        cable_labels (pd.DataFrame): Cable labels.

    Returns:
        int | None: If exist return the ID of the first overlapping anomaly between selected sides and passes.
    """
    all_anomaly_ids = []
    for i, row in selected_anomaly_df.iterrows():
        side, pass_id, begin_frame, end_frame, num_train, num_val = row
        side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)
        anomaly_cond = cable_labels["label_index"] == 1
        after_train_cond = cable_labels["frame_id"] > end_frame
        before_train_cond = cable_labels["frame_id"] < begin_frame
        anomaly_ids = cable_labels[side_pass_cond & after_train_cond & anomaly_cond]["anomaly_id"].unique().tolist()
        anomaly_ids += (
            cable_labels[side_pass_cond & before_train_cond & anomaly_cond]["anomaly_id"].unique().tolist()
        )
        all_anomaly_ids.append(anomaly_ids)

    # Given the anomaly IDs in all sides and passes, return the first anomaly ID presented in all of them.
    if len(all_anomaly_ids) > 1:
        for anomaly_id in all_anomaly_ids[0]:
            found = True
            for sub_list in all_anomaly_ids[1:]:
                if anomaly_id not in sub_list:
                    found = False
            if found:
                return anomaly_id
    return None


def get_anomaly_group(cable_labels: pd.DataFrame, anomaly_group_id: int) -> set:
    """Get groups of connected anomalies and return an anomaly group at the given index.

    Args:
        cable_labels (pd.DataFrame): Cable labels.
        anomaly_group_id (int): ID of the anomaly cluster.

    Returns:
        selected_group_anomaly_ids (set): Set of selected anomalies.
    """
    connected_anomalies_group = []
    individual_anomalies = []

    # Consider sides independently.
    for side in ["A", "B"]:
        # Identify the unique anomalies.
        side_cond = cable_labels["side_id"] == side
        label_cond = cable_labels["label_index"] == 1
        abnormal_img_labels = cable_labels[side_cond & label_cond].copy()
        individual_anomalies += abnormal_img_labels["anomaly_id"].unique().tolist()
        connected_anomalies_group += identify_connected_anomalies(abnormal_img_labels)

    # Extract non-connected anomalies.
    for connected_anomaly in connected_anomalies_group:
        for anomaly in connected_anomaly:
            if anomaly in individual_anomalies:
                individual_anomalies.remove(anomaly)

    # Add groups of non-connected anomalies.
    num_groups = len(connected_anomalies_group)
    connected_anomalies_group += sorted(individual_anomalies)

    if anomaly_group_id >= len(connected_anomalies_group):
        raise ValueError(f"ID out of range. For this cable, valid range is [0, {len(connected_anomalies_group)-1}].")

    selected_group_anomaly_ids = connected_anomalies_group[anomaly_group_id]

    # Apply set operation only when there are clusters, not on individual anomaly IDs.
    if anomaly_group_id < num_groups:
        selected_group_anomaly_ids = set(selected_group_anomaly_ids)
    else:
        selected_group_anomaly_ids = {selected_group_anomaly_ids}

    return selected_group_anomaly_ids


def add_split(cable_labels: pd.DataFrame, side_pass_cond: bool, begin_frame: int, end_frame: int, split: str) -> list:
    """Add appropriate split name based on the beginning and ending position of frames.

    Args:
        cable_labels (pd.DataFrame): Cable labels. Modified in place.
        side_pass_cond (pd.Series): Cable side and pass IDs condition.
        begin_frame (int): Frame where the split begins.
        end_frame (int): Frame where the split ends.
        split (str): Name of the split

    Returns:
        anomaly_ids (list): List of anomalies within the beginning and ending frames.
    """
    end_frames_cond = cable_labels["frame_id"] <= end_frame
    begin_frames_cond = cable_labels["frame_id"] > begin_frame
    anomaly_label_cond = cable_labels["label_index"] == 1

    # This happens at the end of cable, where we have to sample from the beginning
    if end_frame < begin_frame:
        indices = cable_labels.index[side_pass_cond & end_frames_cond].to_list()
        indices += cable_labels.index[side_pass_cond & begin_frames_cond].to_list()
    elif end_frame == begin_frame:
        indices = cable_labels.index[side_pass_cond & end_frames_cond].to_list()
    else:
        indices = cable_labels.index[side_pass_cond & end_frames_cond & begin_frames_cond].to_list()

    cable_labels.loc[indices, "split"] = [split] * len(indices)

    # Additional anomalies that are present within the beginning and ending frames
    anomaly_ids = cable_labels[side_pass_cond & end_frames_cond & begin_frames_cond & anomaly_label_cond][
        "anomaly_id"
    ].to_list()
    return anomaly_ids


def remove_additional_anomalies(
    selected_anomaly_df: pd.DataFrame, cable_labels: pd.DataFrame, additional_train_anomalies: set
):
    """Remove additional anomalies that were captured while sampling the nominal frames.

    Args:
        selected_anomaly_df (pd.DataFrame): Selected anomaly information. Modified inplace.
        cable_labels (pd.DataFrame): Data labels.
        additional_train_anomalies (set): Set of anomalies that were added while sampling nominal frames.
    """
    for i, row in selected_anomaly_df.iterrows():
        side, pass_id, begin_frame, end_frame, num_train, num_val = row
        side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)

        other_cond = cable_labels["anomaly_id"].isin(additional_train_anomalies)
        lost_end_frame = cable_labels[side_pass_cond & other_cond]["frame_id"].max()

        # This happens when lost end frame reaches the beginning of the cable.
        lost_end_frame_min = cable_labels[side_pass_cond & other_cond]["frame_id"].min()
        if lost_end_frame_min < begin_frame:
            lost_end_frame = lost_end_frame_min

        # Check if anomaly is already included.
        # We only need to remove these anomalies from side/pass where this anomaly is still in test set.
        if end_frame > begin_frame:
            # Case where the end reached the end of the cable, and we sample from the beginning
            if begin_frame <= lost_end_frame <= end_frame:
                continue
        else:
            if lost_end_frame >= begin_frame or lost_end_frame <= end_frame:
                continue

        # Update the last frame
        selected_anomaly_df.loc[i, "end_frame"] = lost_end_frame
        add_split(cable_labels, side_pass_cond, end_frame, lost_end_frame, "lost")


def remove_leaking_anomalies_from_passes(selected_anomaly_df: pd.DataFrame, cable_labels: pd.DataFrame):
    """Remove additional anomalies in other passes that were captured while sampling the nominal frames.

    Args:
        selected_anomaly_df (pd.DataFrame): Selected anomaly information.
        cable_labels (pd.DataFrame): Data labels.

    Returns:
        additional_anomalies (list): List of additional anomaly IDs.

    Raises:
        ValueError
            If some anomalies are slipping through while removing anomalies from other passes.
    """
    passes = cable_labels["pass_id"].unique().tolist()
    all_anomalies = []

    # We consider sides to be independant. There should be only 1 value here.
    if len(selected_anomaly_df["side_id"].unique().tolist()) > 0:
        side_id = selected_anomaly_df["side_id"].unique().tolist()[0]
    else:
        side_id = None

    for i, row in selected_anomaly_df.iterrows():
        side, pass_id, begin_frame, end_frame, num_train, num_val = row
        if pass_id in passes:
            passes.remove(pass_id)

        side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)

        before_train_end_cond = cable_labels["frame_id"] <= end_frame
        after_train_begin_cond = cable_labels["frame_id"] >= begin_frame
        anomaly_cond = cable_labels["label_index"] == 1

        if end_frame > begin_frame:
            anomaly_ids = sorted(
                cable_labels[side_pass_cond & before_train_end_cond & after_train_begin_cond & anomaly_cond][
                    "anomaly_id"
                ]
                .unique()
                .tolist()
            )
        else:
            anomaly_ids = sorted(
                cable_labels[side_pass_cond & after_train_begin_cond & anomaly_cond]["anomaly_id"].unique().tolist()
            )
            anomaly_ids += sorted(
                cable_labels[side_pass_cond & before_train_end_cond & anomaly_cond]["anomaly_id"].unique().tolist()
            )

        all_anomalies += anomaly_ids

    additional_anomalies = []
    for pass_id in passes:
        side_pass_cond = (cable_labels["side_id"] == side_id) & (cable_labels["pass_id"] == pass_id)
        split_frame_cond = cable_labels["anomaly_id"].isin(all_anomalies)

        # These are the boundary frames that contain the selected anomaly cluster
        split_begin_frame = cable_labels[side_pass_cond & split_frame_cond]["frame_id"].min()
        split_end_frame = cable_labels[side_pass_cond & split_frame_cond]["frame_id"].max()

        anomaly_ids = add_split(cable_labels, side_pass_cond, split_begin_frame, split_end_frame, "lost")
        anomaly_ids = list(set(anomaly_ids))
        if len(anomaly_ids) > 0:
            additional_anomalies += list(set(anomaly_ids).intersection(*all_anomalies))
    if len(additional_anomalies) > 0:
        raise ValueError("Looks like some anomalies are slipping through while removing anomalies from other passes.")
    return additional_anomalies


def split_cable(
    cable_labels: pd.DataFrame,
    order: list,
    anomaly_ratio_train: float = 0.5,
    test_set: bool = False,
    anomaly_ratio_test: float = 0.5,
) -> pd.DataFrame:
    """Split a cable into train, validation and test set.

    Args:
        cable_labels (pd.DataFrame): Cable labels.
        order (list): Order of the splits.
        anomaly_ratio_train (float, optional): Ratio for the splits. Defaults to 0.5.
        test_set (bool, optional): Whether to create a test set. Defaults to False.
        anomaly_ratio_test (float, optional): Ratio for the splits. Defaults to 0.5.

    Returns:
        cable_labels (pd.DataFrame): Updated cable labels with split feature.

    Raises:
        ValueError
            If order contains wrong values or duplicates.
    """
    order_set = {"train", "val"}
    if test_set:
        order_set.add("test")
    err_msg = f"order should be a list containing those elements without duplicates: {order_set}."
    cond1 = set(order) != order_set
    cond2 = len(order) != len(order_set)
    if cond1 and cond2:
        raise ValueError(err_msg)

    if test_set:
        # Anomalies that fit in the test ratio are extracted first
        # and then the rest is divided between train and val.
        anomaly_ratios = {"test": anomaly_ratio_test}
        anomaly_ratios["train"] = (1.0 - anomaly_ratio_test) * anomaly_ratio_train
        anomaly_ratios["val"] = (1.0 - anomaly_ratio_test) * (1.0 - anomaly_ratio_train)
    else:
        anomaly_ratios = {"train": anomaly_ratio_train, "val": 1.0 - anomaly_ratio_train}

    # Ratio for the first part of the cable
    first_split_ratio = anomaly_ratios[order[0]]
    if test_set:
        # Ratio for the second part of the cable
        second_split_ratio = anomaly_ratios[order[1]]

    for side in ["A", "B"]:
        # Identify the unique anomalies.
        abnormal_img_labels = cable_labels[
            (cable_labels["side_id"] == side) & (cable_labels["label_index"] == LabelName.ABNORMAL)
        ].copy()
        connected_anomalies = identify_connected_anomalies(abnormal_img_labels)

        first_split_anomalies = get_anomaly_ids(abnormal_img_labels, connected_anomalies, first_split_ratio, set())

        if test_set:
            second_split_anomalies = get_anomaly_ids(
                abnormal_img_labels, connected_anomalies, second_split_ratio, first_split_anomalies
            )

        # For each pass find the frame where to cut the cable.
        # In the end, the cable is split in two part if no test set and three part if test set.
        for pass_id in cable_labels["pass_id"].unique():
            indices = []

            split_frame_cond1 = abnormal_img_labels["pass_id"] == pass_id
            split_frame_cond2 = abnormal_img_labels["anomaly_id"].isin(first_split_anomalies)
            first_split_frame = abnormal_img_labels[split_frame_cond1 & split_frame_cond2]["frame_id"].max()

            side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)
            first_frames_cond = cable_labels["frame_id"] <= first_split_frame
            first_set_indices = cable_labels.index[side_pass_cond & first_frames_cond].to_list()

            indices.append(first_set_indices)

            if test_set:
                split_frame_cond3 = abnormal_img_labels["anomaly_id"].isin(second_split_anomalies)
                second_split_frame = abnormal_img_labels[split_frame_cond1 & split_frame_cond3]["frame_id"].max()

                # If the first split anomalies are not in this frame, then it will be nan.
                if pd.isna(first_split_frame):
                    first_split_frame = -1

                second_frames_cond = (cable_labels["frame_id"] > first_split_frame) & (
                    cable_labels["frame_id"] <= second_split_frame
                )
                second_set_indices = cable_labels.index[side_pass_cond & second_frames_cond].to_list()

                indices.append(second_set_indices)

                third_frames_cond = cable_labels["frame_id"] > second_split_frame
                last_set_indices = cable_labels.index[side_pass_cond & third_frames_cond].to_list()

            else:
                second_frames_cond = cable_labels["frame_id"] > first_split_frame
                last_set_indices = cable_labels.index[side_pass_cond & second_frames_cond].to_list()
            indices.append(last_set_indices)

            # Assign split names to the different part of cables.
            for index, split in zip(indices, order):
                cable_labels.loc[index, "split"] = split

    # Assign lost to abnormal labels in train set.
    train_cond = cable_labels["split"] == "train"
    abnormal_cond = cable_labels["label_index"] == LabelName.ABNORMAL
    cable_labels.loc[train_cond & abnormal_cond, "split"] = "lost"

    return cable_labels


def split_cable_rotation(
    cable_labels: pd.DataFrame, num_train: int = 200, num_val: int = 25, anomaly_group_id: int = 1, buffer: int = 5
) -> tuple:
    """Split cable based on the index. Keep it rotational through out the cable.

    Args:
        cable_labels (pd.DataFrame): Data labels.
        num_train (int, optional): Number of frames in the train set. Defaults to 200.
        num_val (int, optional): Number of frames in the validation set. Defaults to 25.
        anomaly_group_id (int, optional): ID of the anomaly cluster where the train starts. Defaults to 1.
        buffer (int, optional): Number of buffer frames between the splits. Defaults to 5.

    Returns:
        tuple: K-fold labels dataframe and selected anomaly dataframe.
    """
    cable_labels["split"] = "test"

    # Get anomaly group, this will be beginning of train split.
    train_split_anomalies = get_anomaly_group(cable_labels, anomaly_group_id)

    # Create a df to store information of the selected group of anomalies to be used later.
    selected_anomaly_df = pd.DataFrame(columns=["side_id", "pass_id", "begin_frame", "end_frame", "total_frames"])

    pass_ids = sorted(cable_labels["pass_id"].unique())
    # Sides are considered independant.
    for side in ["A", "B"]:
        side_cond = cable_labels["side_id"] == side
        anomaly_cond = cable_labels["label_index"] == 1
        abnormal_img_labels = cable_labels[side_cond & anomaly_cond].copy()

        for pass_id in pass_ids:
            split_frame_cond1 = abnormal_img_labels["pass_id"] == pass_id
            split_frame_cond2 = abnormal_img_labels["anomaly_id"].isin(train_split_anomalies)

            abnormal_frame_ids = abnormal_img_labels[split_frame_cond1 & split_frame_cond2]["frame_id"]
            # These are the boundary frames that contain the selected anomaly cluster
            train_split_begin_frame = abnormal_frame_ids.min()
            train_split_end_frame = abnormal_frame_ids.max()

            side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)
            num_frames_each_side_pass = len(cable_labels[side_pass_cond]["frame_id"].unique())

            new_row = {
                "side_id": side,
                "pass_id": pass_id,
                "begin_frame": train_split_begin_frame,
                "end_frame": train_split_end_frame,
                "total_frames": num_frames_each_side_pass,
            }
            selected_anomaly_df = pd.concat([selected_anomaly_df, pd.DataFrame([new_row])], ignore_index=True)

    selected_anomaly_df.dropna(subset=["begin_frame", "end_frame"], inplace=True)
    selected_anomaly_df.reset_index(drop=True, inplace=True)

    selected_anomaly_df["num_train"] = 0
    selected_anomaly_df["num_val"] = 0
    total_frames_all_selected_anomaly_df = selected_anomaly_df["total_frames"].sum()

    # Sample nominal images proportional to the number of available frames per side pass.
    for i, row in selected_anomaly_df.iterrows():
        side, pass_id, begin_frame, end_frame, total_frames, _, _ = row
        num_train_frames = int((total_frames / total_frames_all_selected_anomaly_df) * num_train)
        selected_anomaly_df.loc[i, "num_train"] = num_train_frames

        num_val_frames = int((total_frames / total_frames_all_selected_anomaly_df) * num_val)
        selected_anomaly_df.loc[i, "num_val"] = num_val_frames + 2 * buffer

    # Add "left over" train images if any to first side pass in selected_anomaly_df.
    if selected_anomaly_df["num_train"].sum() < num_train:
        left_over = num_train - selected_anomaly_df["num_train"].sum()
        selected_anomaly_df.loc[0, "num_train"] = selected_anomaly_df.iloc[0]["num_train"] + left_over

    # Adjust the total number of val images including number of buffer images
    # Currently, we include buffer in the val split. We remove all the buffer frames later.
    num_val_with_buffer = num_val + (len(selected_anomaly_df) * buffer * 2)
    if selected_anomaly_df["num_val"].sum() < num_val_with_buffer:
        left_over = num_val_with_buffer - selected_anomaly_df["num_val"].sum()
        selected_anomaly_df.loc[0, "num_val"] = selected_anomaly_df.iloc[0]["num_val"] + left_over

    selected_anomaly_df.drop(columns=["total_frames"], inplace=True)

    splits = ["train"]
    if num_val > 0:
        splits.append("val")

    for split in splits:
        additional_train_anomalies = []
        for i, row in selected_anomaly_df.iterrows():
            side, pass_id, begin_frame, end_frame, num_train, num_val = row
            side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)

            # If the split is train, mark as lost the images in the first anomaly cluster.
            if split == "train":
                _ = add_split(cable_labels, side_pass_cond, begin_frame, end_frame, "lost")

            side_pass_labels = cable_labels[side_pass_cond].copy()

            nominal_cond = side_pass_labels["label_index"] == 0
            after_train_cond = side_pass_labels["frame_id"] > end_frame
            before_train_cond = side_pass_labels["frame_id"] < begin_frame

            # Collect all nominal frames in the side/pass in the order of preference.
            nominal_frames = sorted(set(side_pass_labels[after_train_cond & nominal_cond]["frame_id"].to_list()))
            nominal_frames += sorted(set(side_pass_labels[before_train_cond & nominal_cond]["frame_id"].to_list()))

            if len(nominal_frames) < int(num_train) or len(nominal_frames) < int(num_val):
                raise ValueError("Unable to sample enough train examples")

            if split == "train":
                nominal_end_frame = nominal_frames[int(num_train) - 1]
            else:
                nominal_end_frame = nominal_frames[int(num_val) - 1]

            anomaly_ids = add_split(cable_labels, side_pass_cond, end_frame, nominal_end_frame, split)

            # Update the end frame after adding the split.
            selected_anomaly_df.loc[i, "end_frame"] = nominal_end_frame

            additional_train_anomalies += anomaly_ids

        additional_train_anomalies_set = set(additional_train_anomalies)
        selected_anomaly_df.reset_index(drop=True, inplace=True)

        if len(selected_anomaly_df) > 1:
            intersecting_anomalies = find_next_overlapping_anomaly(selected_anomaly_df, cable_labels)
            additional_train_anomalies_set.add(intersecting_anomalies)

        # Remove all frames containing additional anomalies
        if len(additional_train_anomalies_set) > 0:
            # selected_anomaly_df and cable_labels are updated in place.
            remove_additional_anomalies(selected_anomaly_df, cable_labels, additional_train_anomalies_set)
        remove_leaking_anomalies_from_passes(selected_anomaly_df, cable_labels)

    # Keep only nominal frames in train and val
    train_cond = cable_labels["split"] == "train"
    val_cond = cable_labels["split"] == "val"
    abnormal_cond = cable_labels["label_index"] == 1
    cable_labels.loc[(train_cond | val_cond) & abnormal_cond, "split"] = "lost"

    return cable_labels, selected_anomaly_df


def prepare_cables(labels: pd.DataFrame, cable_id: str) -> pd.DataFrame:
    """Utility function to get the required subset of the dataframe.

    Args:
        labels (pd.DataFrame): Labels for the cables.
        cable_id (str): Cable to be used.

    Returns:
        cable_labels (pd.DataFrame): Labels for the cables.
    """
    cable_labels = labels[labels["cable_id"] == cable_id][COLUMN_NAMES].copy()
    cable_labels.drop_duplicates(inplace=True)
    cable_labels.reset_index(drop=True, inplace=True)
    return cable_labels


def check_overlap(labels: pd.DataFrame) -> None:
    """Utility function to check if there is an overlap between the splits.

    Args:
        labels (pd.DataFrame): Labels for the cable to be split.

    Raises:
        ValueError
            If anomaly ids leak between train, val or test.
    """
    train_cond = labels["split"] == "train"
    val_cond = labels["split"] == "val"
    test_cond = labels["split"] == "test"
    for side in ["A", "B"]:
        side_cond = labels["side_id"] == side
        val_anomaly_id = list(labels[val_cond & side_cond]["anomaly_id"].dropna().unique())
        train_anomaly_id = list(labels[train_cond & side_cond]["anomaly_id"].dropna().unique())
        test_anomaly_id = list(labels[test_cond & side_cond]["anomaly_id"].dropna().unique())
        # Check for overlap
        train_val_intersection = set(val_anomaly_id).intersection(set(train_anomaly_id))
        val_test_intersection = set(val_anomaly_id).intersection(set(test_anomaly_id))
        train_test_intersection = set(train_anomaly_id).intersection(set(test_anomaly_id))
        if train_val_intersection or val_test_intersection or train_test_intersection:
            raise ValueError("Anomaly IDs leak between train, val or test.")


def prep_merge(labels: pd.DataFrame) -> pd.DataFrame:
    """Utility function to merge the dataframes after splits.

    Args:
        labels (pd.DataFrame): Data labels.

    Returns:
        labels (pd.DataFrame): Data labels.
    """
    labels = labels[COLUMN_NAMES[:-2] + ["split"]].copy()
    labels.drop_duplicates(inplace=True)
    labels.reset_index(drop=True, inplace=True)
    return labels


def add_buffer(cable_labels: pd.DataFrame, order: list, buffer: int) -> pd.DataFrame:
    """Utility function to add buffers between splits.

    The buffer is assigned per side pass to the the first N labels of the second
    (and third) part of the cable.
    Note: This does not account for consecutive labels that belongs to the same image.

    Args:
        cable_labels (pd.DataFrame): Cable labels.
        order (list): Order of the splits.
        buffer (int): Buffer to be used between splits.

    Returns:
        cable_labels (pd.DataFrame): Updated cable labels.
    """
    for side in cable_labels["side_id"].unique():
        for pass_id in cable_labels["pass_id"].unique():
            for split in order[1:]:
                side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)
                if split == "train":
                    split_condition = (cable_labels["split"] == "train") | (cable_labels["split"] == "lost")
                else:
                    split_condition = cable_labels["split"] == split
                index_cond = cable_labels[side_pass_cond & split_condition].head(buffer).index

                cable_labels.loc[index_cond, "split"] = "buffer"
    return cable_labels


def add_buffer_kfold(cable_labels: pd.DataFrame, buffer: int, selected_anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Utility function to add buffers for kfold split.

    Note: This does not account for consecutive labels that belongs to the same image.

    Args:
        cable_labels (pd.DataFrame): Cable labels.
        buffer (int): Buffer to be used between splits.
        selected_anomaly_df (pd.DataFrame): Selected anomaly information.

    Returns:
        cable_labels (pd.DataFrame): Updated cable labels.
    """
    for i, row in selected_anomaly_df.iterrows():
        side, pass_id, begin_frame, end_frame, num_train, num_val = row
        side_pass_cond = (cable_labels["side_id"] == side) & (cable_labels["pass_id"] == pass_id)
        if begin_frame > end_frame:
            split_condition = cable_labels["split"] == "test"
            index_cond = cable_labels[side_pass_cond & split_condition].head(buffer).index
            cable_labels.loc[index_cond, "split"] = "buffer"
            index_cond = cable_labels[side_pass_cond & split_condition].tail(buffer).index
            cable_labels.loc[index_cond, "split"] = "buffer"
        else:
            split_condition = cable_labels["split"] == "test"
            before_train = cable_labels["frame_id"] < begin_frame
            index_cond = cable_labels[side_pass_cond & split_condition & before_train].tail(buffer).index
            cable_labels.loc[index_cond, "split"] = "buffer"
            after_train = cable_labels["frame_id"] > end_frame
            index_cond = cable_labels[side_pass_cond & split_condition & after_train].head(buffer).index
            cable_labels.loc[index_cond, "split"] = "buffer"
        if num_val > 0:
            split_condition = cable_labels["split"] == "val"
            index_cond = cable_labels[side_pass_cond & split_condition].head(buffer).index
            cable_labels.loc[index_cond, "split"] = "buffer"
            index_cond = cable_labels[side_pass_cond & split_condition].tail(buffer).index
            cable_labels.loc[index_cond, "split"] = "buffer"
    return cable_labels


def generate_kfold_labels(
    labels: pd.DataFrame,
    cbl: str,
    num_train: int,
    num_val: int,
    anomaly_group_id: int,
    buffer: int,
    num_k_shot: int | None = None,
) -> pd.DataFrame:
    """Generate K-fold labels based on anomalies per cable.

    Labels are assigned based on an anomaly group position. First, we identify an anomaly group to
    indicate the beginning of the train set. First set of 'num_train' frames immediately following
    the selected anomaly group position are assigned to the training set. Then, the next set of
    frames are assigned to the validation set if requested (it is optional). Finally, all
    the remaining frames are assigned to the test set.

    Args:
        labels (pd.DataFrame): Data labels.
        cbl (str): Cable ID of the cable to split.
        num_train (int): Number of frames in the train set.
        num_val (int): Number of frames in the validation set.
        anomaly_group_id (int): ID of the anomaly cluster where the train starts.
        buffer (int): Number of buffer frames between the splits.
        num_k_shot (int | None, optional): Number of frames for few/many shot training. Defaults to None.

    Returns:
        kfold_labels (pd.DataFrame): K-fold labels.
    """

    # Isolate cable of interest labels.
    cbl_labels = prepare_cables(labels, cbl)

    # There are some leaks in the cable/side. To make sides fully independant, uncomment the following line.
    # cbl_labels["anomaly_id"] = cbl_labels["anomaly_id"] + cbl_labels["side_id"]
    cbl_labels, selected_anomaly_df = split_cable_rotation(cbl_labels, num_train, num_val, anomaly_group_id, buffer)
    check_overlap(cbl_labels)

    # Add buffer
    add_buffer_kfold(cbl_labels, buffer, selected_anomaly_df)

    cbl_labels = prep_merge(cbl_labels)

    # Check if the num_train and num_val conditions are met.
    assert len(cbl_labels[cbl_labels["split"] == "train"]) == num_train
    assert len(cbl_labels[cbl_labels["split"] == "val"]) == num_val

    if num_k_shot:
        if num_k_shot > num_train:
            raise ValueError(f"num_k_shot={num_k_shot} can not be greater than num_train={num_train}")
        index_cond = cbl_labels[cbl_labels["split"] == "train"].sample(n=num_train - num_k_shot, random_state=1).index
        cbl_labels.loc[index_cond, "split"] = "lost"

    cbl_labels.reset_index(drop=True, inplace=True)

    return cbl_labels
