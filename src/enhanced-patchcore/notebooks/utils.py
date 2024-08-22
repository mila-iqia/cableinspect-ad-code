#!/usr/bin/env python3

"""Notebook utilities."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import itertools
from collections import OrderedDict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mycolorpy import colorlist as mcp
from sklearn import metrics


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


def plot_cable_splits(samples: pd.DataFrame, exp_name: str, fontsize: int) -> None:
    """Plot cable splits.

    Args:
        samples (pd.DataFrame): Dataframe that is to be shown.
        exp_name (str): Name of the experiment.
        fontsize (int): Font size for all text.
    """
    frame_id = [i.split("_")[-1] for i in samples["image_path"]]
    frame_id = [i.replace(".png", "") for i in frame_id]
    frame_id = [int(i) for i in frame_id]
    samples["frame_id"] = frame_id
    splits = samples["split"].unique().tolist()

    sort_by = ["cable_id", "side_id", "pass_id", "frame_id"]
    samples = samples.sort_values(by=sort_by)
    samples_groups = samples.groupby(sort_by[:-1])
    cmap = OrderedDict(
        {
            "train": "tab:blue",
            "val": "tab:orange",
            "test": "tab:green",
            "lost": "tab:red",
            "buffer": "tab:grey",
        }
    )

    plt.figure(figsize=(15, 7))

    labels = [f"{i[0]}-{i[1]}0{i[2]}" for i in list(samples_groups.groups.keys())]

    labels_pos = []
    for idx, label in enumerate(labels):
        y = np.array(samples_groups["label_index"].apply(list)[idx]) * 0.5 + idx
        x = samples_groups["frame_id"].apply(list)[idx]
        color = [cmap[i] for i in samples_groups["split"].apply(list)[idx]]
        plt.scatter(x, y, label=label, color=color, marker="o", s=5)
        plt.plot(x, y, color="tab:grey", alpha=0.5)
        labels_pos.append(0.25 + idx)

    handles = []
    for split, color in cmap.items():  # type:ignore[assignment]
        if split in splits:
            handles.append(mpatches.Patch(color=color, label=split))
    plt.legend(handles=handles, title="Dataset splits", title_fontsize=fontsize, fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.xlabel("Frame ID", fontsize=fontsize)
    plt.yticks(labels_pos, labels, fontsize=fontsize)
    plt.title(exp_name, fontsize=fontsize)
    plt.show()


def plot_histogram(
    bins: list,
    x: list,
    x_labels: list,
    x_threshold: float,
    legend_title: str,
    fontsize: int,
    colormap: dict | None = None,
    range_num: tuple | None = None,
    mark: bool = False,
) -> None:
    """Plot histogram of the scores and range for image visualization.

    Args:
        bins (list): List of bins to use to build the histogram.
        x (list): Input values, sequence of list or arrays of scores which are not required to be of same length.
        x_labels (list): Labels, corresponding sequence of score labels.
        x_threshold (float): Threshold for the scores.
        legend_title (str): Legend title.
        fontsize (int): Font size for all text.
        colormap (dict | None, optional): Color map.
        range_num (tuple | None): Range to highlight. Defaults to None.
        mark (bool): Whether to shade the area between range_num. Defaults to False.
    """
    plt.figure(figsize=(20, 7), dpi=80)

    # Add color for anomalous image score if available.
    color = []
    if len(x_labels) < 3:
        if "nominal" in x_labels:
            color.append("tab:blue")
        if "anomalous" in x_labels:
            color.append("tab:orange")
        ncol = 1
    else:
        color = mcp.gen_color(cmap="tab20", n=len(x_labels))
        ncol = 2

    if colormap:
        # Reset color
        color = [colormap[label] for label in x_labels]

    if range_num:
        min_num, max_num = range_num

    # Plot hist
    _, bins, _ = plt.hist(x, bins=bins, stacked=True, label=x_labels, color=color)
    plt.xticks(bins, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Plot threshold
    plt.axvline(x=x_threshold, color="r", linestyle="dashed", linewidth=4, label=f"threshold: {round(x_threshold, 2)}")

    if range_num:
        plt.axvline(x=min_num, color="g", linestyle="dotted", linewidth=4, label=f"min num: {round(min_num, 2)}")
        plt.axvline(x=max_num, color="g", linestyle="dotted", linewidth=4, label=f"max num: {round(max_num, 2)}")

    if mark:
        plt.axvspan(min_num, max_num, alpha=0.2, color="green")

    # Add title to legend
    if "nominal" not in x_labels:
        leg = plt.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), title=legend_title, title_fontsize=fontsize, fontsize=fontsize
        )
    else:
        leg = plt.legend(ncol=ncol, loc="upper right", title=legend_title, title_fontsize=fontsize, fontsize=fontsize)
    leg._legend_box.align = "left"

    # Add grid to figure
    plt.minorticks_on()
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="black")
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")

    # Add labels and legend
    plt.ylabel("Count", fontsize=fontsize)
    plt.xlabel("Anomaly Scores", fontsize=fontsize)
    plt.show()


def plot_confusion_matrix(predicted: np.array, actual: np.array, labels: str, title: str, fontsize: int) -> None:
    """Plot confusion matrix of the scores.

    Args:
        predicted (np.array): Binary prediction.
        actual (np.array): Target.
        labels (list): List of labels.
        title (str): Plot title.
        fontsize (int): Font size for all text.
    """
    cm = metrics.confusion_matrix(actual, predicted)

    cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Image count", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=fontsize)
    plt.yticks(tick_marks, labels, fontsize=fontsize)

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(j, i, f"{cm[i, j]:,}", horizontalalignment="center", color=color, fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel("True label", fontsize=fontsize)
    plt.xlabel("Predicted label", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.show()


def compute_metrics(y_true: np.array, y_pred: np.array, threshold: float, metric_names: list) -> list:
    """Compute metric scores.

    Args:
        y_true (np.array): Target.
        y_pred (np.array): Prediction.
        threshold (float): Threshold.
        metric_names (list): List of metric to compute.

    Returns:
        scores (list): List of scores.
    """
    scores = []
    y_pred_binary = np.where(np.array(y_pred) >= threshold, 1, 0)
    for metric_name in metric_names:
        if metric_name == "F1Score":
            f1score = metrics.f1_score(y_true, y_pred_binary)
            scores.append(f1score)
        if metric_name == "Precision":
            precision = metrics.precision_score(y_true, y_pred_binary)
            scores.append(precision)
        if metric_name == "Recall":
            recall = metrics.recall_score(y_true, y_pred_binary)
            scores.append(recall)
        if metric_name == "FPR":
            cm = metrics.confusion_matrix(y_true, y_pred_binary)
            # FPR = FP / (FP + TN)
            fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
            scores.append(fpr)
        if metric_name == "AUROC":
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            auroc = metrics.auc(fpr, tpr)
            scores.append(auroc)
        if metric_name == "AUPR":
            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
            aupr = metrics.auc(recall, precision)
            scores.append(aupr)
    return scores


def plot_precision_recall_curve(
    actual: pd.Series | None,
    predicted: pd.Series | None,
    actual_id: pd.Series | None,
    predicted_id: pd.Series | None,
    threshold: float,
    recall_level: str,
    precision_level: str,
    title: str,
    fontsize: int = 22,
) -> None:
    """Plot precision recall curve.

    Args:
        actual (pd.Series | None): Ground truth at image level.
        predicted (pd.Series | None): Predicted labels (scores) at image level.
        actual_id (pd.Series | None): Ground truth at anomaly ID level.
        predicted_id (pd.Series | None): Predicted labels (scores) at anomaly ID level.
        threshold (float): Threshold for the scores.
        recall_level (str): Recall level in ["image", "ID"].
        precision_level (str): Precision level in ["image", "ID"].
        title (str): Plot title.
        fontsize (int, optional): Font size for all text. Defaults to 22.
    """
    plt.figure(figsize=(7, 7), dpi=80)
    if recall_level == "ID":
        # Precision and recall at anomaly ID level
        precision, recall, thresholds = metrics.precision_recall_curve(actual_id, predicted_id)
        if precision_level == "ID":
            actual = actual_id
        else:
            # Precision at image level
            precision_img = []
            for thr in thresholds:
                predicted_binary = np.where(np.array(predicted) >= thr, 1, 0)
                prec = metrics.precision_score(actual, predicted_binary, zero_division=True)
                precision_img.append(prec)
            precision_img.append(1.0)
            precision = precision_img
    else:
        # Precision and recall at image level
        precision, recall, thresholds = metrics.precision_recall_curve(actual, predicted)

    aupr = round(metrics.auc(recall, precision), 2)
    plt.plot(recall, precision, linewidth=3.0, color="darkorange", label=f"AUC: {aupr}")

    # Plot score at selected threshold
    idx = (np.abs(thresholds - threshold)).argmin()
    prec = precision[idx]
    rec = recall[idx]
    plt.plot(rec, prec, "ro", markersize=14, label=f"threshold: {threshold}")
    plt.annotate(
        f"({rec:0.2f}, {prec:0.2f})",
        xy=(rec - 0.15, prec - 0.07),
        textcoords="data",
        fontsize=fontsize,
    )

    # Baseline in PR-curve is the prevalence of the positive class
    rate = actual.sum() / len(actual)  # type: ignore[union-attr, arg-type]
    plt.plot((0, 1), (rate, rate), color="navy", linestyle="--", linewidth=3.0)

    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)

    plt.ylabel(f"Precision {precision_level} level", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(f"Recall {recall_level} level", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.minorticks_on()
    plt.grid(axis="both", which="major", linestyle="-", linewidth="0.5", color="black")
    plt.grid(axis="both", which="minor", linestyle=":", linewidth="0.5", color="black")

    plt.title(title, fontsize=fontsize)

    plt.legend(fontsize=fontsize, loc="lower left")
    plt.show()


def plot_roc_curve(
    actual: pd.Series | None,
    predicted: pd.Series | None,
    threshold: float,
    title: str,
    fontsize: int = 22,
) -> None:
    """Plot ROC curve.

    Args:
        actual (pd.Series | None): Ground truth at image level.
        predicted (pd.Series | None): Predicted labels (scores) at image level.
        threshold (float): Threshold for the scores.
        title (str): Plot title.
        fontsize (int, optional): Font size for all text. Defaults to 22.
    """
    plt.figure(figsize=(7, 7), dpi=80)

    # Precision and recall at image level
    fpr, tpr, thresholds = metrics.roc_curve(actual, predicted)

    auroc = round(metrics.auc(fpr, tpr), 2)
    plt.plot(fpr, tpr, linewidth=3.0, color="darkorange", label=f"AUC: {auroc}")

    # Plot score at selected threshold
    idx = (np.abs(thresholds - threshold)).argmin()
    fpr = fpr[idx]
    tpr = tpr[idx]
    plt.plot(fpr, tpr, "ro", markersize=14, label=f"threshold: {threshold}")
    plt.annotate(
        f"({fpr:0.2f}, {tpr:0.2f})",
        xy=(fpr, tpr - 0.07),
        textcoords="data",
        fontsize=fontsize,
    )

    # Baseline in ROC-curve is the random model
    np.linspace(0, 1, 10)
    plt.axline((0, 0), slope=1, color="navy", linestyle="--", linewidth=3.0)

    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)

    plt.ylabel("True Positive Rate", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.minorticks_on()
    plt.grid(axis="both", which="major", linestyle="-", linewidth="0.5", color="black")
    plt.grid(axis="both", which="minor", linestyle=":", linewidth="0.5", color="black")

    plt.title(title, fontsize=fontsize)

    plt.legend(fontsize=fontsize, loc="lower right")
    plt.show()


def plot_distribution_per_group(
    df: pd.DataFrame,
    group_var: str,
    variables: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    fontsize: int = 22,
    ylim: dict | None = None,
    sorted_group_names: list | None = None,
) -> None:
    """Plot up to 2 variables distribution per group.

    Args:
        df (pd.DataFrame): Results dataframe.
        group_var (str): Variable to use to create groups.
        variables (list[str]): List of variables of interest.
        xlabel (str): Plot xlabel.
        ylabel (str): Plot ylabel.
        title (str): Plot title.
        fontsize (int, optional): Font size for all text. Defaults to 22.
        ylim (dict | None, optional): y-axis limits, under the form {"ymax": float, "ymin": float}.
            Defaults to None.
        sorted_group_names (list | None, optional): List of sorted group names. Defaults to None.
    """
    plt.figure(figsize=(10, 7), dpi=80)
    n_variables = len(variables)
    if n_variables > 2:
        raise ValueError("To many variables. Max two.")
    groups = df.groupby(group_var)[variables]

    # Box plot
    groups.boxplot(subplots=False, rot=90, grid=False)

    # Add points
    idx = 1
    labels = []
    group_names = df[group_var].unique().tolist()
    if not sorted_group_names:
        sorted_group_names = sorted(group_names)
    for group_name in sorted_group_names:
        cond = df[group_var] == group_name
        for variable in variables:
            y = df[variable][cond]
            x = [idx] * len(y)
            plt.plot(x, y, "r.", alpha=0.2)
            labels_ = f"{group_name}"
            if n_variables == 2:
                split = variable.split("_")[0]
                labels_ += f" {split}"
            if "(multiple folds)" in title or "(# of folds)" in xlabel:
                labels_ += f" ({len(y)} fold)"
            labels += [labels_]
            idx += 1

    rotation = "horizontal" if labels[0] == "Cable 1" else "vertical"
    plt.xticks(range(1, idx), labels, rotation=rotation, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if ylim:
        plt.ylim(**ylim)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    plt.minorticks_on()
    plt.grid(axis="y", which="major", linestyle="-", linewidth="0.5", color="black")
    plt.grid(axis="y", which="minor", linestyle=":", linewidth="0.5", color="black")

    plt.title(title, fontsize=fontsize)
    plt.suptitle("")
    plt.show()


def compute_metrics_per_anomaly_types(pred: pd.DataFrame, thresh: float, metrics_dict: dict) -> dict:
    """Compute metrics per anomaly types and update `metrics_dict` inplace.

    Args:
        pred (pd.DataFrame): Predictions dataframe (contains labels, anomaly scores and target).
        thresh (float): Threshold to apply on prediction.
        metrics_dict (dict): Dictionary containing the metrics
            and values for other runs if any.

    Returns:
        metrics_dict (dict): Updated metrics dict.
    """
    metrics_lst = ["F1Score", "Precision", "Recall", "AUPR"]
    # Compute metrics at the anomaly types level
    groups = pred.groupby(["anomaly_types"])["anomaly_score"].apply(list)
    labels_groups = groups.index.tolist()
    # Put normal (good) in first position
    labels_groups.remove("good")
    labels_groups.insert(0, "good")
    # Compute score by including only one type of anomaly
    for anomaly_type in labels_groups[1:]:
        y_pred = groups["good"] + groups[anomaly_type]
        y_true = [0] * len(groups["good"]) + [1] * len(groups[anomaly_type])
        scores = compute_metrics(y_true, y_pred, thresh, metrics_lst)
        for metric_name, score in zip(metrics_lst, scores):
            if anomaly_type not in metrics_dict:
                metrics_dict[anomaly_type] = {k: [] for k in metrics_lst}
            metrics_dict[anomaly_type][metric_name].append(score.round(2))
    return metrics_dict


def compute_aupr_and_f1(
    actual: pd.Series | None,
    predicted: pd.Series | None,
    actual_id: pd.Series | None,
    predicted_id: pd.Series | None,
    threshold: float,
    recall_level: str,
    precision_level: str,
) -> tuple:
    """Compute AUPR and F1.

    Args:
        actual (pd.Series | None): Ground truth at image level.
        predicted (pd.Series | None): Predicted labels (scores) at image level.
        actual_id (pd.Series | None): Ground truth at anomaly ID level.
        predicted_id (pd.Series | None): Predicted labels (scores) at anomaly ID level.
        threshold (float): Threshold for the scores.
        recall_level (str): Recall level in ["image", "ID"].
        precision_level (str): Precision level in ["image", "ID"].

    Returns:
        tuple: AUPR and F1 score.

    Raises:
        If recall_level == "image" and precision_level == "ID".
    """
    if recall_level == "ID":
        # Precision and recall at anomaly ID level
        precision, recall, thresholds = metrics.precision_recall_curve(actual_id, predicted_id)
        if precision_level == "ID":
            actual = actual_id
        else:
            # Precision at image level
            precision_img = []
            for thr in thresholds:
                predicted_binary = np.where(np.array(predicted) >= thr, 1, 0)
                prec = metrics.precision_score(actual, predicted_binary, zero_division=True)
                precision_img.append(prec)
            precision_img.append(1.0)
            precision = precision_img
    else:
        if precision_level == "image":
            # Precision and recall at image level
            precision, recall, thresholds = metrics.precision_recall_curve(actual, predicted)
        else:
            raise ValueError("precision_level=='ID' not supported with recall_level=='image'.")

    aupr = round(metrics.auc(recall, precision), 6)

    # At selected threshold
    idx = (np.abs(thresholds - threshold)).argmin()
    prec = precision[idx]
    rec = recall[idx]
    f1_score = (2 * prec * rec) / (prec + rec)
    f1_score = round(f1_score, 6)
    return aupr, f1_score
