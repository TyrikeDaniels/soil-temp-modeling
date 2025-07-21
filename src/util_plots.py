from typing import List, Optional
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TITLE_FONTSIZE = 10
FONTSIZE = 7

def rmse_curve(
    results: dict,
    colors: List[str],
    ax: Optional[plt.Axes] = None,
    x_label: str = "Iterations",
    title: str = "RMSE by Validation Strategy",
    legend_title: str = "Validation Strategy"
) -> None:
    """
    Plot RMSE values across different validation strategies.

    Args:
        results (dict): Keys are validation names, values dicts with 'rmses' (List[float]).
        colors (List[str]): Colors for each validation method.
        ax (plt.Axes, optional): Axis to plot on.
        x_label (str): Label for x-axis.
        title (str): Plot title.
        legend_title (str): Legend title.
    """
    if ax is None:
        _, ax = plt.subplots()

    labels = list(results.keys())
    x_vals = np.arange(1, len(results[labels[0]]["rmses"]) + 1)

    for i, label in enumerate(labels):
        linestyle = "--" if "Train" in label else "-"
        ax.plot(
            x_vals,
            results[label]["rmses"],
            color=colors[i],
            linestyle=linestyle,
            marker=".",
            label=label
        )

    ax.set_xticks(x_vals)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_ylabel("RMSE", fontsize=FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    max_rmse = max(np.concatenate([results[label]["rmses"] for label in labels]))
    ax.set_ylim(0, max_rmse * 1.1)
    ax.grid(True, axis='y')
    ax.tick_params(axis='both', labelsize=FONTSIZE)

    legend_handles = [
        Line2D([0], [0], color=colors[i], linestyle="--" if "Train" in labels[i] else "-", marker=".", label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        fontsize=FONTSIZE,
        title_fontsize=FONTSIZE,
        loc='upper right'
    )

def boxplot_rmse(
    data: np.ndarray,  # shape: (n_methods, n_samples_per_method)
    spacing: int,
    colors: List[str],
    legend_title: str,
    labels: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "RMSE Boxplot"
) -> None:
    """
    Plot boxplot of RMSE values for each validation strategy (each row in data is one method).
    """
    if ax is None:
        _, ax = plt.subplots()

    n_methods = data.shape[0]
    positions = np.arange(n_methods) * spacing

    box = ax.boxplot(data.T,  # transpose so each column = 1 box
                    patch_artist=True,
                    positions=positions)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    ax.tick_params(axis='both', labelsize=FONTSIZE)

    legend_handles = [
        Patch(facecolor=color, label=label)
        for color, label in zip(colors, labels)
    ]
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        fontsize=FONTSIZE,
        title_fontsize=FONTSIZE
    )
    ax.grid(True, axis='y')

def kdeplot(
    data: pd.DataFrame,
    colors: List[str],
    hue: str,
    legend_label: str,
    x: str,
    ax: Optional[plt.Axes] = None,
    title: str = "RMSE Distribution"
) -> None:
    """
    Plot KDE distributions grouped by `hue` column with custom colors.

    Args:
        data (pd.DataFrame): Melted dataframe with at least columns for hue and x-axis variable.
        colors (List[str]): Colors corresponding to unique hue values.
        hue (str): Column name to group/color by.
        legend_label (str): Legend title.
        x (str): Column name for x-axis variable.
        ax (plt.Axes, optional): Axis to plot on.
        title (str): Plot title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    unique_hues = data[hue].unique()
    palette = {val: colors[i] for i, val in enumerate(unique_hues)}

    sns.kdeplot(
        data=data,
        x=x,
        hue=hue,
        palette=palette,
        common_norm=False,
        fill=True,
        legend=False,
        ax=ax
    )

    legend_handles = [Patch(facecolor=palette[val], label=val) for val in unique_hues]
    ax.legend(handles=legend_handles, title=legend_label, fontsize=FONTSIZE, title_fontsize=FONTSIZE, loc='upper right')

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel(x, fontsize=FONTSIZE)
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
