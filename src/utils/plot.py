import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TITLE_FONTSIZE = 10
FONTSIZE = 7


def rmse_curve(
    results: dict[str, dict[str, List[float]]],
    colors: List[str],
    ax: Optional[plt.Axes] = None,
    x_label: str = "Iterations",
    title: str = "RMSE by Validation Strategy",
    legend_title: str = "Validation Strategy"
) -> None:
    """
    Plot RMSE values across different validation strategies over iterations.

    Args:
        results (dict): Dictionary where keys are validation strategy names
            and values are dictionaries with key 'rmses' containing lists of RMSE floats.
        colors (List[str]): List of colors corresponding to each validation strategy.
        ax (plt.Axes, optional): Matplotlib axis to plot on. If None, creates a new one.
        x_label (str): Label for the x-axis (default 'Iterations').
        title (str): Title of the plot (default 'RMSE by Validation Strategy').
        legend_title (str): Title for the legend (default 'Validation Strategy').

    Returns:
        None
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
    Plot a boxplot comparing RMSE distributions for multiple validation strategies.

    Args:
        data (np.ndarray): 2D array where each row corresponds to a validation method,
            and each column corresponds to an RMSE sample for that method.
        spacing (int): Spacing between boxes on the x-axis.
        colors (List[str]): Colors for each validation strategy box.
        legend_title (str): Title for the legend.
        labels (List[str]): Labels for each validation strategy.
        ax (plt.Axes, optional): Matplotlib axis to plot on. If None, creates a new one.
        title (str): Plot title (default 'RMSE Boxplot').

    Returns:
        None
    """
    if ax is None:
        _, ax = plt.subplots()

    n_methods = data.shape[0]
    positions = np.arange(n_methods) * spacing

    box = ax.boxplot(
        data.T,  # transpose so each column is one boxplot
        patch_artist=True,
        positions=positions
    )

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
    Plot Kernel Density Estimate (KDE) distributions grouped by a categorical variable.

    Args:
        data (pd.DataFrame): DataFrame with at least columns corresponding to `hue` and `x`.
        colors (List[str]): List of colors corresponding to unique values in `hue`.
        hue (str): Column name in `data` to group by (categorical variable).
        legend_label (str): Title for the legend.
        x (str): Column name in `data` for the x-axis variable.
        ax (plt.Axes, optional): Matplotlib axis to plot on. If None, creates a new one.
        title (str): Plot title (default 'RMSE Distribution').

    Returns:
        None
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
