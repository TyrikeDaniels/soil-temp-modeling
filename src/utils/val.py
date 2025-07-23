import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from utils.plot import boxplot_rmse, kdeplot, rmse_curve
from typing import Dict, List


def val_tt(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Perform a train/test split validation using the provided pipeline.

    Args:
        model (Pipeline): Pipeline including scaler and estimator.
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target vector.

    Returns:
        float: Root Mean Squared Error (RMSE) on the test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, preds))


def val_kf(model: Pipeline, X: pd.DataFrame, y: pd.Series, k: int = 10) -> List[float]:
    """
    Perform K-Fold cross-validation using the provided pipeline.

    Args:
        model (Pipeline): Pipeline including scaler and estimator.
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target vector.
        k (int): Number of folds for cross-validation.

    Returns:
        List[float]: List of RMSE scores, one per fold.
    """
    rmses = []
    kf = KFold(n_splits=k, shuffle=True)
    for train_idx, val_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
    return rmses


def val_rkf(model: Pipeline, X: pd.DataFrame, y: pd.Series, k: int = 10, repeats: int = 5) -> List[float]:
    """
    Perform Repeated K-Fold cross-validation using the provided pipeline.

    Args:
        model (Pipeline): Pipeline including scaler and estimator.
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target vector.
        k (int): Number of folds per repeat.
        repeats (int): Number of repeats.

    Returns:
        List[float]: List of RMSE scores across all repeats and folds.
    """
    rmses = []
    rkf = RepeatedKFold(n_splits=k, n_repeats=repeats)
    for train_idx, val_idx in rkf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
    return rmses


def val_summary(results: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Summarize validation results with average RMSE, RMSE standard deviation, and average runtime.

    Args:
        results (dict): Dictionary mapping validation methods to their RMSEs and runtimes.

    Returns:
        pd.DataFrame: Summary table with metrics for each validation method.
    """
    return pd.DataFrame({
        "Validation Method": list(results.keys()),
        "RMSE Avg": [np.mean(results[m]["rmses"]) for m in results],
        "RMSE Std": [np.std(results[m]["rmses"]) for m in results],
        "Runtime Avg (s)": [np.mean(results[m]["runtime"]) for m in results],
    })


def plot_results(results: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Plot RMSE curves, boxplots, and KDE distributions for each validation strategy.

    Args:
        results (dict): Dictionary of RMSE and runtime results for each validation method.
    """
    colors = ["#FFA500", "#1f77b4", "#2ca02c"]
    labels = list(results.keys())

    rmse_arrays = [results[label]["rmses"] for label in labels]
    box_data = np.vstack(rmse_arrays)
    rmse_df = pd.DataFrame({label: results[label]["rmses"] for label in labels}).melt(var_name='Validation', value_name='RMSE')

    val_summary(results).to_csv("../results/eval_val.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    rmse_curve(results, colors=colors, ax=axes[0], title="RMSE Curve (Fig. 2a)", legend_title="Validation Strategy")
    boxplot_rmse(box_data, spacing=2, colors=colors, legend_title="Validation Strategy", labels=labels, ax=axes[1], title="RMSE Boxplot (Fig. 2b)")
    kdeplot(rmse_df, hue="Validation", colors=colors, x='RMSE', ax=axes[2], title="RMSE Distribution (Fig. 2c)", legend_label="Validation Strategy")

    fig.suptitle("Validation Performance Overview", fontsize=16)
    plt.subplots_adjust(left=0.075, bottom=0.2, right=0.949, top=0.7, wspace=0.3, hspace=0.274)

    plt.savefig("../results/eval_val.png", dpi=300)
    plt.show()
