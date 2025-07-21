import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from util_plots import boxplot_rmse, kdeplot, rmse_curve

def main():


    df = process_data()

    control_est = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=.3,
        max_depth=5,
        subsample=0.7
    )

    k_splits = 10
    repeats = 3
    iterations = 12
    results = { val : { "rmses" : [], "runtime" : [] } for val in ["Train/Test", "K-Fold", "Repeated K-Fold"]}
    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")
        print("-" * 50)

        df_sampled = df.sample(frac=0.8, replace=False, random_state=i)
        X_tv, y_tv = seperate_depth(df_sampled, "ST_50")

        start = time.time()
        tt_rmse = val_tt(control_est, X_tv, y_tv)
        tt_time = time.time() - start
        results["Train/Test"]["rmses"].append(tt_rmse)
        results["Train/Test"]["runtime"].append(tt_time)
        print(f"Train/Test        | RMSE: {tt_rmse:.4f} | Runtime: {tt_time:.3f} s")

        start = time.time()
        kf_scores = val_kf(control_est, X_tv, y_tv, k_splits)
        kf_time = time.time() - start
        kf_rmse = np.mean(kf_scores)
        results["K-Fold"]["rmses"].append(kf_rmse)
        results["K-Fold"]["runtime"].append(kf_time)
        print(f"K-Fold            | RMSE: {kf_rmse:.4f} | Runtime: {kf_time:.3f} s")

        start = time.time()
        rkf_scores = val_rkf(control_est, X_tv, y_tv, k_splits, repeats)
        rkf_time = time.time() - start
        rkf_rmse = np.mean(rkf_scores)
        results["Repeated K-Fold"]["rmses"].append(rkf_rmse)
        results["Repeated K-Fold"]["runtime"].append(rkf_time)
        print(f"Repeated K-Fold   | RMSE: {rkf_rmse:.4f} | Runtime: {rkf_time:.3f} s")

    plot_validation_results(results)

def val_summary(results: dict) -> pd.DataFrame:
    """
    Summarize RMSE and runtime averages and std devs for each validation method.

    Args:
        results (dict): Dictionary with keys as validation names and values with keys "rmses" and "runtime".

    Returns:
        pd.DataFrame: Summary table with avg RMSE, std RMSE, and avg runtime.
    """

    data = {
        "Validation Method": list(results.keys()),
        "RMSE Avg": [np.mean(results[key]["rmses"]) for key in results],
        "RMSE Std": [np.std(results[key]["rmses"]) for key in results],
        "Runtime Avg (s)": [np.mean(results[key]["runtime"]) for key in results],
    }
    return pd.DataFrame(data)

def plot_validation_results(results: dict):
    """
    Plot RMSE curves, boxplots, and KDE distributions from results dict.

    Args:
        results (dict): Dictionary with keys as validation names and values as dict with 'rmses' and 'runtime' lists.
    """

    colors = ["#FFA500", "#1f77b4", "#2ca02c"]
    labels = list(results.keys())

    # Prepare RMSE arrays
    tt_rmses = results[labels[0]]["rmses"]
    kf_rmses = results[labels[1]]["rmses"]
    rkf_rmses = results[labels[2]]["rmses"]

    box_data = np.vstack([tt_rmses, kf_rmses, rkf_rmses])

    # Melted dataframe for KDE plot
    rmse_df = pd.DataFrame({
        labels[0]: tt_rmses,
        labels[1]: kf_rmses,
        labels[2]: rkf_rmses,
    })
    rmse_df_melted = rmse_df.melt(var_name='Validation', value_name='RMSE')

    summary_df = val_summary(results)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))

    rmse_curve(
        results=results,
        colors=colors,
        ax=axes[0],
        title="RMSE Curve (Fig. 1c)",
        legend_title="Validation Strategy",
    )

    boxplot_rmse(
        data=box_data,
        spacing=2,
        colors=colors,
        legend_title="Validation Strategy",
        labels=labels,
        ax=axes[1],
        title="RMSE Boxplot (Fig. 1b)",
    )

    kdeplot(
        data=rmse_df_melted,
        hue="Validation",
        colors=colors,
        x='RMSE',
        ax=axes[2],
        title="RMSE Distribution (Fig. 1a)",
        legend_label="Validation Strategy"
    )

    fig.suptitle("Validation Performance Overview", fontsize=16)
    plt.subplots_adjust(left=0.075, bottom=0.2, right=0.949, top=0.7, wspace=0.3, hspace=0.274)

    write_to_file(summary_df)
    plt.savefig("eval_val.png", dpi=300)
    plt.show()

def write_to_file(model_df):
    with open("eval_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=model_df.columns)
        w.writeheader()
        for _, row in model_df.iterrows():
            w.writerow(row.to_dict())

def seperate_depth(df, chosen_depth):
    tmp_df = df.copy()
    tmp_df = shuffle(tmp_df)  
    new_df = tmp_df.drop(columns=[depth for depth in ["ST_10", "ST_50", "ST_100"] if depth != chosen_depth])

    return new_df.drop(columns=[chosen_depth]), new_df[chosen_depth]

def val_tt(model, X_trainval, y_trainval):
    X_train, X_test, y_train, y_test = train_test_split(X_trainval, y_trainval, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    return rmse

def val_kf(model, X, y, k=10):
    f = KFold(n_splits=k, shuffle=True)
    rmses = []

    for train_idx, val_idx in f.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)

    return rmses

def val_rkf(model, X, y, k=10, repeats=5):
    rf = RepeatedKFold(n_splits=k, n_repeats=repeats)
    rmses = []

    for train_idx, val_idx in rf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)

    return rmses

def process_data(season=None):
    # Path to csv file
    PATH = "Grand Forks_daily updated.csv"

    # Import
    df = pd.read_csv(PATH)

    # Parse dates and extract time-related features
    df['Time(CST)'] = pd.to_datetime(df['Time(CST)'], format='%m/%d/%Y')
    df['Month'] = df['Time(CST)'].dt.month
    df['Year'] = df['Time(CST)'].dt.year
    df['day'] = df['Time(CST)'].dt.dayofyear
    df.drop(columns=['Time(CST)'], inplace=True)

    # Filter by season BEFORE scaling
    if season == 'winter':
        df = df[(df['Month'] >= 11) | ((1 <= df['Month']) & (df['Month'] <= 3))]
    elif season == 'summer':

        df = df[(df['Month'] >= 6) & (df['Month'] <= 9)]
    

    # Scale (excluding temporal columns)
    scalar = StandardScaler()
    exclude = ["Month", "Year", "day"]
    numeric = [col for col in df.columns if df[col].dtype in ["float64", "int64", "int32"] and col not in exclude]
    df[numeric] = scalar.fit_transform(df[numeric])

    return df

if __name__ == "__main__":
    main()