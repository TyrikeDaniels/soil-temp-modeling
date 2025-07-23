import time
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from utils.preprocess import preprocess
from utils.val import plot_results, val_kf, val_rkf, val_tt


def main() -> None:
    """
    Main function to execute validation performance evaluation using a pipeline.

    - Loads raw data (no scaling).
    - Defines a pipeline: scaling + model.
    - Runs three validation strategies.
    - Collects RMSE and runtime results.
    - Plots and saves summary results.
    """

    # Import dataset
    df = preprocess() 

    # Create pipeline (scale data -> use on control model)
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.3, max_depth=5, subsample=0.7))
    ])

    # Create dictionary to log results
    results: Dict[str, Dict[str, List[float]]] = {name: {"rmses": [], "runtime": []} for name in ["Train/Test", "K-Fold", "Repeated K-Fold"]}

    # Validation method configurations
    k_splits, repeats, iterations = 10, 3, 13

    # Depth of focus
    focus_depth = "ST_50"

    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}\n{'-'*50}")

        # Sample without replacement, drop irrelevant depths
        df_sampled = df.sample(frac=0.8) 
        df_shuffled = shuffle(df_sampled).drop(columns=[d for d in ["ST_10", "ST_50", "ST_100"] if d != focus_depth])

        # Split features and labels
        X = df_shuffled.drop(columns=[focus_depth])
        y = df_shuffled[focus_depth]

        # For-loop for calling validation functions
        for val_name, val_func in [
            ("Train/Test", val_tt),
            ("K-Fold", val_kf),
            ("Repeated K-Fold", lambda m, X_, y_: val_rkf(m, X_, y_, k_splits, repeats))
        ]:
            
            # Perform validation on pipeline
            start = time.time()
            scores = val_func(pipeline, X, y) 
            runtime = time.time() - start
            rmse = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores

            # Append RMSE score and validation runtime
            results[val_name]["rmses"].append(rmse)
            results[val_name]["runtime"].append(runtime)

            print(f"{val_name:<17} | RMSE: {rmse:.4f} | Runtime: {runtime:.3f} s")

    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()
