import multiprocessing, csv
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Any, Dict
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler


DEPTHS = ["ST_10", "ST_50", "ST_100"]



def search(df: pd.DataFrame, k: int, season: str) -> Dict[str, Dict[str, Any]]:
    """
    Conducts nested cross-validation to identify the best-performing model
    for each soil temperature depth (e.g., ST_10, ST_50, ST_100).

    For each depth:
    - Drops other depth columns to prevent leakage.
    - Uses nested CV (outer loop for evaluation, inner loop for hyperparameter tuning).
    - Compares RandomForest, GradientBoost, and XGBoost regressors.
    - Selects the model with the lowest mean RMSE across outer folds.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset filtered by season.
        k (int): Number of splits for K-Fold CV.
        season (str): Season label ('winter' or 'summer') for print/log output.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of depth → dictionary containing:
            - 'best_model': best model name (str),
            - 'best_score': best RMSE score (float),
            - 'best_params': best hyperparameters (dict).
    """
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_jobs=multiprocessing.cpu_count() // 2,
            booster="gbtree",
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0),
        'RandomForest': RandomForestRegressor(
            n_jobs=multiprocessing.cpu_count() // 2
        ),
        'GradientBoost': GradientBoostingRegressor()
    }

    param_grids = {
        'XGBoost': {
            'model__learning_rate': [0.1, 0.5],
            'model__max_depth': [4, 6, 8],
            'model__n_estimators': [150, 200],
            'model__subsample': [0.8],
            'model__colsample_bytree': [0.8],
            'model__gamma': [0, 0.1],
            'model__reg_alpha': [0],
            'model__reg_lambda': [1]
        },
        'RandomForest': {
            'model__criterion': ['squared_error'],
            'model__n_estimators': [400, 500],
            'model__max_depth': [None, 20, 30],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
            'model__max_features': ['sqrt']
        },
        'GradientBoost': {
            'model__learning_rate': [0.05, 0.1, 0.5],
            'model__max_depth': [3, 5, 7],
            'model__n_estimators': [200, 250],
            'model__subsample': [0.8, 1.0],
            'model__max_features': ['sqrt', 'log2', 0.8],
            'model__min_samples_split': [2],
            'model__min_samples_leaf': [3]
        }
    }

    results = {}

    for depth in DEPTHS:
        title = f"Modeling Soil Temperature at Depth {depth} During {season.capitalize()}\n"
        print(title + ("-" * len(title)))

        # Prepare features and target for the current depth
        copy_df = df.drop(columns=[col for col in df.columns if col in DEPTHS and col != depth])
        X_cv = copy_df.drop(columns=[depth])
        y_cv = copy_df[depth]

        best_model = None
        best_score = float('inf')
        best_name = None

        outer_cv = KFold(n_splits=k, shuffle=True)
        inner_cv = KFold(n_splits=k)

        for name in models:
            rmse_scores = []

            for train_idx, test_idx in outer_cv.split(X_cv, y_cv):
                X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
                y_train, y_test = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('model', models[name])
                ])

                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grids[name],
                    cv=inner_cv,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )

                search.fit(X_train, y_train)
                y_pred = search.predict(X_test)

                rmse = root_mean_squared_error(y_test, y_pred)
                rmse_scores.append(rmse)

            mean_score = np.mean(rmse_scores)
            print(f"→ {name} Mean RMSE (nested CV): {mean_score:.4f}")

            if mean_score < best_score:
                best_score = mean_score
                best_model = search
                best_name = name

        best_model.fit(X_cv, y_cv)

        print(f"\n✔ Best Model for {depth}: {best_name}")
        print(f"→ Final RMSE: {best_score:.3f}")
        print(f"→ Best Params: {best_model.best_params_}\n")

        results[depth] = {
            'best_model': best_name,
            'best_score': best_score,
            'best_params': best_model.best_params_
        }

    return results


def evaluation_table(search_results: Dict[str, Dict[str, Any]], season: str) -> pd.DataFrame:
    """
    Formats the model selection results into a structured DataFrame 
    for summary and export.

    Parameters:
        search_results (Dict[str, Dict[str, Any]]): Output from `search()` for one season.
        season (str): The season the models were trained on.

    Returns:
        pd.DataFrame: Summary table of best model, score, and parameters per depth.
    """
    rows = []

    for depth in DEPTHS:
        model = search_results[depth]["best_model"]
        params = search_results[depth]["best_params"]
        score = search_results[depth]["best_score"]
        summary = {
            "GradientBoost": f"lr={params.get('model__learning_rate')}, depth={params.get('model__max_depth')}, n_est={params.get('model__n_estimators')}",
            "XGBoost": f"lr={params.get('model__learning_rate')}, depth={params.get('model__max_depth')}, n_est={params.get('model__n_estimators')}",
            "RandomForest": f"depth={params.get('model__max_depth')}, n_est={params.get('model__n_estimators')}",
        }.get(model, "—")

        rows.append({
            "Depth": depth,
            "Season": season,
            "Model": model,
            "K-fold RMSE": score,
            "Key Parameters": summary
        })

    return pd.DataFrame(rows)


def write_csv(model_df: pd.DataFrame, path: str) -> None:
    """
    Saves the evaluation summary DataFrame to a CSV file.

    Parameters:
        model_df (pd.DataFrame): The summary table with best models and metrics.
        path (str): Path to the output CSV file.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=model_df.columns)
        w.writeheader()
        for _, row in model_df.iterrows():
            w.writerow(row.to_dict())