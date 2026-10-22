from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb


def compute_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Percentage Error (RMSPE)."""
    mask = y_true != 0
    return float(np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100)


def load_data(store_path: str | Path, sales_path: str | Path) -> pd.DataFrame:
    """Load store metadata and historical transaction sales, merging on Store ID."""
    df_store = pd.read_csv(store_path)
    df_sales = pd.read_csv(sales_path, low_memory=False)

    merged = pd.merge(df_sales, df_store, on="Store", how="left")

    # Filter out closed days and non-positive sales
    cleaned = merged[(merged["Open"] != 0) & (merged["Sales"] > 0)].copy()
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar signals, competitor flags, and numeric encodings."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.day_of_week
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    # Categorical mappings
    store_type_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    assortment_map = {"a": 1, "b": 2, "c": 3}
    holiday_map = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3}

    df["StoreType"] = df["StoreType"].map(store_type_map).fillna(0).astype(int)
    df["Assortment"] = df["Assortment"].map(assortment_map).fillna(0).astype(int)
    df["StateHoliday"] = df["StateHoliday"].map(holiday_map).fillna(0).astype(int)

    # Competition distance
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["CompetitionDistanceLog"] = np.log1p(df["CompetitionDistance"])

    # Competition presence
    if "CompetitionOpenSinceYear" in df.columns and "CompetitionOpenSinceMonth" in df.columns:
        comp_year = df["CompetitionOpenSinceYear"].fillna(9999).astype(int)
        comp_month = df["CompetitionOpenSinceMonth"].fillna(1).astype(int)
        comp_date = pd.to_datetime(
            comp_year.astype(str) + "-" + comp_month.astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        df["CompetitionActive"] = ((df["Date"] >= comp_date) & comp_date.notna()).astype(int)
    else:
        df["CompetitionActive"] = 0

    return df


def split_time_series(
    df: pd.DataFrame, split_date: str = "2015-06-15"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Temporal split respecting chronological ordering to avoid lookahead bias."""
    sorted_df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = [
        "Store",
        "DayOfWeek",
        "Promo",
        "SchoolHoliday",
        "StateHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionDistanceLog",
        "CompetitionActive",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "IsMonthStart",
        "IsMonthEnd",
    ]
    target_col = "Sales"

    train_mask = sorted_df["Date"] < split_date
    val_mask = sorted_df["Date"] >= split_date

    X_train = sorted_df.loc[train_mask, feature_cols]
    y_train = sorted_df.loc[train_mask, target_col]
    X_val = sorted_df.loc[val_mask, feature_cols]
    y_val = sorted_df.loc[val_mask, target_col]

    return X_train, y_train, X_val, y_val, feature_cols


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "xgboost",
    n_estimators: int = 500,
) -> Tuple[object, Dict[str, float]]:
    """Train gradient booster (XGBoost or LightGBM) and evaluate on holdout set."""
    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )
    elif model_type == "lightgbm":
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(100)],
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    preds = model.predict(X_val)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
        "mae": float(mean_absolute_error(y_val, preds)),
        "rmspe": compute_rmspe(y_val.to_numpy(), preds),
    }

    return model, metrics


def save_artifacts(
    model: object,
    feature_cols: List[str],
    metrics: Dict[str, float],
    store_df: pd.DataFrame,
    output_dir: str | Path = "models",
) -> Path:
    """Persist trained model weights and store profiles for app consumption."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build per-store metadata dictionary for fast lookup in UI
    store_profiles = {}
    store_groups = store_df.drop_duplicates(subset=["Store"])
    for _, row in store_groups.iterrows():
        store_id = int(row["Store"])
        store_profiles[store_id] = {
            "StoreType": int(row["StoreType"]),
            "Assortment": int(row["Assortment"]),
            "CompetitionDistance": float(row["CompetitionDistance"]),
            "CompetitionDistanceLog": float(np.log1p(row["CompetitionDistance"])),
            "CompetitionActive": int(row.get("CompetitionActive", 0)),
        }

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "store_profiles": store_profiles,
    }

    model_file = out_path / "sales_model.joblib"
    joblib.dump(bundle, model_file)
    return model_file


def plot_feature_importance(
    model: object,
    feature_names: List[str],
    output_path: str | Path = "feature_importance.png",
) -> None:
    """Export horizontal bar chart of feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x="Importance", y="Feature", color="#2b5c8f")
    plt.title("Feature Importance in Sales Prediction", fontsize=14, weight="bold")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Rossmann Store Sales Pipeline")
    parser.add_argument("--store-path", default="data/store.csv", help="Path to store CSV")
    parser.add_argument("--sales-path", default="data/train.csv", help="Path to sales CSV")
    parser.add_argument("--split-date", default="2015-06-15", help="Validation cutoff date")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--output-dir", default="models", help="Directory for model artifacts")
    args = parser.parse_args()

    print(f"Loading data from {args.sales_path} and {args.store_path}...")
    raw_df = load_data(args.store_path, args.sales_path)

    print("Running feature engineering...")
    processed_df = engineer_features(raw_df)

    print(f"Splitting dataset chronologically at {args.split_date}...")
    X_train, y_train, X_val, y_val, feature_cols = split_time_series(
        processed_df, split_date=args.split_date
    )

    print(f"Training {args.model_type.upper()} model ({len(X_train):,} train, {len(X_val):,} val)...")
    model, metrics = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
    )

    print("\n--- Validation Metrics ---")
    print(f"RMSE  : {metrics['rmse']:.2f}")
    print(f"MAE   : {metrics['mae']:.2f}")
    print(f"RMSPE : {metrics['rmspe']:.2f}%")

    print("\nSaving model artifacts and plotting importance...")
    save_artifacts(model, feature_cols, metrics, processed_df, output_dir=args.output_dir)
    plot_feature_importance(model, feature_cols, "feature_importance.png")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()