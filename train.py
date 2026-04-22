"""
train.py — Train the RandomForest salary model and save the bundle.
Run: python train.py
"""

import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Paths:
    data_csv:    str = os.path.join("data", "salary_data.csv")
    model_dir:   str = "models"
    bundle_path: str = os.path.join("models", "salary_model.pkl")


def _parse_skills(series: pd.Series) -> List[List[str]]:
    """Parse skills column stored as Python-list strings."""
    result = []
    for raw in series.astype(str):
        val = ast.literal_eval(raw)
        if not isinstance(val, list):
            raise ValueError("Skills must be a list.")
        result.append([str(x) for x in val])
    return result


def _fit_transformers(
    df: pd.DataFrame,
) -> Tuple[StandardScaler, OneHotEncoder, MultiLabelBinarizer]:
    """Fit all feature transformers on the full dataset."""
    scaler = StandardScaler()
    ohe    = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    mlb    = MultiLabelBinarizer()
    scaler.fit(df[["years_experience"]])
    ohe.fit(df[["position", "location"]])
    mlb.fit(df["skills"].tolist())
    return scaler, ohe, mlb


def _build_X(
    df: pd.DataFrame,
    scaler: StandardScaler,
    ohe: OneHotEncoder,
    mlb: MultiLabelBinarizer,
) -> np.ndarray:
    """Transform dataframe into feature matrix."""
    x_years  = scaler.transform(df[["years_experience"]])
    x_cat    = ohe.transform(df[["position", "location"]])
    x_skills = mlb.transform(df["skills"].tolist())
    return np.hstack([x_years, x_cat, x_skills])


def main() -> None:
    paths = Paths()
    if not os.path.exists(paths.data_csv):
        raise FileNotFoundError(
            f"Dataset not found at {paths.data_csv}. Run: python generate_data.py"
        )

    df = pd.read_csv(paths.data_csv)
    df["skills"] = _parse_skills(df["skills"])

    scaler, ohe, mlb = _fit_transformers(df)
    X = _build_X(df, scaler, ohe, mlb)
    y = df["final_salary"].to_numpy(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(
        n_estimators=450, min_samples_split=3,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    metrics: Dict[str, float] = {
        "mae":  float(mean_absolute_error(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "r2":   float(r2_score(y_te, pred)),
    }

    pos_names   = [f"position={c}" for c in ohe.categories_[0]]
    loc_names   = [f"location={c}" for c in ohe.categories_[1]]
    skill_names = [f"skill={c}"    for c in mlb.classes_]
    feature_names = ["years_experience"] + pos_names + loc_names + skill_names

    os.makedirs(paths.model_dir, exist_ok=True)
    bundle = dict(
        model=model, scaler=scaler, ohe=ohe, mlb=mlb,
        feature_names=feature_names, metrics=metrics,
    )
    joblib.dump(bundle, paths.bundle_path)

    print(f"Model saved → {paths.bundle_path}")
    print(f"MAE : INR {metrics['mae']:>12,.0f}")
    print(f"RMSE: INR {metrics['rmse']:>12,.0f}")
    print(f"R²  :     {metrics['r2']:>12.4f}")


if __name__ == "__main__":
    main()