# src/train_models.py
# This module:
# - Loads data
#- Builds preprocessing
#- Trains Logistic Regression, Random Forest, and XGBoost
#- Saves each model to data/models/
#- Returns a dictionary of artifacts for downstream evaluation

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features import load_dataset, build_preprocessor, split_data
from src.config import MODEL_DIR, RANDOM_STATE

def get_models():
    """Define the ML models used in the pipeline."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

def train_and_save_models():
    """Train all models and save them to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df, X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()

    models = get_models()
    artifacts = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipeline, model_path)

        artifacts[name] = {
            "model_path": model_path,
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
        }

        print(f"[✓] Saved {name} → {model_path}")

    return artifacts

if __name__ == "__main__":
    train_and_save_models()