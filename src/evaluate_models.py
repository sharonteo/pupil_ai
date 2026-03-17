# src/evaluate_models.py
#This module computes:
#- Accuracy
#- Precision
#- Recall
#- F1
#- ROC AUC
#- Sensitivity
#- Specificity
#- ROC curve arrays (FPR/TPR)


# src/evaluate_models.py

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from src.config import MODEL_DIR

def load_model(model_name: str):
    """Load a trained model from disk."""
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    return joblib.load(path)

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics including sensitivity & specificity."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "tpr": tpr,
    }

def evaluate_model(model_name: str, X_test, y_test):
    """Load a model and compute evaluation metrics."""
    model = load_model(model_name)
    y_prob = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_prob)

def evaluate_all(models_dict):
    """Evaluate all trained models and return a DataFrame."""
    rows = []
    for name, artifacts in models_dict.items():
        X_test = artifacts["X_test"]
        y_test = artifacts["y_test"]
        pipeline = artifacts["pipeline"]

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        metrics["model"] = name
        rows.append(metrics)

    return pd.DataFrame(rows)