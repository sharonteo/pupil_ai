# app/streamlit_app.py

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.features import load_dataset, build_preprocessor, split_data
from src.train_models import get_models
from src.evaluate_models import compute_metrics
from src.fda_narrative_claude import generate_fda_summary
from src.config import DATA_PATH, MODEL_DIR, TARGET, ID_COLS


# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeurOptics Pupillometry – Clinical AI Dashboard",
    layout="wide",
)


# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_data
def load_df():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_or_train_models():
    """Load saved models or train them if missing."""
    df, X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    models = get_models()

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {}

    for name, model in models.items():
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")

        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
        else:
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, model_path)

        # Predictions
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        metrics = compute_metrics(y_test, y_prob)

        # ROC curve
        fpr = metrics["fpr"]
        tpr = metrics["tpr"]

        results[name] = {
            "pipeline": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "metrics": metrics,
            "fpr": fpr,
            "tpr": tpr,
        }

    return results


# ---------------------------------------------------------
# Load Data & Models
# ---------------------------------------------------------
df = load_df()
model_results = load_or_train_models()


# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
tab_overview, tab_models, tab_patients, tab_narrative = st.tabs(
    [
        "Dataset Overview",
        "Model Performance",
        "Patient Explorer",
        "Narrative Summary (FDA)",
    ]
)


# ---------------------------------------------------------
# TAB 1 — Dataset Overview
# ---------------------------------------------------------
with tab_overview:
    st.header("Dataset Overview")
    st.write(
        "Synthetic pupillometry dataset modeled after NeurOptics clinical device outputs."
    )

    st.dataframe(df.head(50), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        st.plotly_chart(px.histogram(df, x="age", nbins=20), use_container_width=True)

    with col2:
        st.subheader("NPi Distribution")
        st.plotly_chart(px.histogram(df, x="npi", nbins=20), use_container_width=True)

    st.subheader("GCS vs NPi")
    st.plotly_chart(
        px.scatter(
            df,
            x="npi",
            y="gcs",
            color="severity",
            trendline="ols",
            hover_data=["patient_id", "diagnosis"],
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------
# TAB 2 — Model Performance
# ---------------------------------------------------------
with tab_models:
    st.header("Model Performance")
    st.write("Binary endpoint: **GCS severe (≤ 8)** vs non-severe.")

    # Build metrics table
    rows = []
    for name, res in model_results.items():
        m = res["metrics"]
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df.style.format(precision=3), use_container_width=True)

    # ROC Curves
    st.subheader("ROC Curves")
    fig = go.Figure()

    for name, res in model_results.items():
        fig.add_trace(
            go.Scatter(
                x=res["fpr"],
                y=res["tpr"],
                mode="lines",
                name=f"{name} (AUC={res['metrics']['roc_auc']:.2f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=900,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# TAB 3 — Patient Explorer
# ---------------------------------------------------------
with tab_patients:
    st.header("Patient Explorer")
    st.write("Filter and explore synthetic patient-level pupillometry data.")

    col1, col2 = st.columns(2)

    with col1:
        selected_site = st.selectbox(
            "Site", ["All"] + sorted(df["site_id"].unique().tolist())
        )

    with col2:
        selected_diag = st.selectbox(
            "Diagnosis", ["All"] + sorted(df["diagnosis"].unique().tolist())
        )

    subset = df.copy()

    if selected_site != "All":
        subset = subset[subset["site_id"] == selected_site]

    if selected_diag != "All":
        subset = subset[subset["diagnosis"] == selected_diag]

    st.write(f"{len(subset)} patients in selection.")
    st.dataframe(subset.head(100), use_container_width=True)

    st.subheader("Pupil Size vs GCS")
    st.plotly_chart(
        px.scatter(
            subset,
            x="measured_pupil_size",
            y="gcs",
            color="severity",
            hover_data=["patient_id", "npi", "constriction_velocity", "latency_ms"],
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------
# TAB 4 — FDA Narrative Summary
# ---------------------------------------------------------
with tab_narrative:
    st.header("Narrative Summary for Regulatory Review")
    st.write(
        "This section uses an LLM (Claude) to generate a regulatory-style summary "
        "of dataset characteristics and model performance. All data are synthetic."
    )

    if st.button("Generate FDA-Style Narrative"):
        narrative = generate_fda_summary(df, metrics_df)
        st.markdown(narrative)