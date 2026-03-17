# src/fda_narrative_claude.py

import os
import textwrap
import pandas as pd
from anthropic import Anthropic


def build_prompt(df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    """Construct the FDA-style narrative prompt for Claude."""

    n_patients = len(df)
    severe_rate = df["gcs_severe"].mean()
    diag_counts = df["diagnosis"].value_counts().to_dict()

    metrics_text = metrics_df.to_markdown(index=False)

    prompt = f"""
    You are a biostatistics and regulatory writing expert preparing a technical
    summary for an FDA-style clinical report. The dataset is fully synthetic and
    modeled after NeurOptics pupillometer outputs. The goal is to demonstrate how
    quantitative pupillary metrics may support classification of severe vs
    non-severe neurological status (GCS ≤ 8).

    Dataset Summary:
    - Number of synthetic patients: {n_patients}
    - Proportion severe (GCS ≤ 8): {severe_rate:.3f}
    - Diagnosis distribution: {diag_counts}

    Model Performance Summary:
    {metrics_text}

    Please produce a concise, structured narrative suitable for inclusion in a
    clinical study report or statistical analysis plan. Include:

    1. Brief description of dataset and endpoints.
    2. Overview of modeling approach (logistic regression, random forest, XGBoost).
    3. Interpretation of performance metrics (accuracy, precision, recall, F1,
       ROC AUC, sensitivity, specificity) with emphasis on clinical relevance.
    4. Discussion of limitations, including synthetic nature of data and need for
       prospective validation on real-world NeurOptics pupillometer data.
    5. Clear statement that this analysis is exploratory and not intended to
       support clinical decision-making without further validation.

    Use precise, regulatory-appropriate language. Limit to 3–5 paragraphs.
    """

    return textwrap.dedent(prompt).strip()


def generate_fda_summary(df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    """Generate an FDA-style narrative using Claude Sonnet 4-6."""

    prompt = build_prompt(df, metrics_df)

    # If no API key, return placeholder
    if "ANTHROPIC_API_KEY" not in os.environ:
        return (
            "Claude summary placeholder: No ANTHROPIC_API_KEY detected.\n\n"
            "Add your API key to enable full FDA-style narrative generation."
        )

    # Real Claude Sonnet 4-6 call
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=900,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.content[0].text