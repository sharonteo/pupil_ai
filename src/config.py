# src/config.py

import os

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "pupillometry_synthetic.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")

# -----------------------------
# Target & Feature Definitions
# -----------------------------
TARGET = "gcs_severe"

ID_COLS = ["patient_id", "site_id"]

CATEGORICAL_COLS = [
    "sex",
    "diagnosis",
]

NUMERIC_COLS = [
    "age",
    "npi",
    "true_pupil_size",
    "measured_pupil_size",
    "pupil_left",
    "pupil_right",
    "constriction_velocity",
    "dilation_velocity",
    "latency_ms",
]

# -----------------------------
# Modeling Parameters
# -----------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42

# -----------------------------
# Synthetic Data Parameters
# -----------------------------
N_SAMPLES = 5000
RNG_SEED = 42

DIAGNOSES = [
    "TBI",
    "Stroke",
    "Post-cardiac arrest",
    "Sepsis-associated encephalopathy",
    "Normal pressure hydrocephalus",
    "Other",
]

SEXES = ["M", "F"]

SITES = [f"SITE_{i:02d}" for i in range(1, 11)]