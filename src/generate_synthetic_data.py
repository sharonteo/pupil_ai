# src/generate_synthetic_data.py

import numpy as np
import pandas as pd

from src.config import (
    N_SAMPLES,
    RNG_SEED,
    DIAGNOSES,
    SEXES,
    SITES,
    DATA_PATH,
)

def generate_synthetic_pupillometry(n_samples: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    patient_id = np.arange(1, n_samples + 1)
    site_id = rng.choice(SITES, size=n_samples)
    age = rng.integers(18, 90, size=n_samples)
    sex = rng.choice(SEXES, size=n_samples, p=[0.55, 0.45])
    diagnosis = rng.choice(DIAGNOSES, size=n_samples)

    npi = np.clip(rng.normal(loc=3.0, scale=1.0, size=n_samples), 0, 5)

    true_pupil_size = np.clip(rng.normal(loc=3.5, scale=0.7, size=n_samples), 1.0, 7.0)
    measurement_noise = rng.normal(loc=0.0, scale=0.3, size=n_samples)
    measured_pupil_size = np.clip(true_pupil_size + measurement_noise, 1.0, 7.0)

    asym = rng.normal(loc=0.0, scale=0.4, size=n_samples)
    pupil_left = np.clip(measured_pupil_size + asym / 2, 1.0, 7.0)
    pupil_right = np.clip(measured_pupil_size - asym / 2, 1.0, 7.0)

    severity_signal = (5 - npi) + (age - 50) / 20

    constriction_velocity = np.clip(
        rng.normal(loc=2.0 - 0.15 * severity_signal, scale=0.4, size=n_samples),
        0.1, 3.5
    )

    dilation_velocity = np.clip(
        rng.normal(loc=1.8 - 0.12 * severity_signal, scale=0.4, size=n_samples),
        0.1, 3.0
    )

    latency_ms = np.clip(
        rng.normal(loc=220 + 20 * severity_signal, scale=40, size=n_samples),
        120, 600
    )

    gcs = np.clip(
        np.round(15 - severity_signal + rng.normal(0, 1.0, size=n_samples)),
        3, 15
    ).astype(int)

    gcs_severe = (gcs <= 8).astype(int)

    severity = pd.cut(
        gcs,
        bins=[2, 8, 12, 15],
        labels=["Severe", "Moderate", "Mild"]
    )

    df = pd.DataFrame({
        "patient_id": patient_id,
        "site_id": site_id,
        "age": age,
        "sex": sex,
        "diagnosis": diagnosis,
        "npi": npi,
        "true_pupil_size": true_pupil_size,
        "measured_pupil_size": measured_pupil_size,
        "pupil_left": pupil_left,
        "pupil_right": pupil_right,
        "constriction_velocity": constriction_velocity,
        "dilation_velocity": dilation_velocity,
        "latency_ms": latency_ms,
        "gcs": gcs,
        "gcs_severe": gcs_severe,
        "severity": severity,
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_pupillometry()
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset saved to: {DATA_PATH}")
    print(df.head())