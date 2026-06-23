"""
Section: Dataset / EDA
Run this against the REAL training_setA / training_setB folders.
Point DATA_DIRS at your actual downloaded data.
"""
import pandas as pd
import numpy as np
import glob
import os

DATA_DIRS = [
      "physionet.org/files/challenge-2019/1.0.0/training/training_setA",
      "physionet.org/files/challenge-2019/1.0.0/training/training_setB",
  ] # adjust paths as needed

def load_all_patients(data_dirs):
    """Load every .psv file, tag with patient id and hospital source."""
    frames = []
    for d in data_dirs:
        source = "A" if "setA" in d else "B"
        files = sorted(glob.glob(os.path.join(d, "*.psv")))
        for fp in files:
            pid = os.path.basename(fp).replace(".psv", "")
            df = pd.read_csv(fp, sep="|")
            df["patient_id"] = pid
            df["hospital_source"] = source
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

def summarize(df):
    print(f"Total rows (patient-hours): {len(df):,}")
    print(f"Total patients: {df['patient_id'].nunique():,}")
    print(f"Hospital sources: {df['hospital_source'].value_counts().to_dict()}")

    # Patient-level label: did this patient ever develop sepsis?
    patient_label = df.groupby("patient_id")["SepsisLabel"].max()
    print(f"\nPatients who develop sepsis at some point: "
          f"{patient_label.sum():,} / {len(patient_label):,} "
          f"({patient_label.mean()*100:.2f}%)")

    print(f"\nRow-level label balance (hourly): "
          f"{df['SepsisLabel'].mean()*100:.2f}% of all hours are labeled septic")

    # Missingness - the single most important EDA fact for this dataset
    miss = df.drop(columns=["patient_id", "hospital_source"]).isna().mean().sort_values(ascending=False)
    print("\nTop 15 most-missing columns (fraction missing):")
    print(miss.head(15))

    # ICU stay length distribution
    los = df.groupby("patient_id")["ICULOS"].max()
    print(f"\nICU stay length (hours): median={los.median():.0f}, "
          f"mean={los.mean():.1f}, min={los.min()}, max={los.max()}")

    return patient_label, miss, los

def time_to_onset_stats(df):
    """
    Critical for THIS project: how many hours of pre-onset data exist
    per septic patient? This determines how much "lead time" is even
    possible to predict, and shapes our windowing strategy.
    """
    septic_ids = df.loc[df["SepsisLabel"] == 1, "patient_id"].unique()
    onset_hours = []
    for pid in septic_ids:
        pdf = df[df["patient_id"] == pid].sort_values("ICULOS")
        onset_row = pdf.loc[pdf["SepsisLabel"] == 1, "ICULOS"].iloc[0]
        onset_hours.append(onset_row)
    onset_hours = pd.Series(onset_hours)
    print(f"\nAmong septic patients, hour of sepsis onset (ICULOS at first label=1):")
    print(onset_hours.describe())
    return onset_hours

if __name__ == "__main__":
    df = load_all_patients(DATA_DIRS)
    patient_label, miss, los = summarize(df)
    onset_hours = time_to_onset_stats(df)
