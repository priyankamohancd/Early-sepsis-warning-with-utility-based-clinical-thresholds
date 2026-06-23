"""
Section: Methodology - Early-Warning Eligibility Tagging

Following the diagnostic in 07_onset_window_diagnostic.py, we found that
~25% of septic patients have onset within the first WINDOW hours of their
ICU stay, meaning they cannot contribute a genuine pre-onset "early warning"
training/evaluation example under our windowing scheme.

Rather than silently blending these patients into one evaluation number,
we tag every patient into exactly one of three groups:

  - "never_septic"          : SepsisLabel never reaches 1
  - "early_warning_eligible": septic, with >= WINDOW hours of pre-onset data
  - "immediate_only"        : septic, with < WINDOW hours of pre-onset data
                               (onset happens too early in the stay for a
                               true early-warning example to exist)

This tag is computed ONCE per patient and merged into the windowed feature
dataframe, so every row inherits its patient's group. We can then report
metrics overall AND per group.
"""
import pandas as pd
import numpy as np

WINDOW = 6  # must match the window used in 02_features_split.py

def tag_patient_eligibility(df, window=WINDOW):
    """
    df: the RAW (not yet windowed) dataframe, one row per patient-hour,
        with columns patient_id and SepsisLabel (output of load_all_patients).

    Returns a DataFrame: patient_id -> eligibility_group
    """
    records = []
    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        labels = pdf["SepsisLabel"].values
        if labels.max() == 0:
            group = "never_septic"
        else:
            onset_idx = labels.argmax()  # row position (0-indexed) of first label=1
            group = "early_warning_eligible" if onset_idx >= window else "immediate_only"
        records.append({"patient_id": pid, "eligibility_group": group})
    return pd.DataFrame(records)

def attach_eligibility(feature_df, eligibility_df):
    """Merge the eligibility tag onto the windowed feature dataframe."""
    merged = feature_df.merge(eligibility_df, on="patient_id", how="left")
    assert merged["eligibility_group"].isna().sum() == 0, \
        "Some patients in feature_df have no eligibility tag - check patient_id matching"
    return merged

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")
    fast_mod = import_module("02b_features_fast")

    df = eda.load_all_patients(eda.DATA_DIRS)
    elig_df = tag_patient_eligibility(df)

    print("Patient counts by eligibility group:")
    print(elig_df["eligibility_group"].value_counts())
    print(f"\nTotal patients: {len(elig_df)}")

    # Build windowed features using the FAST vectorized version
    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = fast_mod.build_windowed_features_fast(
        df, feat_mod.VITALS, feat_mod.LABS, feat_mod.DEMOGRAPHICS
    )
    feat_df = attach_eligibility(feat_df, elig_df)

    print("\nRow counts by eligibility group (after windowing):")
    print(feat_df["eligibility_group"].value_counts())
    print("\nPositive rate (label=1) within each group's rows:")
    print(feat_df.groupby("eligibility_group")["label"].mean())
