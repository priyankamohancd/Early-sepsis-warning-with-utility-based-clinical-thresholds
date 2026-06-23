"""
Section: Methodology - windowing, feature engineering, patient-level split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

VITALS = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp"]
LABS = ["BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
        "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
        "Glucose","Lactate","Magnesium","Phosphate","Potassium",
        "Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC",
        "Fibrinogen","Platelets"]
DEMOGRAPHICS = ["Age","Gender","HospAdmTime"]
WINDOW = 6  # hours of history used to build features for each prediction point

def add_missingness_indicators(df, cols):
    for c in cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)
    return df

def forward_fill_within_patient(df, cols):
    df[cols] = df.groupby("patient_id")[cols].ffill()
    # Anything still missing at the very start of a stay (no prior value yet)
    # gets filled with the population median - documented and justified,
    # not silently imputed.
    for c in cols:
        df[c] = df[c].fillna(df[c].median())
    return df

def build_windowed_features(df, window=WINDOW):
    """
    For each (patient, hour T) build features summarizing the past `window`
    hours: latest value, mean, and trend (slope) for vitals; latest value
    for labs (since labs change slowly and are sparse).
    Label = SepsisLabel at hour T (already pre-shifted 6h before clinical
    onset by PhysioNet's own convention - see project write-up).
    """
    feature_rows = []
    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS").reset_index(drop=True)
        n = len(pdf)
        for t in range(window, n):  # need at least `window` hours of history
            hist = pdf.iloc[t - window : t + 1]  # inclusive of current hour
            row = {"patient_id": pid, "ICULOS": pdf.loc[t, "ICULOS"]}

            for v in VITALS:
                vals = hist[v].values
                row[f"{v}_last"] = vals[-1]
                row[f"{v}_mean"] = np.mean(vals)
                row[f"{v}_slope"] = np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 1 else 0.0

            for lab in LABS:
                row[f"{lab}_last"] = hist[lab].values[-1]
                row[f"{lab}_missing_rate"] = hist[f"{lab}_missing"].mean()

            for d in DEMOGRAPHICS:
                row[d] = pdf.loc[t, d]

            row["hospital_source"] = pdf.loc[t, "hospital_source"]
            row["label"] = pdf.loc[t, "SepsisLabel"]
            feature_rows.append(row)

    return pd.DataFrame(feature_rows)

def patient_level_split(feature_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Critical: split by patient_id, not by row, to avoid subject leakage.
    """
    patients = feature_df["patient_id"].unique()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(feature_df, groups=feature_df["patient_id"]))
    train_val_df = feature_df.iloc[train_val_idx]
    test_df = feature_df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df["patient_id"]))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # Sanity check: confirm zero patient overlap across splits (always verify, don't assume)
    assert set(train_df.patient_id) & set(val_df.patient_id) == set()
    assert set(train_df.patient_id) & set(test_df.patient_id) == set()
    assert set(val_df.patient_id) & set(test_df.patient_id) == set()

    return train_df, val_df, test_df

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    df = eda.load_all_patients(["training_setA"])
    df = add_missingness_indicators(df, LABS)
    df = forward_fill_within_patient(df, VITALS + LABS)
    feat_df = build_windowed_features(df)
    print(f"Built {len(feat_df):,} windowed examples from {feat_df.patient_id.nunique()} patients")
    print(f"Positive rate in windowed examples: {feat_df['label'].mean()*100:.2f}%")

    train_df, val_df, test_df = patient_level_split(feat_df)
    print(f"\nTrain: {len(train_df):,} rows, {train_df.patient_id.nunique()} patients")
    print(f"Val:   {len(val_df):,} rows, {val_df.patient_id.nunique()} patients")
    print(f"Test:  {len(test_df):,} rows, {test_df.patient_id.nunique()} patients")
    print("\nNo patient overlap across splits: CONFIRMED (assertions passed)")