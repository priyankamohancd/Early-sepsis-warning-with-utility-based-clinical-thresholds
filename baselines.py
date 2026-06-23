"""
Section: Selected Models - Baselines
1. Clinical-rule reference (qSOFA-inspired, no training, sanity-check only)
2. Logistic regression (required learned baseline)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def clinical_rule_score(feature_df):
    """
    qSOFA-inspired rule, ADAPTED to available columns (true qSOFA needs
    a mentation/GCS check, which this dataset doesn't include - this
    limitation should be stated explicitly in your write-up).
    Each row scores 0-2 based on:
      - Resp_last >= 22
      - SBP_last <= 100
    Score >= 1 is treated as "flagged" for comparison purposes - this
    threshold is a deliberately simple reference point, not a tuned model.
    """
    score = (feature_df["Resp_last"] >= 22).astype(int) + \
            (feature_df["SBP_last"] <= 100).astype(int)
    flagged = (score >= 1).astype(int)
    return score, flagged

def train_logistic_baseline(train_df, val_df, feature_cols):
    """
    Logistic regression with standardization (required for LR to behave
    well) and class_weight='balanced' to address the severe label
    imbalance documented in EDA - without this, LR would likely predict
    the majority class almost everywhere.
    """
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)

    val_probs = pipe.predict_proba(X_val)[:, 1]
    return pipe, val_probs

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")

    df = eda.load_all_patients(["training_setA"])
    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = feat_mod.build_windowed_features(df)
    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    # Clinical rule on validation set
    score, flagged = clinical_rule_score(val_df)
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("=== Clinical rule reference (val set) ===")
    print(f"Flag rate: {flagged.mean()*100:.2f}%")
    print(f"AUROC (using raw 0/1/2 score as a ranking): "
          f"{roc_auc_score(val_df['label'], score):.3f}")

    # Logistic regression
    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source", "label")]
    pipe, val_probs = train_logistic_baseline(train_df, val_df, feature_cols)
    print("\n=== Logistic Regression baseline (val set) ===")
    print(f"AUROC: {roc_auc_score(val_df['label'], val_probs):.3f}")
    print(f"AUPRC: {average_precision_score(val_df['label'], val_probs):.3f}")
