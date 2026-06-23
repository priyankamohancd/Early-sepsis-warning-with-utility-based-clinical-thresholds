"""
Run this ONCE before building/presenting the final notebook.

It runs the full pipeline exactly as we've validated it across sessions,
then saves everything the notebook needs to disk:
  - trained XGBoost model (joblib)
  - test set predictions (for instant metric recomputation/display)
  - small sample of validation data (for the live demo cell)
  - all key results tables, as CSVs (stratified metrics, hospital subgroup,
    threshold sweep, error analysis patient examples)

After running this, the notebook loads from artifacts/ in seconds instead
of re-running 5-10 minutes of training each time you open it.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from importlib import import_module

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")
    fast_mod = import_module("02b_features_fast")
    elig_mod = import_module("08_eligibility_tagging")
    xgb_mod = import_module("04_xgboost_main")
    strat_mod = import_module("09_stratified_evaluation")
    hosp_mod = import_module("10_hospital_subgroup")
    err_mod = import_module("11_error_analysis")

    print("Loading data...")
    df = eda.load_all_patients(eda.DATA_DIRS)
    elig_df = elig_mod.tag_patient_eligibility(df)

    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = fast_mod.build_windowed_features_fast(
        df, feat_mod.VITALS, feat_mod.LABS, feat_mod.DEMOGRAPHICS
    )
    feat_df = elig_mod.attach_eligibility(feat_df, elig_df)

    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source",
                                  "label", "eligibility_group")]

    print("Training XGBoost...")
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

    THRESHOLD = 0.80  # final chosen threshold, from 06_threshold_leadtime.py sweep

    # --- Save the model itself ---
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "xgb_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    # --- Save val/test predictions + metadata (for instant metric recompute) ---
    val_out = val_df[["patient_id", "ICULOS", "label", "eligibility_group", "hospital_source"]].copy()
    val_out["pred_prob"] = val_probs
    val_out.to_csv(os.path.join(ARTIFACT_DIR, "val_predictions.csv"), index=False)

    test_out = test_df[["patient_id", "ICULOS", "label", "eligibility_group", "hospital_source"]].copy()
    test_out["pred_prob"] = test_probs
    test_out.to_csv(os.path.join(ARTIFACT_DIR, "test_predictions.csv"), index=False)

    # --- Save a SMALL sample of raw + feature data for the live demo cell ---
    # Pick 5 test patients spanning different eligibility groups, save their
    # raw vitals + computed features, so the notebook can demo a real
    # prediction live without reloading the full 40K-patient dataset.
    demo_patients = []
    for group in ["early_warning_eligible", "immediate_only", "never_septic"]:
        ids = test_df[test_df["eligibility_group"] == group]["patient_id"].unique()
        if len(ids) > 0:
            demo_patients.extend(ids[:2])  # 2 per group, up to 6 total

    demo_raw = df[df["patient_id"].isin(demo_patients)]
    demo_raw.to_csv(os.path.join(ARTIFACT_DIR, "demo_raw_patients.csv"), index=False)

    demo_features = test_df[test_df["patient_id"].isin(demo_patients)]
    demo_features.to_csv(os.path.join(ARTIFACT_DIR, "demo_feature_rows.csv"), index=False)

    # --- Save results tables ---
    strat_results = strat_mod.stratified_metrics(val_out, "pred_prob")
    strat_results.to_csv(os.path.join(ARTIFACT_DIR, "stratified_metrics.csv"), index=False)

    hosp_results = hosp_mod.hospital_subgroup_metrics(val_out, "pred_prob")
    hosp_results.to_csv(os.path.join(ARTIFACT_DIR, "hospital_subgroup_metrics.csv"), index=False)

    # --- Save error analysis example patient IDs (for notebook to look up) ---
    examples = err_mod.find_examples(
        test_df[["patient_id", "ICULOS", "label", "eligibility_group"]],
        df, test_probs
    )
    examples_serializable = {k: (str(v[0]), (int(v[1]) if v[1] is not None else None))
                              for k, v in examples.items()}
    with open(os.path.join(ARTIFACT_DIR, "error_examples.json"), "w") as f:
        json.dump(examples_serializable, f, indent=2)

    # --- Save key summary numbers for quick reference in the notebook ---
    summary = {
        "n_total_patients": int(df["patient_id"].nunique()),
        "threshold_used": THRESHOLD,
        "n_train_patients": int(train_df["patient_id"].nunique()),
        "n_val_patients": int(val_df["patient_id"].nunique()),
        "n_test_patients": int(test_df["patient_id"].nunique()),
    }
    with open(os.path.join(ARTIFACT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll artifacts saved to {ARTIFACT_DIR}/")
    print("Files written:")
    for f in sorted(os.listdir(ARTIFACT_DIR)):
        size = os.path.getsize(os.path.join(ARTIFACT_DIR, f))
        print(f"  {f} ({size:,} bytes)")

if __name__ == "__main__":
    main()