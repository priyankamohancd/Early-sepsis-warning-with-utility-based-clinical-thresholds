"""
Section: Results - Subgroup Analysis by Hospital Source (A vs B)

The dataset combines two hospital systems. Reporting one global metric
risks hiding meaningful performance differences between them - a real
robustness/distribution-shift concern, not just a formality. This script
reports AUROC/AUPRC/Utility-proxy separately for hospital A and hospital B,
on the SAME trained model (we do not retrain per hospital - we want to know
how one model, trained on the combined population, generalizes to each
source individually).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def hospital_subgroup_metrics(df_with_preds, prob_col, label_col="label",
                                source_col="hospital_source"):
    """
    df_with_preds must contain: label_col, prob_col, source_col
    Returns one row per hospital source + one "Overall" row.
    """
    rows = []

    def _compute(sub_df, name):
        y_true = sub_df[label_col]
        y_prob = sub_df[prob_col]
        n_patients = sub_df["patient_id"].nunique()
        if y_true.nunique() < 2:
            return {"group": name, "n_rows": len(sub_df), "n_patients": n_patients,
                     "n_positive": int(y_true.sum()),
                     "AUROC": np.nan, "AUPRC": np.nan,
                     "note": "only one class present"}
        return {"group": name, "n_rows": len(sub_df), "n_patients": n_patients,
                 "n_positive": int(y_true.sum()),
                 "AUROC": roc_auc_score(y_true, y_prob),
                 "AUPRC": average_precision_score(y_true, y_prob),
                 "note": ""}

    rows.append(_compute(df_with_preds, "Overall"))
    rows.append(_compute(df_with_preds[df_with_preds[source_col] == "A"], "Hospital A"))
    rows.append(_compute(df_with_preds[df_with_preds[source_col] == "B"], "Hospital B"))

    return pd.DataFrame(rows)

def hospital_subgroup_lead_time(df_with_preds, prob_col, threshold,
                                  label_col="label", source_col="hospital_source"):
    """
    Lead-time analysis computed separately per hospital source, restricted
    to septic patients only (mirrors stratified_lead_time in 09, but
    grouped by hospital_source instead of eligibility_group).
    """
    df = df_with_preds.copy()
    df["pred_label"] = (df[prob_col] >= threshold).astype(int)

    for source in ["A", "B"]:
        sub = df[df[source_col] == source]
        lead_times, caught, missed = [], 0, 0
        for pid, pdf in sub.groupby("patient_id"):
            pdf = pdf.sort_values("ICULOS")
            if pdf[label_col].max() == 0:
                continue  # never-septic patient, not relevant to lead-time
            onset_idx = pdf[label_col].values.argmax()
            pre_onset_flags = pdf["pred_label"].values[:onset_idx]
            if pre_onset_flags.sum() > 0:
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                lead_times.append(onset_idx - first_flag_idx)
                caught += 1
            else:
                missed += 1

        print(f"\n--- Hospital {source} ---")
        print(f"Septic patients caught before onset: {caught}, Missed: {missed}")
        if lead_times:
            print(pd.Series(lead_times).describe())

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")
    fast_mod = import_module("02b_features_fast")
    elig_mod = import_module("08_eligibility_tagging")
    xgb_mod = import_module("04_xgboost_main")

    df = eda.load_all_patients(eda.DATA_DIRS)
    elig_df = elig_mod.tag_patient_eligibility(df)

    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = fast_mod.build_windowed_features_fast(
        df, feat_mod.VITALS, feat_mod.LABS, feat_mod.DEMOGRAPHICS
    )
    feat_df = elig_mod.attach_eligibility(feat_df, elig_df)

    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    # Worth checking: confirm both hospitals are represented in train AND
    # test - if patient_level_split happened to put almost all of one
    # hospital into test, that would itself be a finding worth reporting.
    print("Hospital source distribution across splits:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name}: {split_df['hospital_source'].value_counts().to_dict()}")

    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source",
                                  "label", "eligibility_group")]
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)
    val_df = val_df.copy()
    val_df["pred_prob"] = val_probs

    print("\n=== Hospital subgroup metrics (validation set) ===")
    results = hospital_subgroup_metrics(val_df, "pred_prob")
    print(results.to_string(index=False))

    # Using threshold=0.80, the same threshold chosen via the sweep in
    # 06_threshold_leadtime.py, for consistency with our reported results.
    print("\n=== Hospital subgroup lead-time (validation set, threshold=0.80) ===")
    hospital_subgroup_lead_time(val_df, "pred_prob", threshold=0.80)