"""
Section: Error Analysis and Limitations

Pulls concrete patient-level examples from the TEST set (final held-out
evaluation, not validation) in four categories:

  1. SUCCESS: septic patient caught early, with good lead time
  2. MISS: septic patient never flagged before onset
  3. FALSE ALARM: non-septic patient flagged anyway
  4. STRUCTURAL LIMITATION: an immediate_only patient, to make concrete
     why this group cannot be "caught early" by construction

For each, we print the patient's vital-sign trajectory around the
relevant hour, so you can SEE what the model saw (or missed).
"""
import numpy as np
import pandas as pd

THRESHOLD = 0.80  # matches the threshold chosen in 06_threshold_leadtime.py

def find_examples(test_df_with_meta, raw_df, pred_probs, threshold=THRESHOLD,
                   vitals_to_show=("HR", "Resp", "MAP", "SBP")):
    """
    test_df_with_meta: windowed test rows with patient_id, ICULOS, label,
                        eligibility_group
    raw_df: the ORIGINAL (pre-windowing) hourly data, so we can show full
            vital-sign trajectories including the warm-up hours that
            windowing dropped
    pred_probs: model's predicted probability, same row order as
                test_df_with_meta
    """
    df = test_df_with_meta.copy()
    df["pred_prob"] = pred_probs
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)

    examples = {}

    # --- 1. SUCCESS: best lead-time catch among early_warning_eligible ---
    success_candidates = []
    for pid, pdf in df[df["eligibility_group"] == "early_warning_eligible"].groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        if pdf["label"].max() == 0:
            continue
        onset_idx = pdf["label"].values.argmax()
        pre_onset_flags = pdf["pred_label"].values[:onset_idx]
        if pre_onset_flags.sum() > 0:
            first_flag_idx = np.argmax(pre_onset_flags == 1)
            lead = onset_idx - first_flag_idx
            success_candidates.append((pid, lead))
    if success_candidates:
        # pick a representative (median lead time) case rather than cherry-picking the single best
        success_candidates.sort(key=lambda x: x[1])
        median_case = success_candidates[len(success_candidates) // 2]
        examples["success"] = median_case

    # --- 2. MISS: a septic patient with substantial pre-onset history,
    #     never flagged - the most "should have been catchable" miss ---
    miss_candidates = []
    for pid, pdf in df[df["eligibility_group"] == "early_warning_eligible"].groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        if pdf["label"].max() == 0:
            continue
        onset_idx = pdf["label"].values.argmax()
        pre_onset_flags = pdf["pred_label"].values[:onset_idx]
        if pre_onset_flags.sum() == 0 and onset_idx >= 12:  # plenty of history, still missed
            miss_candidates.append((pid, onset_idx))
    if miss_candidates:
        miss_candidates.sort(key=lambda x: -x[1])  # most pre-onset history, still missed
        examples["miss"] = miss_candidates[0]

    # --- 3. FALSE ALARM: never-septic patient with the most false-positive hours ---
    fa_candidates = []
    for pid, pdf in df[df["eligibility_group"] == "never_septic"].groupby("patient_id"):
        n_false_alarms = pdf["pred_label"].sum()
        if n_false_alarms > 0:
            fa_candidates.append((pid, n_false_alarms))
    if fa_candidates:
        fa_candidates.sort(key=lambda x: -x[1])
        examples["false_alarm"] = fa_candidates[0]

    # --- 4. STRUCTURAL LIMITATION: any immediate_only patient ---
    imm_patients = df[df["eligibility_group"] == "immediate_only"]["patient_id"].unique()
    if len(imm_patients) > 0:
        examples["structural_limit"] = (imm_patients[0], None)

    return examples

def print_patient_trajectory(pid, raw_df, vitals=("HR", "Resp", "MAP", "SBP"), label_note=""):
    pdf = raw_df[raw_df["patient_id"] == pid].sort_values("ICULOS")
    onset_idx = pdf["SepsisLabel"].values.argmax() if pdf["SepsisLabel"].max() == 1 else None
    print(f"\nPatient {pid} {label_note}")
    print(f"ICU stay length: {len(pdf)} hours" +
          (f", sepsis onset at hour {onset_idx + 1}" if onset_idx is not None else " (never septic)"))
    cols = ["ICULOS"] + list(vitals) + ["SepsisLabel"]
    print(pdf[cols].to_string(index=False))

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

    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source",
                                  "label", "eligibility_group")]
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

    examples = find_examples(
        test_df[["patient_id", "ICULOS", "label", "eligibility_group"]],
        df,  # raw pre-windowing data, for showing full trajectories
        test_probs
    )

    if "success" in examples:
        pid, lead = examples["success"]
        print_patient_trajectory(pid, df, label_note=f"[SUCCESS CASE - caught {lead}h before onset]")

    if "miss" in examples:
        pid, onset_idx = examples["miss"]
        print_patient_trajectory(pid, df, label_note=f"[MISS CASE - had {onset_idx}h of pre-onset history, never flagged]")

    if "false_alarm" in examples:
        pid, n_fa = examples["false_alarm"]
        print_patient_trajectory(pid, df, label_note=f"[FALSE ALARM CASE - {n_fa} hours incorrectly flagged]")

    if "structural_limit" in examples:
        pid, _ = examples["structural_limit"]
        print_patient_trajectory(pid, df, label_note="[STRUCTURAL LIMITATION CASE - immediate_only group]")