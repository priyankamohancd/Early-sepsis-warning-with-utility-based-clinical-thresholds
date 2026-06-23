# Sepsis Early Warning Project — Working Files (Session 1)

These scripts were built and explained step-by-step in our session. Each one
is runnable on its own (it includes a synthetic-data test harness under
`if __name__ == "__main__"`), but you should adapt the `DATA_DIRS` paths and
re-run everything against your REAL downloaded PhysioNet data before trusting
any numbers.

## Files, in the order we built them

1. `make_fake_data.py` — NOT part of your project. Only generates fake .psv
   files so code can be tested without the real dataset. Delete before
   submission.
2. `01_eda.py` — Exploratory Data Analysis: loads all .psv files, reports
   missingness, label balance, ICU stay length, and onset-timing stats.
3. `02_features_split.py` — Builds 6-hour sliding-window features with
   forward-fill + missingness indicators, then splits by patient_id
   (GroupShuffleSplit) with explicit leakage-check assertions.
4. `03_baselines.py` — qSOFA-inspired clinical-rule reference (no training)
   + logistic regression baseline with class balancing.
5. `04_xgboost_main.py` — Main model: XGBoost with scale_pos_weight for
   imbalance, early stopping on AUPRC, feature importances.
6. `05_utility_eval_prep.py` — Formats your model's predictions into the
   file format expected by the OFFICIAL PhysioNet utility scorer. You still
   need to:
     git clone https://github.com/physionetchallenges/evaluation-2019.git
   and run its evaluate_sepsis_score.py against the folders this produces.
7. `06_threshold_leadtime.py` — Fast approximate utility proxy for sweeping
   thresholds, plus lead-time analysis (hours-before-onset for caught
   patients, explicit miss tracking).

## Before next session, please:

- [ ] Download the real PhysioNet 2019 data (training_setA + training_setB)
- [ ] Run 01_eda.py on it and look at the REAL missingness/imbalance numbers
- [ ] Run 02-04 end to end and sanity-check the patient counts/splits
- [ ] Clone evaluation-2019 and try running the official scorer once
- [ ] Come with questions about anything that doesn't make sense — that's
      the point of working through this together rather than me just
      handing you a finished pipeline.

## Still to cover next session

- Hospital-source (A vs B) subgroup analysis
- Error analysis with concrete failure case examples
- Assembling the final presentation notebook structure
