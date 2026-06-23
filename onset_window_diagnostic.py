"""
Diagnostic: how many septic patients actually have >= `window` hours of
pre-onset data, given the onset-hour distribution we just observed
(min=1, 25th percentile=7)?

Run this from your project folder, with training_setA / training_setB
already in place (same DATA_DIRS as 01_eda.py).
"""
from importlib import import_module
import pandas as pd

eda = import_module("01_eda")

WINDOW = 6

df = eda.load_all_patients(eda.DATA_DIRS)

septic_ids = df.loc[df["SepsisLabel"] == 1, "patient_id"].unique()
usable = 0
unusable = 0
onset_hours_unusable = []

for pid in septic_ids:
    pdf = df[df["patient_id"] == pid].sort_values("ICULOS")
    onset_idx = pdf["SepsisLabel"].values.argmax()  # row position of first label=1
    if onset_idx >= WINDOW:
        usable += 1
    else:
        unusable += 1
        onset_hours_unusable.append(onset_idx)

print(f"Septic patients with >= {WINDOW}h pre-onset history (usable for early-warning windowing): {usable}")
print(f"Septic patients with < {WINDOW}h pre-onset history (effectively UNUSABLE for early prediction): {unusable}")
print(f"-> {unusable / (usable + unusable) * 100:.1f}% of septic patients can't contribute a true 'early warning' positive example with a {WINDOW}h window")

if onset_hours_unusable:
    print("\nOnset row-position distribution among the 'unusable' group:")
    print(pd.Series(onset_hours_unusable).describe())

