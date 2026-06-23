"""
THIS FILE IS NOT PART OF YOUR PROJECT.
It only generates fake .psv files shaped like the real PhysioNet 2019 files,
so we can test that our pipeline code actually runs before you point it at
real downloaded data. Do not include this in your submission.
"""
import numpy as np
import os

COLUMNS = [
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
    "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium",
    "Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC",
    "Fibrinogen","Platelets","Age","Gender","Unit1","Unit2",
    "HospAdmTime","ICULOS","SepsisLabel"
]

def make_patient(pid, rng, hours=None, sepsis=False):
    hours = hours or rng.integers(20, 120)
    n = hours
    df = {}
    df["HR"] = rng.normal(85, 15, n).clip(40, 180)
    df["O2Sat"] = rng.normal(97, 3, n).clip(70, 100)
    df["Temp"] = rng.normal(37, 0.7, n).clip(34, 41)
    df["SBP"] = rng.normal(120, 20, n).clip(60, 220)
    df["MAP"] = rng.normal(80, 12, n).clip(40, 140)
    df["DBP"] = rng.normal(70, 12, n).clip(30, 120)
    df["Resp"] = rng.normal(18, 4, n).clip(8, 40)
    # sparse labs: mostly missing, drawn occasionally
    for col in COLUMNS[8:34]:
        vals = np.full(n, np.nan)
        n_draws = rng.integers(1, max(2, n // 8))
        idx = rng.choice(n, size=min(n_draws, n), replace=False)
        vals[idx] = rng.normal(1, 0.3, len(idx))
        df[col] = vals
    df["EtCO2"] = np.full(n, np.nan)
    df["Age"] = np.full(n, rng.integers(20, 90))
    df["Gender"] = np.full(n, rng.integers(0, 2))
    df["Unit1"] = np.full(n, rng.integers(0, 2))
    df["Unit2"] = np.full(n, rng.integers(0, 2))
    df["HospAdmTime"] = np.full(n, -rng.integers(0, 48))
    df["ICULOS"] = np.arange(1, n + 1)

    label = np.zeros(n, dtype=int)
    if sepsis:
        onset = rng.integers(n // 3, n - 2)
        # vitals drift before/at onset to simulate a learnable signal
        drift = np.clip(np.arange(n) - (onset - 6), 0, None) * 0.8
        df["HR"] = df["HR"] + drift
        df["Resp"] = df["Resp"] + drift * 0.3
        df["MAP"] = df["MAP"] - drift * 0.4
        label[onset:] = 1
    df["SepsisLabel"] = label
    return df

def write_psv(path, patient_dict):
    n = len(patient_dict["HR"])
    with open(path, "w") as f:
        f.write("|".join(COLUMNS) + "\n")
        for i in range(n):
            row = [patient_dict[c][i] for c in COLUMNS]
            f.write("|".join("NaN" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + "\n")

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    os.makedirs("training_setA", exist_ok=True)
    n_patients = 300
    sepsis_rate = 0.08
    for i in range(n_patients):
        sepsis = rng.random() < sepsis_rate
        patient = make_patient(i, rng, sepsis=sepsis)
        write_psv(f"training_setA/p{i:06d}.psv", patient)
    print(f"Wrote {n_patients} fake patients to training_setA/")
