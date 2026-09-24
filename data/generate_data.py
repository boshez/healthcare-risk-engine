from faker import Faker
import pandas as pd
import numpy as np
import random
import datetime

fake = Faker()
random.seed(42)
np.random.seed(42)

NUM_PATIENTS = 1500
TARGET_ADMISSIONS = 10000

DEPARTMENTS = ["ICU", "Cardiology", "General Surgery", "Neurology", "Orthopedics", "Emergency"]
DEPT_WEIGHTS = [15, 12, 15, 15, 10, 33]
DEPT_COST_FACTOR = {"ICU": 2500, "Cardiology": 1800, "General Surgery": 1500,
                     "Neurology": 1400, "Orthopedics": 1300, "Emergency": 900}
DEST_CHOICES = ["Home", "Home Health Care", "Skilled Nursing Facility", "Against Medical Advice"]
DEST_WEIGHTS = [60, 20, 15, 5]
DEST_GAP_MULTIPLIER = {"Home": 1.0, "Home Health Care": 0.8,
                        "Skilled Nursing Facility": 0.45, "Against Medical Advice": 0.35}

WINDOW_START = datetime.date.today() - datetime.timedelta(days=730)
WINDOW_END = datetime.date.today()

# Fixed per-patient attributes, generated once - a patient's baseline age/frailty
# doesn't reset on every visit, and every downstream feature depends on that being true.
patients = []
for i in range(NUM_PATIENTS):
    patients.append({
        "patient_id": f"P{i:05d}",
        "age": int(np.random.normal(58, 18)),
        "frailty_score": float(np.random.beta(2, 5)),
    })
patients_df = pd.DataFrame(patients)
patients_df["age"] = patients_df["age"].clip(1, 100)


def random_start_date():
    # Bias toward the early part of the window so there's room for a realistic
    # chain of follow-up admissions before WINDOW_END.
    span = (WINDOW_END - WINDOW_START).days
    return WINDOW_START + datetime.timedelta(days=random.randint(0, int(span * 0.2)))


# Per-patient generator state: where each patient's admission timeline currently sits
state = {}
for _, p in patients_df.iterrows():
    state[p["patient_id"]] = {"next_date": random_start_date(), "exhausted": False}

rows = []
admission_counter = 0
active_ids = list(patients_df["patient_id"])

while len(rows) < TARGET_ADMISSIONS and active_ids:
    pid = random.choice(active_ids)
    s = state[pid]
    if s["exhausted"] or s["next_date"] > WINDOW_END:
        s["exhausted"] = True
        active_ids = [x for x in active_ids if not state[x]["exhausted"]]
        continue

    prow = patients_df[patients_df["patient_id"] == pid].iloc[0]
    frailty = prow["frailty_score"]
    department = random.choices(DEPARTMENTS, weights=DEPT_WEIGHTS)[0]

    admission_date = s["next_date"]
    length_of_stay = int(np.random.exponential(scale=3 + frailty * 5)) + 1
    discharge_date = admission_date + datetime.timedelta(days=length_of_stay)

    base_cost = DEPT_COST_FACTOR[department] * length_of_stay
    total_cost = round(base_cost * np.random.uniform(0.85, 1.25), 2)
    medication_count = int(np.random.poisson(3 + frailty * 6))
    discharge_destination = random.choices(DEST_CHOICES, weights=DEST_WEIGHTS)[0]

    rows.append({
        "admission_id": f"A{admission_counter:06d}",
        "patient_id": pid,
        "age": int(prow["age"]),
        "frailty_score": round(frailty, 3),
        "department": department,
        "admission_date": admission_date,
        "length_of_stay": length_of_stay,
        "discharge_date": discharge_date,
        "total_cost": total_cost,
        "medication_count": medication_count,
        "discharge_destination": discharge_destination,
    })
    admission_counter += 1

    # Gap to next admission: shorter for frailer patients and for SNF/AMA discharges -
    # the deliberate, CMS-backed signal the model will learn.
    base_scale = 170  # days
    frailty_factor = 1.0 - 0.65 * frailty
    dest_factor = DEST_GAP_MULTIPLIER[discharge_destination]
    scale = max(4.0, base_scale * frailty_factor * dest_factor)
    gap_days = int(np.random.exponential(scale=scale)) + 1
    s["next_date"] = discharge_date + datetime.timedelta(days=gap_days)

df = pd.DataFrame(rows).sort_values(["patient_id", "admission_date"])
df.to_csv("data/healthcare_admissions.csv", index=False)
print(f"Generated {len(df)} admissions across {df['patient_id'].nunique()} patients -> "
      f"data/healthcare_admissions.csv")