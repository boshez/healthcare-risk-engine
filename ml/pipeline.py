import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_score, recall_score, f1_score)

con = duckdb.connect()
con.execute("CREATE TABLE admissions AS SELECT * FROM read_csv_auto('data/healthcare_admissions.csv')")

full_feature_query = '''
WITH ordered AS (
    SELECT
        *,
        LEAD(admission_date, 1) OVER (
            PARTITION BY patient_id ORDER BY admission_date ASC
        ) AS next_admission_date
    FROM admissions
),
flagged AS (
    SELECT
        *,
        DATE_DIFF('day', discharge_date, next_admission_date) AS days_to_next_admission,
        CASE
            WHEN DATE_DIFF('day', discharge_date, next_admission_date) BETWEEN 0 AND 30
            THEN 1 ELSE 0
        END AS is_30day_readmit
    FROM ordered
)
SELECT
    *,
    ROW_NUMBER() OVER (
        PARTITION BY patient_id ORDER BY admission_date ASC
    ) - 1 AS prior_admission_count
FROM flagged
'''

df = con.execute(full_feature_query).df()
df = pd.get_dummies(df, columns=["discharge_destination", "department"], drop_first=False)

feature_cols = [c for c in df.columns if c not in
                ["admission_id", "patient_id", "admission_date", "discharge_date",
                 "next_admission_date", "days_to_next_admission", "is_30day_readmit"]]

X = df[feature_cols]
y = df["is_30day_readmit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print("roc_auc:", round(roc_auc_score(y_test, probs), 3))
print("pr_auc:", round(average_precision_score(y_test, probs), 3))
print("precision:", round(precision_score(y_test, preds), 3))
print("recall:", round(recall_score(y_test, preds), 3))
print("f1:", round(f1_score(y_test, preds), 3))

# Score every admission (not just the test split), so every patient has a risk score -
# the dashboard and the later RAG project both need full coverage, not just test-set rows.
df["readmission_risk_score"] = model.predict_proba(X)[:, 1]
risk_threshold = df["readmission_risk_score"].quantile(0.90)
high_risk = df[df["readmission_risk_score"] >= risk_threshold]
print("High-risk threshold (90th percentile):", round(risk_threshold, 3))
print(f"{len(high_risk)} patients flagged high-risk")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 5 feature importances:")
print(importances.head(5))

df.to_csv("data/patient_risk_scores.csv", index=False)
print("\nDone. Wrote data/patient_risk_scores.csv")