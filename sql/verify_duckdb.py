import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE admissions AS SELECT * FROM read_csv_auto('data/healthcare_admissions.csv')")

readmit_query = '''
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
SELECT * FROM flagged
'''

fact_admissions = con.execute(readmit_query).df()
total = len(fact_admissions)
readmits = fact_admissions["is_30day_readmit"].sum()
print(f"total_admissions={total}, readmissions={readmits}, "
      f"readmit_rate_pct={round(100*readmits/total, 1)}")

feature_query = '''
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

full_features = con.execute(feature_query).df()
full_features.to_csv("data/fact_admissions.csv", index=False)
print(f"Wrote data/fact_admissions.csv with {len(full_features)} rows, "
      f"{full_features.shape[1]} columns")