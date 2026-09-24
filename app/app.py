import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Healthcare AI Risk Engine", layout="wide")


@st.cache_data
def load_data():
    admissions = pd.read_csv("../data/healthcare_admissions.csv")
    risk = pd.read_csv("../data/patient_risk_scores.csv")
    return admissions.merge(
        risk[["admission_id", "readmission_risk_score"]], on="admission_id", how="left"
    )


df = load_data()
risk_threshold = df["readmission_risk_score"].quantile(0.90)

st.title("Healthcare AI Risk Engine")
st.caption("Synthetic data only. Personal portfolio project, not a clinical tool.")

st.sidebar.header("Filters")
departments = sorted(df["department"].unique())
selected = st.sidebar.multiselect("Department", departments, default=departments)
filtered = df[df["department"].isin(selected)]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Admissions", f"{len(filtered):,}")
m2.metric("Avg Readmission Risk", f"{filtered['readmission_risk_score'].mean():.1%}")
m3.metric("High-Risk Cases", f"{filtered['readmission_risk_score'].ge(risk_threshold).sum():,}")
m4.metric("Avg Total Cost", f"${filtered['total_cost'].mean():,.0f}")

dept_risk = (filtered.groupby("department")["readmission_risk_score"]
             .mean().sort_values(ascending=False).reset_index())
fig = px.bar(dept_risk, x="department", y="readmission_risk_score",
             text_auto=".1%", color="readmission_risk_score", color_continuous_scale="Reds",
             title="Avg Readmission Risk by Department")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.histogram(filtered, x="readmission_risk_score", nbins=40,
                     title="Distribution of Readmission Risk Scores")
fig2.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
               annotation_text="High-risk threshold (90th pct)")
st.plotly_chart(fig2, use_container_width=True)

dest_risk = (filtered.groupby("discharge_destination")["readmission_risk_score"]
             .mean().sort_values(ascending=False).reset_index())
fig3 = px.bar(dest_risk, x="discharge_destination", y="readmission_risk_score",
              text_auto=".1%", color="readmission_risk_score", color_continuous_scale="Oranges",
              title="Avg Readmission Risk by Discharge Destination")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("High-Risk Patient Table")
high_risk = filtered[filtered["readmission_risk_score"] >= risk_threshold]
st.dataframe(
    high_risk[["patient_id", "department", "age", "length_of_stay",
               "total_cost", "discharge_destination", "readmission_risk_score"]]
    .sort_values("readmission_risk_score", ascending=False),
    use_container_width=True, hide_index=True,
)