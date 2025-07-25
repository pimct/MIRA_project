import streamlit as st
import os
import pandas as pd
import plotly.express as px

# === Resolve log folder path ===
log_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "engine", "optimizer", "pso", "logs")
)

# === Load data files ===
df = pd.read_csv(os.path.join(log_dir, "pso_log.csv"))
convergence = pd.read_csv(os.path.join(log_dir, "convergence.csv"))
best_result_path = os.path.join(log_dir, "best_result.json")

with open(best_result_path, "r") as f:
    best = pd.read_json(f, typ="series")

# === Streamlit UI ===
st.title("🧠 PSO Optimization Visualization")

st.subheader("📉 Convergence Curve")
st.line_chart(convergence.set_index("iteration")[["score"]])

st.subheader(f"📊 Final Iteration Trade-off (Iteration {df['iteration'].max()})")
final_df = df[df["iteration"] == df["iteration"].max()]
fig = px.scatter(final_df, x="co2_emission", y="revenue", color="particle_id", hover_data=["score", "parameters"])
st.plotly_chart(fig)

st.subheader("📈 Best Revenue and CO₂ over Iterations")
fig2 = px.line(convergence, x="iteration", y=["revenue", "co2"])
st.plotly_chart(fig2)

st.subheader("🏆 Final Best Result")
st.json(best.to_dict())
