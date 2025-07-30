import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load config ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(BASE_DIR, "config", "run_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

process_cfg = config["model_config"]["htc"]
data_path = process_cfg["ANN_TRAINING"]["data_path"]
data_path = os.path.join(BASE_DIR, data_path)

# Load dataset
df = pd.read_csv(data_path)

# Drop the first column (e.g., ID or index)
df = df.iloc[:, 1:]

# Add new column: Char Yield (% dry ash-free basis)
yield_dry = df["Yield (% dry basis)"]
ash_percent = df["Ash,out (%)"]
df["Yield (% dry ash-free basis)"] = yield_dry - ash_percent

# Reorder columns: insert new column right after "Yield (% dry basis)"
cols = df.columns.tolist()
yield_idx = cols.index("Yield (% dry basis)")
# Move the new column to right after original yield
cols.insert(yield_idx + 1, cols.pop(cols.index("Yield (% dry ash-free basis)")))
df = df[cols]

# Compute correlation matrix
corr = df.corr(numeric_only=True)

# Create lower triangle mask
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm_r",
    annot=True,
    fmt=".2f",
    square=True,
    cbar_kws={"shrink": 0.75}
)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/heatmap_with_daf_yield.png", dpi=300)
plt.savefig("results/heatmap_with_daf_yield.svg", format='svg')
plt.show()
