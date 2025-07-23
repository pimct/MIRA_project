import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from config.config import DATA_PATHS

# === Mapping Excel-like column letters to 0-based indices ===
col_map = {
    'Feed_Ash': 9,         # Column J
    'Feed_VM': 7,          # Column H
    'Temp (°C)': 10,       # Column K
    'Char_HHV (MJ/kg)': 23 # Column X
}

# === Load raw data (assumes header is in first row) ===
# Define the row ID ranges (1-based indexing)
row_ranges = list(range(1, 11)) + list(range(31, 45)) # feed OHWD, AGR and MSW

df_raw = pd.read_csv(DATA_PATHS["htc_dataset"])
df_raw = df_raw.iloc[[i - 1 for i in row_ranges]]
print(df_raw)  # shows the first 5 rows

df_all = df_raw.iloc[:, list(col_map.values())].copy()
df_all.columns = list(col_map.keys())  # Rename for consistency with plotting code

# === Surface Plots ===
surface_plots_general = [
    ("Feed_Ash", "Feed_VM", "Char_HHV (MJ/kg)", "a)"),
    ("Feed_Ash", "Temp (°C)", "Char_HHV (MJ/kg)", "b)"),
    ("Temp (°C)", "Feed_VM", "Char_HHV (MJ/kg)", "c)"),
]

fig1 = plt.figure(figsize=(18, 6))

for i, (x, y, z, title) in enumerate(surface_plots_general, 1):
    ax = fig1.add_subplot(1, 3, i, projection='3d')
    df = df_all[[x, y, z]].dropna()

    xi = np.linspace(df[x].min(), df[x].max(), 100)
    yi = np.linspace(df[y].min(), df[y].max(), 100)
    X, Y = np.meshgrid(xi, yi)
    interp = RBFInterpolator(df[[x, y]], df[z], smoothing=0.5)
    Z = interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Z = np.clip(Z, a_min=0, a_max=None)

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title, pad=0)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
os.makedirs("results", exist_ok=True)
plt.savefig("results/surface_plot_char_HHV.png", dpi=300)
plt.savefig("results/surface_plot_char_HHV.svg", format='svg')
plt.show()
