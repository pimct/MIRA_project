import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D


def load_feed_composition(feed_csv_path):
    df_feed = pd.read_csv(feed_csv_path)
    df_feed.columns = df_feed.columns.str.strip()
    return df_feed.set_index("Feed")[["Ash", "VM"]].to_dict("index")


def load_simulation_data(json_dir, feed_composition):
    feed_files = ["pareto_OHWD.json", "pareto_AGR.json", "pareto_MSW.json"]

    records = []
    for file in feed_files:
        file_path = os.path.join(json_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                feed_name = entry["feed"]
                if feed_name not in feed_composition:
                    continue
                ash = feed_composition[feed_name]["Ash"]
                vm = feed_composition[feed_name]["VM"]
                records.append({
                    "Feed": feed_name,
                    "Feed_Ash": ash,
                    "Feed_VM": vm,
                    "Temp (°C)": entry["x_input"]["temp"],
                    "Char Routing": entry["x_input"]["char_routing"],
                    "Char_HHV (MJ/kg)": entry["outputs"]["raw"]["char_HHV"]
                })
    return pd.DataFrame(records)


def plot_surface(df, x, y, z, ax, title):
    df_subset = df[[x, y, z]].dropna()
    xi = np.linspace(df_subset[x].min(), df_subset[x].max(), 100)
    yi = np.linspace(df_subset[y].min(), df_subset[y].max(), 100)
    X, Y = np.meshgrid(xi, yi)

    interp = RBFInterpolator(df_subset[[x, y]], df_subset[z], smoothing=0.01)
    Z = interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Z = np.clip(Z, a_min=0, a_max=None)

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.95)
    ax.scatter(df_subset[x], df_subset[y], df_subset[z], color='red', s=10, alpha=0.8, label="Data Points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title)
    ax.legend()


def generate_all_surface_plots(feed_csv_path, json_dir, output_dir="results"):
    feed_composition = load_feed_composition(feed_csv_path)
    df = load_simulation_data(json_dir, feed_composition)

    plots = [
        ("Feed_Ash", "Temp (°C)", "Char_HHV (MJ/kg)", "a) Feed Ash vs Temp"),
        ("Feed_VM", "Temp (°C)", "Char_HHV (MJ/kg)", "b) Feed VM vs Temp"),
        ("Char Routing", "Temp (°C)", "Char_HHV (MJ/kg)", "c) Char Routing vs Temp"),
    ]

    fig = plt.figure(figsize=(18, 6))
    for i, (x, y, z, title) in enumerate(plots, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        plot_surface(df, x, y, z, ax, title)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "surface_plot_char_HHV_combined.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "surface_plot_char_HHV_combined.svg"), format='svg')
    plt.show()


# === Run if standalone ===
if __name__ == "__main__":
    project_root = os.path.abspath("..")
    feed_csv = os.path.join(project_root, "data", "datasets", "feed_data.csv")
    json_dir = os.path.join(project_root, "pareto_results")
    generate_all_surface_plots(feed_csv, json_dir)
