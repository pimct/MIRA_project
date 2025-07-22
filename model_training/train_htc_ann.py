import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def train_and_save_ann_model(data_path, model_paths, column_indices):
    # === Load dataset ===
    df_raw = pd.read_csv(data_path, skiprows=1, header=None)

    # === Select input/output columns using passed-in indices
    X = df_raw.iloc[:, column_indices["input"]]
    y = df_raw.iloc[:, column_indices["output"]]
    output_cols = [
        "Yield (% dry basis)",
        "C,out (wt%)",
        "H,out (wt%)",
        "N,out (wt%)",
        "S,out (wt%)",
        "O,out (wt%)",
        "VM,out (%)",
        "FC,out (%)",
        "Ash,out (%)"
    ]

    # === Drop rows with NaN values
    df_clean = pd.concat([X, y], axis=1).dropna()
    X = df_clean.iloc[:, :len(column_indices["input"])].values
    y = df_clean.iloc[:, len(column_indices["input"]):].values

    # === Normalize features
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # === Split dataset
    val_frac = 0.15  # final desired validation fraction
    test_frac = 0.15
    val_frac_within_temp = val_frac / (1 - test_frac)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_frac, random_state=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_within_temp, random_state=2)

    # === Train and evaluate
    hidden_nodes = list(range(1, 101))
    mse_val_list = []
    models = []

    for h in hidden_nodes:
        if h < 2:
            hidden_layer_sizes = (h,)  # Single-layer case
        else:
            h1 = h // 2
            h2 = h - h1
            hidden_layer_sizes = (h1, h2)  # Two-layer case

        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam',
                             max_iter=5000, random_state=2)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        mse_val_list.append(mse_val)
        models.append(model)
        print(f"Hidden nodes: {h}, Val MSE: {mse_val:.6f}")

    # === Custom score: balance MSE and complexity
    alpha = 0.85  # weight score on MSE (0-1)
    beta = 0.15   # weight score on model size (0-1)

    mse_vals = np.array(mse_val_list)
    nodes = np.array(hidden_nodes)
    mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min())
    node_norm = (nodes - nodes.min()) / (nodes.max() - nodes.min())
    combined_score = alpha * mse_norm + beta * node_norm

    best_idx = np.argmin(combined_score)
    best_model = models[best_idx]

    print(f"\n✅ Balanced selection: {nodes[best_idx]} hidden nodes → Val MSE = {mse_vals[best_idx]:.6f}, Score = {combined_score[best_idx]:.4f}")

    # === Compute MSE for train and test as well (already needed for plotting)
    mse_train_list = []
    mse_test_list = []

    for model in models:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)

    # === Plot: Train vs Validation vs Test MSE
    # Custom colors
    train_color = "#2077B5"
    val_color = "#39A4C8"
    test_color = "#75C3D4"

    plt.figure(figsize=(12, 6))
    plt.plot(hidden_nodes, mse_train_list, label='Train MSE', marker='o', linestyle='-', linewidth=1.5, alpha=0.9, color=train_color)
    plt.plot(hidden_nodes, mse_val_list, label='Validation MSE', marker='s', linestyle='-', linewidth=1.5, alpha=0.9, color=val_color)
    plt.plot(hidden_nodes, mse_test_list, label='Test MSE', marker='^', linestyle='-', linewidth=1.5, alpha=0.9, color=test_color)

    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Mean Squared Error")
    plt.title("ANN Architecture Optimization (Train vs Val vs Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/hidden_nodes_mse_all.png", dpi=300)



    # === Parity plots: predicted vs actual for selected outputs
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # Inverse transform
    y_train_true_orig = scaler_y.inverse_transform(y_train)
    y_val_true_orig = scaler_y.inverse_transform(y_val)
    y_test_true_orig = scaler_y.inverse_transform(y_test)

    y_train_pred_orig = scaler_y.inverse_transform(y_train_pred)
    y_val_pred_orig = scaler_y.inverse_transform(y_val_pred)
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred)

    # Skip the 5th output (index 4)
    skip_index = 4
    num_outputs = y_test_true_orig.shape[1]
    plot_indices = [i for i in range(num_outputs) if i != skip_index]

    # Setup plot grid: 4 columns x 2 rows
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()



    for ax_i, i in enumerate(plot_indices):
        ax = axes[ax_i]

        # R² scores
        r2_train = r2_score(y_train_true_orig[:, i], y_train_pred_orig[:, i])
        r2_val = r2_score(y_val_true_orig[:, i], y_val_pred_orig[:, i])
        r2_test = r2_score(y_test_true_orig[:, i], y_test_pred_orig[:, i])

        # Scatter plots
        ax.scatter(y_train_true_orig[:, i], y_train_pred_orig[:, i], label=f"Train (R²={r2_train:.2f})", alpha=0.7, color=train_color)
        ax.scatter(y_val_true_orig[:, i], y_val_pred_orig[:, i], label=f"Val (R²={r2_val:.2f})", alpha=0.7, color=val_color)
        ax.scatter(y_test_true_orig[:, i], y_test_pred_orig[:, i], label=f"Test (R²={r2_test:.2f})", alpha=0.7, color=test_color)

        # Diagonal reference line
        all_true = np.concatenate([y_train_true_orig[:, i], y_val_true_orig[:, i], y_test_true_orig[:, i]])
        min_val, max_val = all_true.min(), all_true.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(output_cols[i])
        ax.legend()

    # Remove any unused axes
    for j in range(len(plot_indices), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/parity_plots.png", dpi=300)

    # === Save best model and scalers
    os.makedirs(os.path.dirname(model_paths["ann_model"]), exist_ok=True)
    joblib.dump(best_model, model_paths["ann_model"])
    joblib.dump(scaler_x, model_paths["ann_scaler_x"])
    joblib.dump(scaler_y, model_paths["ann_scaler_y"])