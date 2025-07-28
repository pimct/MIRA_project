# import os
# import math
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
#
#
# def normalize_and_split(X, y, val_frac=0.15, test_frac=0.15, random_state=42):
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#     X_scaled = scaler_x.fit_transform(X)
#     y_scaled = scaler_y.fit_transform(y)
#
#     val_frac_within_temp = val_frac / (1 - test_frac)
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X_scaled, y_scaled, test_size=test_frac, random_state=random_state)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=val_frac_within_temp, random_state=random_state)
#
#     return X_train, X_val, X_test, y_train, y_val, y_test, scaler_x, scaler_y
#
#
# def evaluate_model_architectures(X_train, y_train, X_val, y_val, hidden_nodes_range, max_iter=5000):
#     from sklearn.neural_network import MLPRegressor
#
#     models, mse_val_list = [], []
#     for h in hidden_nodes_range:
#         layers = (h,) if h < 2 else (h // 2, h - h // 2)
#         model = MLPRegressor(hidden_layer_sizes=layers, activation='relu', solver='adam',
#                              max_iter=max_iter, random_state=42)
#         model.fit(X_train, y_train)
#         y_val_pred = model.predict(X_val)
#         mse_val = mean_squared_error(y_val, y_val_pred)
#         mse_val_list.append(mse_val)
#         models.append(model)
#     return models, mse_val_list
#
#
# def select_best_model(models, mse_val_list, alpha=0.85, beta=0.15):
#     mse_vals = np.array(mse_val_list)
#     nodes = np.arange(1, len(models) + 1)
#     mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min())
#     node_norm = (nodes - nodes.min()) / (nodes.max() - nodes.min())
#     combined_score = alpha * mse_norm + beta * node_norm
#     best_idx = np.argmin(combined_score)
#     return best_idx, models[best_idx]
#
#
# ann_utils.py (Keras version)

import os
import math
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def normalize_and_split(X, y, val_frac=0.15, test_frac=0.15, random_state=42):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    val_frac_within_temp = val_frac / (1 - test_frac)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_frac, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_within_temp, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_x, scaler_y


def build_model(input_dim, output_dim, hidden_nodes):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    model.add(Dense(hidden_nodes, activation='relu'))
    model.add(Dense(output_dim, activation='softplus'))  # Non-negative output
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[])
    return model


def evaluate_model_architectures(X_train, y_train, X_val, y_val, input_dim, output_dim, hidden_nodes_range, epochs=200, batch_size=16):
    models, mse_val_list = [], []

    for h in hidden_nodes_range:
        model = build_model(input_dim, output_dim, h)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        mse_val_list.append(mse_val)
        models.append(model)

    return models, mse_val_list


def select_best_model(models, mse_val_list, alpha=0.85, beta=0.15):
    mse_vals = np.array(mse_val_list)
    nodes = np.arange(1, len(models) + 1)
    mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min())
    node_norm = (nodes - nodes.min()) / (nodes.max() - nodes.min())
    combined_score = alpha * mse_norm + beta * node_norm
    best_idx = np.argmin(combined_score)
    return best_idx, models[best_idx]


def save_model_and_scalers(model, scaler_x, scaler_y, paths):
    os.makedirs(os.path.dirname(paths["ann_model"]), exist_ok=True)
    model.save(paths["ann_model"])
    joblib.dump(scaler_x, paths["ann_scaler_x"])
    joblib.dump(scaler_y, paths["ann_scaler_y"])

# def plot_mse(hidden_nodes, mse_train, mse_val, mse_test, save_path):
#     plt.figure(figsize=(12, 6))
#     plt.plot(hidden_nodes, mse_train, label='Train MSE', marker='o', color="#2077B5")
#     plt.plot(hidden_nodes, mse_val, label='Validation MSE', marker='s', color="#39A4C8")
#     plt.plot(hidden_nodes, mse_test, label='Test MSE', marker='^', color="#75C3D4")
#     plt.xlabel("Number of Hidden Nodes")
#     plt.ylabel("Mean Squared Error")
#     plt.title("ANN Architecture Optimization (Train vs Val vs Test)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300)
#
#
# def plot_parity(y_true_dict, y_pred_dict, output_cols, skip_index, save_path):
#     # === Determine which indices to plot ===
#     if skip_index is not None:
#         plot_indices = [i for i in range(len(output_cols)) if i != skip_index]
#     else:
#         plot_indices = list(range(len(output_cols)))
#
#     num_plots = len(plot_indices)
#     ncols = 2
#     nrows = math.ceil(num_plots / ncols)
#
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
#
#     axes = axes.flatten()
#     colors = {"train": "#2077B5", "val": "#39A4C8", "test": "#75C3D4"}
#
#     for ax_i, i in enumerate(plot_indices):
#         ax = axes[ax_i]
#         for key in ["train", "val", "test"]:
#             r2 = r2_score(y_true_dict[key][:, i], y_pred_dict[key][:, i])
#             ax.scatter(y_true_dict[key][:, i], y_pred_dict[key][:, i],
#                        label=f"{key.capitalize()} (R²={r2:.2f})", alpha=0.7, color=colors[key])
#
#         min_val = min(y_true_dict[key][:, i].min() for key in y_true_dict)
#         max_val = max(y_true_dict[key][:, i].max() for key in y_true_dict)
#         ax.plot([min_val, max_val], [min_val, max_val], 'k--')
#         ax.set_xlabel("Actual")
#         ax.set_ylabel("Predicted")
#         ax.set_title(output_cols[i])
#         ax.legend()
#
#     for j in range(len(plot_indices), len(axes)):
#         fig.delaxes(axes[j])
#
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300)
#
#
# def save_model_and_scalers(model, scaler_x, scaler_y, paths):
#     os.makedirs(os.path.dirname(paths["ann_model"]), exist_ok=True)
#     joblib.dump(model, paths["ann_model"])
#     joblib.dump(scaler_x, paths["ann_scaler_x"])
#     joblib.dump(scaler_y, paths["ann_scaler_y"])
