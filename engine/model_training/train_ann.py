# train_ann.py

import os
import json
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error

from engine.model_training.ann_utils import (
    normalize_and_split,
    evaluate_model_architectures,
    select_best_model,
    plot_mse,
    plot_parity,
    save_model_and_scalers
)

# === Resolve base directory ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# === Load config ===
config_path = os.path.join(BASE_DIR, "config", "run_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# === Extract ANN config ===
ann_cfg = config["ann_training"]
process_name = ann_cfg["process_name"]
data_path = os.path.join(BASE_DIR, ann_cfg["dataset_path"])
input_indices = ann_cfg["input_indices"]
output_indices = ann_cfg["output_indices"]
output_cols = ann_cfg["output_columns"]
skip_index = ann_cfg.get("skip_index", None)

# === Define output paths ===
model_dir = os.path.join(BASE_DIR, "ann_models", process_name)
figures_dir = os.path.join(BASE_DIR, "data", "figures")

model_paths = {
    "ann_model": os.path.join(model_dir, f"model.pkl"),
    "ann_scaler_x": os.path.join(model_dir, f"scaler_x.pkl"),
    "ann_scaler_y": os.path.join(model_dir, f"scaler_y.pkl"),
}

# ✅ Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# === Load and prepare data ===
df = pd.read_csv(data_path, skiprows=1, header=None)
X = df.iloc[:, input_indices].values
y = df.iloc[:, output_indices].values

X_train, X_val, X_test, y_train, y_val, y_test, scaler_x, scaler_y = normalize_and_split(X, y)

# === Train and evaluate ANN ===
hidden_nodes = list(range(1, 101))
models, mse_val = evaluate_model_architectures(X_train, y_train, X_val, y_val, hidden_nodes)
best_idx, best_model = select_best_model(models, mse_val)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.info("✅ Best architecture: %d hidden nodes", hidden_nodes[best_idx])


# === Plot MSE ===
mse_train = [mean_squared_error(y_train, m.predict(X_train)) for m in models]
mse_test = [mean_squared_error(y_test, m.predict(X_test)) for m in models]
plot_mse(hidden_nodes, mse_train, mse_val, mse_test,
         save_path=os.path.join(figures_dir, f"{process_name}_mse.png"))

# === Plot parity plots ===
y_pred = {
    "train": scaler_y.inverse_transform(best_model.predict(X_train)),
    "val": scaler_y.inverse_transform(best_model.predict(X_val)),
    "test": scaler_y.inverse_transform(best_model.predict(X_test)),
}
y_true = {
    "train": scaler_y.inverse_transform(y_train),
    "val": scaler_y.inverse_transform(y_val),
    "test": scaler_y.inverse_transform(y_test),
}
plot_parity(y_true, y_pred, output_cols, skip_index,
            save_path=os.path.join(figures_dir, f"{process_name}_parity.png"))

# === Save model and scalers ===
save_model_and_scalers(best_model, scaler_x, scaler_y, model_paths)
