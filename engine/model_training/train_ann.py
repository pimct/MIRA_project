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
#
# # === Extract ANN config ===
# ann_cfg = config["ann_training"]
# process_name = ann_cfg["process_name"]
# data_path = os.path.join(BASE_DIR, ann_cfg["dataset_path"])
# input_indices = ann_cfg["input_indices"]
# output_indices = ann_cfg["output_indices"]
# output_cols = ann_cfg["output_columns"]
# skip_index = ann_cfg.get("skip_index", None)


# === Let user select process for ANN training ===
process_options = config["process_system"]
print("🧠 Available processes for ANN training:")
for i, proc in enumerate(process_options, 1):
    print(f"{i}. {proc}")

while True:
    try:
        choice = int(input("Enter the number of the process to train ANN for: "))
        if 1 <= choice <= len(process_options):
            process_name = process_options[choice - 1]
            break
        else:
            print("⚠️ Invalid number. Please select a valid option.")
    except ValueError:
        print("⚠️ Please enter a number.")

# === Extract model config ===
if process_name not in config["model_config"]:
    raise ValueError(f"❌ No model_config found for selected process: {process_name}")

process_cfg = config["model_config"][process_name]

if "ANN_TRAINING" not in process_cfg:
    raise ValueError(f"❌ Selected process '{process_name}' has no ANN_TRAINING. Not supported for ANN training.")

data_path =  process_cfg["ANN_TRAINING"]["data_path"]
data_path = os.path.join(BASE_DIR, data_path)
input_indices = process_cfg["ANN_TRAINING"]["input"]
output_indices = process_cfg["ANN_TRAINING"]["output"]
output_cols = process_cfg["OUTPUT_COLUMNS"]
skip_index = process_cfg.get("SKIP_INDEX", None)


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
