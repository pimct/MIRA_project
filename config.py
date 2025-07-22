import os

# HTC constants
TIME_CONST = 2       # hours (or adjust as appropriate)
SOLID_CONST = 30     # % Solid loading


# === PSO Settings ===
PSO_CONFIG = {
    "max_iter": 100,
    "num_particles": 30,
    "x_bounds": {
        "x1": (0, 1),         # Process selection
        "x2": (250, 550),     # HTC temperature (°C)
        "x3": (0.1, 0.5),     # Char routing fraction
        "x4": (0, 0),         # Reserved
    },
    "objective_weights": {
        "revenue": -0.5,  # Negative for maximization
        "co2": 0.5        # Positive for minimization
    },
    "aspen_template_dir": "models/aspen/",
    "ann_model_dir": "models/ann/",
    "default_feedstock": "MSW"
}

# === Feedstock Properties ===
# Feedstock compositions are defined as lists of wt% for [C, H, N, S, O, VM, FC, Ash]
FEEDS = {
    "OHWD": {
        "composition": [53.4, 6.2, 3, 0.3, 37.1, 54.6, 9.6, 35.8]
    },
    "AGR": {
        "composition": [52.5, 6.07, 3.81, 0.36, 37.26, 70.2, 13.8, 16.0]
    },
    "MSW": {
        "composition": [54.16, 3.82, 3.37, 0.45, 38.2, 36.2, 8.3, 55.5]
    },

}


# === Product Prices ===
PRODUCT_PRICES = {
    "char": 4.285 / 1000,       # $/MJ
    "electricity": 85.7 / 1000  # $/kWh
}


# === Auto-detect project root (1 level up from config.py) ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# === Data Paths ===
DATA_PATHS = {
    "htc_dataset": os.path.join(ROOT_DIR, "data", "htc_dataset.csv")
}

# === Model Paths ===
MODEL_PATHS = {
    "ann_model": os.path.join(ROOT_DIR, "models", "ann", "htc_ann_model.pkl"),
    "ann_scaler_x": os.path.join(ROOT_DIR, "models", "ann", "htc_input_scaler.pkl"),
    "ann_scaler_y": os.path.join(ROOT_DIR, "models", "ann", "htc_output_scaler.pkl"),
    "aspen_direct": os.path.join(ROOT_DIR, "models", "aspen", "direct.apw"),
    "aspen_htc": os.path.join(ROOT_DIR, "models", "aspen", "htc.apw")
}

# === ANN Input/Output Column Indices (0-based)
ANN_COLUMN_INDICES = {
    "input": list(range(2, 13)),     # 11 input features
    "output": list(range(14, 23))    # 9 output targets
}
