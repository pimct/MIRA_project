import os

# HTC constants
TIME_CONST = 2       # hours (or adjust as appropriate)
SOLID_CONST = 30     # kg/hr


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
FEEDS = {
    "OHWD": {
        "composition": [34.3, 4.0, 1.9, 0.2, 23.8, 54.6342, 9.5658, 35.8]
    },
    "AGR": {
        "composition": [44.1, 5.1, 3.2, 0.3, 31.3, 70.2, 13.8, 16.0]
    },
    "MSW": {
        "composition": [24.1, 1.7, 1.5, 0.2, 17.0, 36.2, 8.3, 55.5]
    },
    "SS1": {
        "composition": [28.6, 3.1, 3.4, 1.5, 16.5, 51.0, 2.1, 46.9]
    },
    "VGF": {
        "composition": [29.5, 3.0, 2.0, 0.3, 21.4, 47.2, 9.0, 43.8]
    },
    "DSS": {
        "composition": [39.88, 6.2, 6.04, 5.62, 20.46, 69.0, 9.2, 21.8]
    },
    "SS2": {
        "composition": [26.542098, 4.195434, 3.833134, 0.0, 52.29, 68.56, 3.9, 27.54]
    }
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
