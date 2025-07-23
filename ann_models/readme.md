# ANN Models Repository

This folder contains trained surrogate models (ANNs) and associated scalers for each supported process in the MIRA platform.

## Structure

- `htc/`: Hydrothermal carbonization surrogate model
- `pyrolysis/`: Surrogate model for pyrolysis product prediction
- `combustion/`: Optional ANN-based energy/revenue estimator

Each subfolder contains:
- `*_model.pkl`: Trained `MLPRegressor` object
- `*_scaler_x.pkl`: Scaler for input normalization
- `*_scaler_y.pkl`: Scaler for output normalization
- `metadata.json` (optional): Includes training metrics, architecture info, and timestamp

## Notes

- Model versioning should be reflected in filenames or logged in metadata.
- Use `engine/model_training/train_<process>_ann.py` to retrain models.
