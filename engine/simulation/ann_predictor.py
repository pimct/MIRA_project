import os
import joblib
import numpy as np

class ANNModelSklearn:
    def __init__(self, process_name: str):
        """
        Initialize and load the ANN model and scalers for the given process.
        Expected directory structure:
            ann_models/
                └── <process_name>/
                      ├── model.pkl
                      ├── scaler_x.pkl
                      └── scaler_y.pkl
        """
        self.base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Go to project root
            "ann_models",
            process_name
        )

        try:
            self.model = joblib.load(os.path.join(self.base_dir, "model.pkl"))
            self.scaler_x = joblib.load(os.path.join(self.base_dir, "scaler_x.pkl"))
            self.scaler_y = joblib.load(os.path.join(self.base_dir, "scaler_y.pkl"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ Could not find ANN model/scalers for process '{process_name}': {e}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict output from ANN model.

        Parameters:
            x (np.ndarray): 1D or 2D input array of shape (n_features,) or (n_samples, n_features)

        Returns:
            np.ndarray: Output prediction(s), shape matches output scaler
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_scaled = self.scaler_x.transform(x)
        y_scaled = self.model.predict(x_scaled)
        y = self.scaler_y.inverse_transform(y_scaled)

        return y[0] if y.shape[0] == 1 else y
