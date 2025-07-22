import joblib
import numpy as np
from config import MODEL_PATHS

class ANNModelSklearn:
    def __init__(self):
        self.model = joblib.load(MODEL_PATHS["ann_model"])
        self.scaler_x = joblib.load(MODEL_PATHS["ann_scaler_x"])
        self.scaler_y = joblib.load(MODEL_PATHS["ann_scaler_y"])

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
