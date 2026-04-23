import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import os

class VelocityDataset(Dataset):
    """
    PyTorch Dataset for pedestrian velocity regression.
    Handles normalization with a scaler fit only on training data.
    """
    def __init__(self, X, y, scaler_X=None, scaler_y=None, fit_scalers=False):
        """
        Args:
            X:            numpy array (N, 40)
            y:            numpy array (N, 2)
            scaler_X:     pre-fit StandardScaler for X (pass None to create)
            scaler_y:     pre-fit StandardScaler for y (pass None to create)
            fit_scalers:  if True, fit scalers on this data (training set only)
        """
        if fit_scalers:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y)
        else:
            # Use pre-fit scalers from training set
            assert scaler_X is not None and scaler_y is not None, \
                "Must provide fitted scalers for val/test sets"
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
            X = self.scaler_X.transform(X)
            y = self.scaler_y.transform(y)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def save_scalers(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, os.path.join(output_dir, "scaler_y.pkl"))
        print(f"Scalers saved to {output_dir}/")
    
    @staticmethod
    def load_scalers(scaler_dir):
        scaler_X = joblib.load(os.path.join(scaler_dir, "scaler_X.pkl"))
        scaler_y = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))
        return scaler_X, scaler_y


if __name__ == '__main__':
    # Quick test
    X = np.random.randn(1000, 40).astype(np.float32)
    y = np.random.randn(1000, 2).astype(np.float32)
    
    ds = VelocityDataset(X, y, fit_scalers=True)
    print(f"Dataset length: {len(ds)}")
    
    sample_X, sample_y = ds[0]
    print(f"Sample X shape: {sample_X.shape}")
    print(f"Sample y shape: {sample_y.shape}")