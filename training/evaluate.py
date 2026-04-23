import numpy as np
import torch
import os
import joblib
from torch.utils.data import DataLoader
from model import VelocityMLP
from dataset import VelocityDataset
import glob

# ── Config ────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'dataset', 'thor_magni_windows')
CKPT_DIR  = os.path.join(BASE_DIR, 'checkpoints')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def kalman_baseline(X_test, scaler_X, scaler_y):
    """
    Simple constant-velocity Kalman baseline.
    Uses the last displacement in the window as velocity estimate.
    dx and dy are features 2 and 3 of each frame (0-indexed within frame).
    At T=10, the last frame occupies indices 36-39: [rel_x, rel_y, dx, dy]
    So last dx = index 38, last dy = index 39 in the raw (unscaled) window.
    """
    # Inverse transform to get original scale
    X_raw = scaler_X.inverse_transform(X_test)
    
    # Last frame displacement — already in meters per 0.1s step
    # Convert to m/s by dividing by dt=0.1
    vx_pred = X_raw[:, 38] / 0.1
    vy_pred = X_raw[:, 39] / 0.1
    
    return np.stack([vx_pred, vy_pred], axis=1)

def compute_metrics(preds_ms, targets_ms, label=""):
    mae_vx    = np.mean(np.abs(preds_ms[:, 0] - targets_ms[:, 0]))
    mae_vy    = np.mean(np.abs(preds_ms[:, 1] - targets_ms[:, 1]))
    mae_speed = np.mean(np.sqrt(
        (preds_ms[:, 0] - targets_ms[:, 0])**2 +
        (preds_ms[:, 1] - targets_ms[:, 1])**2
    ))
    rmse = np.sqrt(np.mean(
        (preds_ms[:, 0] - targets_ms[:, 0])**2 +
        (preds_ms[:, 1] - targets_ms[:, 1])**2
    ))
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  MAE vx:    {mae_vx:.4f} m/s")
    print(f"  MAE vy:    {mae_vy:.4f} m/s")
    print(f"  MAE speed: {mae_speed:.4f} m/s")
    print(f"  RMSE:      {rmse:.4f} m/s")
    return {'mae_vx': mae_vx, 'mae_vy': mae_vy,
            'mae_speed': mae_speed, 'rmse': rmse}

def main():
    # Load test data
    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    print(f"Test set: {X_test.shape}")

    # Load scalers fit on training data
    scaler_X, scaler_y = VelocityDataset.load_scalers(CKPT_DIR)

    # Build test dataset
    test_ds = VelocityDataset(
        X_test, y_test,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Load best checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "best_model_*.pt")))
    assert ckpt_files, "No checkpoint found in checkpoints/"
    ckpt_path  = ckpt_files[-1]  # most recent
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = VelocityMLP(
        input_dim   =ckpt['config']['input_dim'],
        hidden_dims =ckpt['config']['hidden_dims'],
        dropout     =ckpt['config']['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']} "
          f"(val loss {ckpt['val_loss']:.6f})")

    # ── MLP predictions ───────────────────────────────────────────
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Inverse transform both to m/s
    preds_ms   = scaler_y.inverse_transform(all_preds)
    targets_ms = scaler_y.inverse_transform(all_targets)

    mlp_metrics = compute_metrics(preds_ms, targets_ms, "MLP (Ours)")

    # ── Kalman baseline ───────────────────────────────────────────
    kalman_preds = kalman_baseline(X_test, scaler_X, scaler_y)
    kalman_metrics = compute_metrics(
        kalman_preds, targets_ms, "Kalman Baseline (constant velocity)"
    )

    # ── Improvement summary ───────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"  Improvement over Kalman")
    print(f"{'='*40}")
    for k in ['mae_vx', 'mae_vy', 'mae_speed', 'rmse']:
        imp = (kalman_metrics[k] - mlp_metrics[k]) / kalman_metrics[k] * 100
        print(f"  {k:12s}: {imp:+.1f}%")

if __name__ == '__main__':
    main()