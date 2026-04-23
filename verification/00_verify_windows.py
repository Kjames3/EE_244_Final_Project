import numpy as np

X_train = np.load("dataset/thor_magni_windows/X_train.npy")
y_train = np.load("dataset/thor_magni_windows/y_train.npy")
X_val   = np.load("dataset/thor_magni_windows/X_val.npy")
y_val   = np.load("dataset/thor_magni_windows/y_val.npy")
X_test  = np.load("dataset/thor_magni_windows/X_test.npy")
y_test  = np.load("dataset/thor_magni_windows/y_test.npy")

print("=== Shapes ===")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

print("\n=== X_train stats (should have no NaN, bounded range) ===")
print(f"NaN count: {np.isnan(X_train).sum()}")
print(f"Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
print(f"Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")

print("\n=== y_train stats (vx, vy in m/s) ===")
print(f"NaN count: {np.isnan(y_train).sum()}")
print(f"vx — Min: {y_train[:,0].min():.3f}, Max: {y_train[:,0].max():.3f}, "
      f"Mean: {y_train[:,0].mean():.3f}, Std: {y_train[:,0].std():.3f}")
print(f"vy — Min: {y_train[:,1].min():.3f}, Max: {y_train[:,1].max():.3f}, "
      f"Mean: {y_train[:,1].mean():.3f}, Std: {y_train[:,1].std():.3f}")

print("\n=== Label distribution ===")
speeds = np.sqrt(y_train[:,0]**2 + y_train[:,1]**2)
print(f"Speed — Mean: {speeds.mean():.3f}, Std: {speeds.std():.3f}, "
      f"Max: {speeds.max():.3f}")
print(f"Stationary frames (speed < 0.1): "
      f"{(speeds < 0.1).sum():,} ({(speeds < 0.1).mean()*100:.1f}%)")