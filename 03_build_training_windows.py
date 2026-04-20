import pandas as pd
import numpy as np
import os
from pathlib import Path

processed_dir = "dataset/thor_magni_processed"
output_dir    = "dataset/thor_magni_windows"
os.makedirs(output_dir, exist_ok=True)

# Window parameters
T         = 10    # input history frames (1 second at 10Hz)
STRIDE    = 1     # slide window by 1 frame each step
VAL_SEQS  = 5     # number of sequences held out for validation
TEST_SEQS = 5     # number of sequences held out for test

def make_windows(seq_df, T=10, stride=1):
    """
    For each body in a sequence, slide a window of length T
    across time and extract:
      X: [rel_x, rel_y, delta_x, delta_y] for T frames  (4*T features)
      y: [vx, vy] at the last frame of the window
    """
    X_list, y_list = [], []
    
    for body, group in seq_df.groupby('body'):
        group = group.sort_values('time').reset_index(drop=True)
        
        # Compute frame-to-frame displacement as additional feature
        group['dx'] = group['rel_x'].diff().fillna(0)
        group['dy'] = group['rel_y'].diff().fillna(0)
        
        n = len(group)
        if n < T + 1:
            continue
        
        for start in range(0, n - T, stride):
            window = group.iloc[start:start + T]
            target = group.iloc[start + T]
            
            # Skip if any NaN in window or target velocity
            if window[['rel_x','rel_y','dx','dy']].isnull().any().any():
                continue
            if pd.isnull(target['vx']) or pd.isnull(target['vy']):
                continue
            
            # Flatten window into feature vector
            features = window[['rel_x','rel_y','dx','dy']].values.flatten()
            label    = np.array([target['vx'], target['vy']])
            
            X_list.append(features)
            y_list.append(label)
    
    if not X_list:
        return None, None
    
    return np.array(X_list), np.array(y_list)

# Load all sequence file paths
feature_files = sorted([
    f for f in Path(processed_dir).glob("*_features.csv")
])

print(f"Found {len(feature_files)} processed sequences\n")

# Deterministic train/val/test split by sequence
# Use scenario diversity — pull test/val from different dates
np.random.seed(42)
indices      = np.random.permutation(len(feature_files))
test_idx     = indices[:TEST_SEQS]
val_idx      = indices[TEST_SEQS:TEST_SEQS + VAL_SEQS]
train_idx    = indices[TEST_SEQS + VAL_SEQS:]

splits = {
    'train': [feature_files[i] for i in train_idx],
    'val':   [feature_files[i] for i in val_idx],
    'test':  [feature_files[i] for i in test_idx],
}

print("Split assignment:")
for split, files in splits.items():
    print(f"  {split}: {len(files)} sequences")

# Build windows per split
for split, files in splits.items():
    print(f"\nBuilding {split} windows...")
    X_all, y_all = [], []
    
    for fpath in files:
        seq_df = pd.read_csv(fpath)
        X, y   = make_windows(seq_df, T=T, stride=STRIDE)
        
        if X is not None:
            X_all.append(X)
            y_all.append(y)
            print(f"  {fpath.name}: {len(X):,} windows")
    
    if X_all:
        X_combined = np.concatenate(X_all, axis=0)
        y_combined = np.concatenate(y_all, axis=0)
        
        np.save(os.path.join(output_dir, f"X_{split}.npy"), X_combined)
        np.save(os.path.join(output_dir, f"y_{split}.npy"), y_combined)
        
        print(f"  → {split}: X={X_combined.shape}, y={y_combined.shape}")

print(f"\nDone. Windows saved to {output_dir}/")
print("Files: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")