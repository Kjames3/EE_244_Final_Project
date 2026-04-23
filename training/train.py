import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime

from model   import VelocityMLP
from dataset import VelocityDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────
CONFIG = {
    'input_dim':    40,
    'hidden_dims':  [256, 128, 64],
    'dropout':      0.2,
    'batch_size':   512,
    'lr':           1e-3,
    'weight_decay': 1e-4,
    'epochs':       100,
    'patience':     10,       # early stopping patience
    'data_dir':     os.path.join(BASE_DIR, 'dataset', 'thor_magni_windows'),
    'ckpt_dir':     os.path.join(BASE_DIR, 'checkpoints'),
    'results_dir':  os.path.join(BASE_DIR, 'results'),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, scaler_y):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Inverse transform to get real m/s values for MAE
    preds_ms   = scaler_y.inverse_transform(all_preds)
    targets_ms = scaler_y.inverse_transform(all_targets)
    
    mae_vx    = np.mean(np.abs(preds_ms[:, 0] - targets_ms[:, 0]))
    mae_vy    = np.mean(np.abs(preds_ms[:, 1] - targets_ms[:, 1]))
    mae_speed = np.mean(np.sqrt(
        (preds_ms[:, 0] - targets_ms[:, 0])**2 +
        (preds_ms[:, 1] - targets_ms[:, 1])**2
    ))
    
    return total_loss / len(loader.dataset), mae_vx, mae_vy, mae_speed

def main():
    os.makedirs(CONFIG['ckpt_dir'],    exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    X_train = np.load(os.path.join(CONFIG['data_dir'], 'X_train.npy'))
    y_train = np.load(os.path.join(CONFIG['data_dir'], 'y_train.npy'))
    X_val   = np.load(os.path.join(CONFIG['data_dir'], 'X_val.npy'))
    y_val   = np.load(os.path.join(CONFIG['data_dir'], 'y_val.npy'))
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Build datasets — fit scalers on train only
    print("Fitting scalers on training data...")
    train_ds = VelocityDataset(X_train, y_train, fit_scalers=True)
    train_ds.save_scalers(CONFIG['ckpt_dir'])
    
    val_ds = VelocityDataset(
        X_val, y_val,
        scaler_X=train_ds.scaler_X,
        scaler_y=train_ds.scaler_y,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Build model
    model = VelocityMLP(
        input_dim   =CONFIG['input_dim'],
        hidden_dims =CONFIG['hidden_dims'],
        dropout     =CONFIG['dropout'],
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Huber loss — robust to remaining outliers
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           =CONFIG['lr'],
        weight_decay =CONFIG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop with early stopping
    best_val_loss  = float('inf')
    best_epoch     = 0
    patience_count = 0
    history        = []
    
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(CONFIG['ckpt_dir'], f"best_model_{run_id}.pt")
    
    print(f"\nStarting training — run ID: {run_id}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'MAE vx':>8} {'MAE vy':>8} {'MAE spd':>9} {'LR':>10}")
    print("-" * 70)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, mae_vx, mae_vy, mae_spd = evaluate(
            model, val_loader, criterion, train_ds.scaler_y
        )
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} "
              f"{mae_vx:>8.4f} {mae_vy:>8.4f} {mae_spd:>9.4f} {current_lr:>10.6f}")
        
        history.append({
            'epoch':      int(epoch),
            'train_loss': float(train_loss),
            'val_loss':   float(val_loss),
            'mae_vx':     float(mae_vx),
            'mae_vy':     float(mae_vy),
            'mae_speed':  float(mae_spd),
            'lr':         float(current_lr),
        })
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss':    val_loss,
                'mae_speed':   mae_spd,
                'config':      CONFIG,
            }, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= CONFIG['patience']:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best was epoch {best_epoch})")
                break
    
    # Save training history
    history_path = os.path.join(CONFIG['results_dir'], f"history_{run_id}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nBest model: epoch {best_epoch}, "
          f"val loss {best_val_loss:.6f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History:    {history_path}")

if __name__ == '__main__':
    main()