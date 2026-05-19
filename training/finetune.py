#!/usr/bin/env python3
"""
finetune.py — Week 5-6 Fine-Tuning on Domain Adaptation Data
=============================================================
Fine-tunes the pre-trained VelocityMLP on domain adaptation windows collected
from the Yahboom X3 (processed by preprocessing/05_process_rosbag.py).

Strategy:
  - Freeze first 2 hidden layers (preserves general velocity priors from THÖR-MAGNI)
  - Fine-tune remaining layers with lr ≤ 1e-4
  - Mixed replay: 20% THÖR-MAGNI train samples concatenated with domain data
    to prevent catastrophic forgetting
  - Uses the SAME scalers from original training (no re-fitting)

Usage:
    python3 training/finetune.py
    python3 training/finetune.py --checkpoint checkpoints/best_model_XXXXXXXX.pt
    python3 training/finetune.py --replay-ratio 0.3 --lr 5e-5 --epochs 50
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR / "training"))

from model import VelocityMLP
from dataset import VelocityDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return the most recently modified checkpoint .pt file."""
    pts = sorted(ckpt_dir.glob("best_model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return pts[-1]


def freeze_layers(model: VelocityMLP, n_frozen_blocks: int = 2):
    """
    Freeze the first n_frozen_blocks hidden blocks of the network.
    Each block is: Linear → BatchNorm → ReLU → Dropout (4 modules).
    """
    frozen_modules = n_frozen_blocks * 4  # 4 sub-modules per block
    for i, (name, param) in enumerate(model.network.named_parameters()):
        # Parameters in first n blocks
        layer_idx = int(name.split(".")[0]) if name.split(".")[0].isdigit() else 999
        if layer_idx < frozen_modules:
            param.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params:   {frozen:,}")
    print(f"  Trainable params:{trainable:,}")


def build_mixed_dataset(X_adapt, y_adapt, X_thor, y_thor,
                        scaler_X, scaler_y, replay_ratio: float, batch_size: int):
    """
    Combine domain adaptation data with a random replay subset of THÖR-MAGNI.
    Applies the pre-fit scalers (no re-fitting).
    """
    n_replay = int(len(X_adapt) * replay_ratio / (1 - replay_ratio))
    n_replay = min(n_replay, len(X_thor))

    idx = np.random.choice(len(X_thor), n_replay, replace=False)
    X_mix = np.concatenate([X_adapt, X_thor[idx]], axis=0)
    y_mix = np.concatenate([y_adapt, y_thor[idx]], axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_mix))
    X_mix, y_mix = X_mix[perm], y_mix[perm]

    print(f"  Domain adapt samples:  {len(X_adapt):,}")
    print(f"  THÖR-MAGNI replay:     {n_replay:,}")
    print(f"  Total mixed samples:   {len(X_mix):,}")

    X_scaled = scaler_X.transform(X_mix).astype(np.float32)
    y_scaled = scaler_y.transform(y_mix).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(X_scaled), torch.tensor(y_scaled)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(Xb)
    return total / len(loader.dataset)


def eval_epoch(model, loader, criterion, scaler_y):
    model.eval()
    total = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb)
            total += criterion(pred, yb).item() * len(Xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    preds   = scaler_y.inverse_transform(np.concatenate(preds))
    targets = scaler_y.inverse_transform(np.concatenate(targets))
    mae_spd = np.mean(np.hypot(preds[:,0]-targets[:,0], preds[:,1]-targets[:,1]))
    return total / len(loader.dataset), mae_spd


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VelocityMLP on domain adaptation data")
    parser.add_argument("--checkpoint",    default=None,
                        help="Path to base checkpoint .pt (default: latest in checkpoints/)")
    parser.add_argument("--adapt-dir",     default="dataset/domain_adaptation",
                        help="Directory with X_adapt.npy / y_adapt.npy")
    parser.add_argument("--thor-dir",      default="dataset/thor_magni_windows",
                        help="Directory with THÖR-MAGNI X_train.npy / y_train.npy")
    parser.add_argument("--ckpt-dir",      default="checkpoints",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--replay-ratio",  type=float, default=0.20,
                        help="Fraction of batch from THÖR-MAGNI replay (default: 0.20)")
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="Fine-tuning learning rate (default: 1e-4, max per proposal)")
    parser.add_argument("--epochs",        type=int,   default=60,
                        help="Maximum fine-tuning epochs (default: 60)")
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--patience",      type=int,   default=12)
    parser.add_argument("--frozen-blocks", type=int,   default=2,
                        help="Number of hidden blocks to freeze (default: 2 of 3)")
    args = parser.parse_args()

    ckpt_dir  = BASE_DIR / args.ckpt_dir
    adapt_dir = BASE_DIR / args.adapt_dir
    thor_dir  = BASE_DIR / args.thor_dir

    # ── Load checkpoint ────────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = BASE_DIR / ckpt_path
    else:
        ckpt_path = find_latest_checkpoint(ckpt_dir)

    print(f"\n[finetune.py] Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

    model = VelocityMLP(input_dim=40, hidden_dims=[256, 128, 64], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    print(f"  Base checkpoint: epoch {ckpt.get('epoch','?')}, "
          f"val_MAE_spd={ckpt.get('mae_speed', '?'):.4f}")

    # ── Freeze early layers ────────────────────────────────────────────────────
    print(f"\nFreezing first {args.frozen_blocks} hidden blocks …")
    freeze_layers(model, args.frozen_blocks)

    # ── Load scalers (MUST reuse originals — do NOT refit) ────────────────────
    scaler_X = joblib.load(str(ckpt_dir / "scaler_X.pkl"))
    scaler_y = joblib.load(str(ckpt_dir / "scaler_y.pkl"))
    print("  Scalers loaded from checkpoints/ (not refitted)")

    # ── Load domain adaptation data ────────────────────────────────────────────
    x_adapt_path = adapt_dir / "X_adapt.npy"
    y_adapt_path = adapt_dir / "y_adapt.npy"
    if not x_adapt_path.exists():
        print(f"\n[ERROR] Domain adaptation data not found at {adapt_dir}/")
        print("  Run: python3 preprocessing/05_process_rosbag.py --bag bags/<bag_name>")
        sys.exit(1)

    X_adapt = np.load(str(x_adapt_path))
    y_adapt = np.load(str(y_adapt_path))
    print(f"\nDomain adaptation data: X={X_adapt.shape}, y={y_adapt.shape}")

    # ── Load THÖR-MAGNI for replay ─────────────────────────────────────────────
    X_thor = np.load(str(thor_dir / "X_train.npy"))
    y_thor = np.load(str(thor_dir / "y_train.npy"))
    print(f"THÖR-MAGNI train data:  X={X_thor.shape}")

    # ── Build mixed train loader ───────────────────────────────────────────────
    print(f"\nBuilding mixed dataset (replay ratio={args.replay_ratio}) …")
    train_loader = build_mixed_dataset(
        X_adapt, y_adapt, X_thor, y_thor,
        scaler_X, scaler_y, args.replay_ratio, args.batch_size
    )

    # ── Validation loader (THÖR-MAGNI val — no domain adapt val needed) ────────
    X_val = np.load(str(thor_dir / "X_val.npy"))
    y_val = np.load(str(thor_dir / "y_val.npy"))
    X_val_sc = scaler_X.transform(X_val).astype(np.float32)
    y_val_sc  = scaler_y.transform(y_val).astype(np.float32)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_sc), torch.tensor(y_val_sc)),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # ── Optimizer & loss ───────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=1.0)

    # ── Fine-tuning loop ───────────────────────────────────────────────────────
    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path   = ckpt_dir / f"finetuned_best_{run_id}.pt"
    best_loss  = float("inf")
    best_mae   = float("inf")
    patience_c = 0
    history    = []

    print(f"\nFine-tuning for up to {args.epochs} epochs  (lr={args.lr}, device={DEVICE})")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val MAE':>9} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae = eval_epoch(model, val_loader, criterion, scaler_y)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} {val_mae:>9.4f} {lr:>10.2e}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_mae_speed": val_mae, "lr": lr})

        if val_loss < best_loss:
            best_loss = val_loss
            best_mae  = val_mae
            patience_c = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_loss":     val_loss,
                "mae_speed":    val_mae,
                "base_ckpt":    str(ckpt_path.name),
                "replay_ratio": args.replay_ratio,
                "frozen_blocks": args.frozen_blocks,
                "lr":           args.lr,
            }, str(out_path))
        else:
            patience_c += 1
            if patience_c >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Save history ───────────────────────────────────────────────────────────
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    hist_path = results_dir / f"finetune_history_{run_id}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Fine-tuning complete")
    print(f"  Best val MAE speed: {best_mae:.4f} m/s")
    print(f"  Checkpoint: {out_path}")
    print(f"  History:    {hist_path}")
    print(f"\nNext: python3 training/evaluate.py --checkpoint {out_path.name}")


if __name__ == "__main__":
    main()
