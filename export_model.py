#!/usr/bin/env python3
"""
export_model.py — Export trained VelocityMLP to TorchScript for Jetson deployment.

Run from anywhere in the project:
    python3 export_model.py
    python3 export_model.py --checkpoint checkpoints/finetuned_best_XXXXXXXX.pt

Outputs to checkpoints/:
    velocity_mlp.torchscript   ← load with torch.jit.load() on the Jetson
    scaler_X.pkl               ← (already there) feature normaliser
    scaler_y.pkl               ← (already there) output denormaliser

Then scp all three files to the Jetson:
    scp checkpoints/velocity_mlp.torchscript \\
        checkpoints/scaler_X.pkl \\
        checkpoints/scaler_y.pkl \\
        kamren@<JETSON_IP>:~/x3_ws/src/
"""

import sys
import os
import glob
import argparse
from pathlib import Path

# ── Fix import path so this script works from the project root ─────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
TRAINING_DIR = PROJECT_ROOT / "training"
sys.path.insert(0, str(TRAINING_DIR))

import torch
from model import VelocityMLP   # now resolves correctly regardless of cwd


def main():
    parser = argparse.ArgumentParser(description="Export VelocityMLP to TorchScript")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pt checkpoint (default: latest best_model_*.pt in checkpoints/)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output .torchscript path (default: checkpoints/velocity_mlp.torchscript)"
    )
    args = parser.parse_args()

    ckpt_dir = PROJECT_ROOT / "checkpoints"

    # ── Find checkpoint ────────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
    else:
        candidates = sorted(ckpt_dir.glob("best_model_*.pt")) + \
                     sorted(ckpt_dir.glob("finetuned_best_*.pt"))
        if not candidates:
            print(f"[ERROR] No checkpoint found in {ckpt_dir}/")
            print("  Run: python3 training/train.py  (to generate one)")
            sys.exit(1)
        ckpt_path = candidates[-1]   # most recent by filename sort

    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # ── Build model ───────────────────────────────────────────────────────────
    cfg = ckpt.get("config", {})
    model = VelocityMLP(
        input_dim   = cfg.get("input_dim",    40),
        hidden_dims = cfg.get("hidden_dims",  [256, 128, 64]),
        dropout     = cfg.get("dropout",      0.2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_mae = ckpt.get("mae_speed", "?")
    epoch   = ckpt.get("epoch",     "?")
    print(f"  epoch={epoch}  val_MAE_speed={val_mae:.4f} m/s" if isinstance(val_mae, float)
          else f"  epoch={epoch}")

    # ── TorchScript trace ─────────────────────────────────────────────────────
    dummy  = torch.randn(1, 40)
    traced = torch.jit.trace(model, dummy)

    out_path = Path(args.out) if args.out else ckpt_dir / "velocity_mlp.torchscript"
    traced.save(str(out_path))
    print(f"\n✓ Exported: {out_path}")

    # ── Verify round-trip ─────────────────────────────────────────────────────
    loaded = torch.jit.load(str(out_path), map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        original_out = model(dummy).numpy()
        loaded_out   = loaded(dummy).numpy()
    max_diff = abs(original_out - loaded_out).max()
    print(f"  Round-trip max diff: {max_diff:.2e}  {'✓ OK' if max_diff < 1e-5 else '⚠ CHECK'}")

    # ── Remind user to copy scalers ───────────────────────────────────────────
    scaler_x = ckpt_dir / "scaler_X.pkl"
    scaler_y = ckpt_dir / "scaler_y.pkl"
    print(f"\nDeploy these 3 files to ~/x3_ws/src/ on the Jetson:")
    for f in [out_path, scaler_x, scaler_y]:
        exists = "✓" if f.exists() else "✗ MISSING"
        print(f"  {exists}  {f.name}")
    print(f"\n  scp {out_path} {scaler_x} {scaler_y} kamren@<JETSON_IP>:~/x3_ws/src/")


if __name__ == "__main__":
    main()