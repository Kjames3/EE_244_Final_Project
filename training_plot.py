import json
import matplotlib.pyplot as plt
import glob
import os

# Load history
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
history_files = glob.glob(os.path.join(BASE_DIR, "results", "history_*.json"))
assert history_files, "No history file found — fix JSON bug and retrain first"

with open(sorted(history_files)[-1]) as f:
    history = json.load(f)

epochs     = [h['epoch']     for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss   = [h['val_loss']   for h in history]
mae_speed  = [h['mae_speed']  for h in history]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
axes[0].plot(epochs, val_loss,   label='Val Loss',   linewidth=2)
axes[0].axvline(x=35, color='red', linestyle='--', 
                alpha=0.7, label='Best epoch (35)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Huber Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE speed over training
axes[1].plot(epochs, mae_speed, color='green', linewidth=2, label='MAE Speed')
axes[1].axhline(y=0.2160, color='green', linestyle='--', 
                alpha=0.7, label=f'Best MAE: 0.216 m/s')
axes[1].axhline(y=0.6432, color='red', linestyle='--', 
                alpha=0.7, label='Kalman baseline: 0.643 m/s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE Speed (m/s)')
axes[1].set_title('Velocity Estimation Accuracy vs Kalman Baseline')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('EE244 Project — Velocity MLP Training Results', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results", "training_curves.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to results/training_curves.png")