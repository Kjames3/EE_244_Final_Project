# Robot Integration — EE244 Domain Adaptation

This directory contains the files copied from `x3_ws` that are needed for
Week 4 data collection and Weeks 5–7 deployment. The workflow is:

```
Jetson (robot)                    Laptop (this project)
──────────────────────────────    ────────────────────────────────────
1. ros2 launch x3_bringup         4. python3 preprocessing/05_process_rosbag.py
2. ./robot/record_bag.sh             --bag bags/<bag_name>
3. scp bag → laptop          →    5. python3 training/finetune.py
                                  6. (later) Deploy finetuned_best.pt back
```

---

## Directory Layout

```
robot/
├── record_bag.sh          ← Run ON JETSON to capture sensor data
├── drivers_x3.py          ← Hardware abstraction (Rosmaster, LiDAR, camera)
├── Rosmaster_Lib.py       ← Low-level Yahboom serial protocol
├── nav2_client.py         ← Nav2 action client for autonomous navigation
├── params/
│   ├── nav2_params_x3.yaml    ← MPPI controller + costmap config
│   ├── ydlidar_x3.yaml        ← YDLidar X3 TOF driver parameters
│   └── ekf_x3.yaml            ← EKF sensor fusion config
└── launch/
    ├── x3_bringup.launch.py   ← Hardware bringup (all drivers)
    ├── x3_slam.launch.py      ← SLAM Toolbox mapping
    └── x3_nav2.launch.py      ← Nav2 autonomous navigation
```

---

## Step 1 — Jetson Bringup

SSH into the Jetson, then:

```bash
source /opt/ros/humble/setup.bash
source ~/x3_ws/install/setup.bash
export ROS_DOMAIN_ID=42

# Start all hardware drivers (LiDAR, odometry, EKF, camera)
ros2 launch yahboomcar_nav x3_bringup.launch.py

# In another terminal — verify sensors are live
ros2 topic hz /scan                     # should be ~8 Hz
ros2 topic hz /camera/depth/image_raw  # should be ~30 Hz
ros2 topic hz /odom                    # should be ~50 Hz
```

---

## Step 2 — Record Domain Adaptation Bag

On the Jetson (after bringup is running):

```bash
cd ~/x3_ws
chmod +x src/record_bag.sh   # or copy this robot/record_bag.sh to the Jetson

# Record a session (Ctrl+C to stop)
./robot/record_bag.sh ~/bags/domain_adapt/

# Target: ~30 minutes total across multiple sessions
# Scenarios: people walking at 0.5–2m, crossing paths, approaching robot
```

Bags are saved as timestamped `.db3` files (compressed with zstd).

---

## Step 3 — Transfer Bag to Laptop

```bash
# On your laptop (replace JETSON_IP and USERNAME)
mkdir -p ~/EE_244_Final_Project/bags
scp -r kamren@<JETSON_IP>:~/bags/domain_adapt/ ~/EE_244_Final_Project/bags/
```

---

## Step 4 — Process Bag (Laptop)

```bash
cd ~/EE_244_Final_Project

# Process one bag
python3 preprocessing/05_process_rosbag.py \
    --bag bags/domain_adapt/domain_adapt_20260601_XXXXXX

# Process all bags at once
python3 preprocessing/05_process_rosbag.py \
    --bag bags/domain_adapt/domain_adapt_*

# Output: dataset/domain_adaptation/X_adapt.npy  (N, 40)
#         dataset/domain_adaptation/y_adapt.npy   (N, 2)
```

The processor uses:
- **Depth blob detection**: connected components on 16UC1 depth in 0.5–4 m range
- **LiDAR DBSCAN clustering**: eps=0.25 m, min_samples=3
- **Fusion**: LiDAR XY + depth fallback for detections not matched by LiDAR
- **Output format**: identical to THÖR-MAGNI (`[rel_x, rel_y, dx, dy]` × 10 frames)

---

## Step 5 — Fine-Tune Model (Laptop)

```bash
python3 training/finetune.py

# Options
python3 training/finetune.py \
    --checkpoint checkpoints/best_model_XXXXXXXX.pt \
    --replay-ratio 0.20 \   # 20% THÖR-MAGNI replay to prevent forgetting
    --lr 1e-4 \             # max LR per proposal
    --frozen-blocks 2       # freeze first 2 of 3 hidden blocks
```

Fine-tuned checkpoint saved to `checkpoints/finetuned_best_XXXXXXXX.pt`.

---

## Nav2 Notes

The robot uses **MPPI controller** (`nav2_mppi_controller::MPPIController`)
with Omni motion model — not DWA as originally proposed. The Week 7 A/B
comparison will be MPPI with/without velocity estimates injected into the
costmap's obstacle layer via a custom costmap plugin.

Key params in `params/nav2_params_x3.yaml`:
- `vx_max / vy_max`: 0.26 m/s (holonomic)
- `motion_model: "Omni"` — supports full x/y/theta motion
- Costmap: `ObstacleLayer` on `/scan` topic

---

## Hardware Reference

| Component      | Spec                                    |
|----------------|-----------------------------------------|
| Platform       | Yahboom ROSMASTER X3 (4-wheel mecanum)  |
| Compute        | Jetson Orin Nano 8GB (JetPack 6.2)      |
| LiDAR          | YDLidar X3 TOF, `/dev/ttyUSB0`, 8 Hz   |
| Depth Camera   | Orbbec Astra Pro SC, 640×480 @ 30 fps  |
| Serial         | Rosmaster board on `/dev/ttyCH341USB0`  |
| ROS Domain ID  | 42 (Jetson default)                     |
