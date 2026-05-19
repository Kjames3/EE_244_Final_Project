#!/usr/bin/env python3
"""
05_process_rosbag.py — Offline Domain Adaptation Bag Processor
==============================================================
Reads a ROS2 .db3 rosbag recorded on the Yahboom X3, extracts person
candidates using depth blob detection + LiDAR DBSCAN clustering, tracks
them across frames, and builds sliding windows in the EXACT same format as
the THÖR-MAGNI preprocessing pipeline:

  X: (N, 40)  — 10 frames × [rel_x, rel_y, dx, dy]
  y: (N, 2)   — [vx, vy] from finite-difference on smoothed tracks

Outputs are saved to dataset/domain_adaptation/ for use with finetune.py.

Usage:
    python3 preprocessing/05_process_rosbag.py --bag bags/domain_adapt_YYYYMMDD_HHMMSS
    python3 preprocessing/05_process_rosbag.py --bag bags/domain_adapt_* --min-track-len 20
"""

import argparse
import sqlite3
import struct
import math
import os
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import label as scipy_label
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN

# ── Constants (must match 03_build_training_windows.py) ────────────────────────
T            = 10       # window length (frames)
STRIDE       = 1        # window stride
HZ           = 10.0     # target processing frequency (Hz)
MAX_SPEED    = 3.0      # m/s — discard tracks with faster velocity (noise)
MIN_TRACK    = T + 5    # minimum track length to produce any windows

# Depth detection parameters (Orbbec Astra Pro)
DEPTH_MIN_M  = 0.5      # minimum valid depth (m)
DEPTH_MAX_M  = 4.0      # maximum detection range (m)
BLOB_MIN_PX  = 200      # minimum blob area (pixels)
BLOB_MAX_PX  = 40000    # maximum blob area (pixels)
DEPTH_SCALE  = 0.001    # Astra: raw value in mm → meters
ASTRA_FX     = 570.0    # approximate focal length x (pixels)
ASTRA_FY     = 570.0    # approximate focal length y (pixels)
ASTRA_CX     = 320.0    # principal point x
ASTRA_CY     = 240.0    # principal point y

# LiDAR DBSCAN parameters
LIDAR_EPS    = 0.25     # m — cluster radius
LIDAR_MIN    = 3        # minimum points per cluster
LIDAR_MIN_R  = 0.3      # m — ignore clusters closer than this (robot body)
LIDAR_MAX_R  = 3.5      # m — ignore clusters farther than this

# Tracking
MATCH_DIST   = 0.8      # m — max distance to match detection to existing track


# ── ROS2 Bag Reader (pure sqlite3, no rosbag2_py needed) ───────────────────────

def iter_messages(bag_dir: Path, topic: str):
    """
    Yield (timestamp_ns, raw_bytes) for every message on `topic` in a
    ROS2 sqlite3 bag. Works without rosbag2_py installed.
    """
    db_files = sorted(bag_dir.glob("*.db3"))
    if not db_files:
        raise FileNotFoundError(f"No .db3 files found in {bag_dir}")

    for db_path in db_files:
        conn = sqlite3.connect(str(db_path))
        cur  = conn.cursor()

        # Find topic id
        cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
        row = cur.fetchone()
        if row is None:
            conn.close()
            continue
        topic_id = row[0]

        cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_id,)
        )
        for ts, data in cur:
            yield int(ts), bytes(data)
        conn.close()


# ── ROS2 Message Deserializers (CDR format, hand-rolled for speed) ─────────────

def decode_image(data: bytes):
    """
    Decode sensor_msgs/Image (16UC1 depth) from CDR bytes.
    Returns (stamp_ns, height, width, np.uint16 array).
    """
    off = 4  # skip CDR header
    sec,  = struct.unpack_from('<I', data, off); off += 4
    nsec, = struct.unpack_from('<I', data, off); off += 4
    stamp_ns = sec * 10**9 + nsec

    # frame_id string
    slen, = struct.unpack_from('<I', data, off); off += 4
    off += slen

    height, = struct.unpack_from('<I', data, off); off += 4
    width,  = struct.unpack_from('<I', data, off); off += 4

    # encoding string
    elen, = struct.unpack_from('<I', data, off); off += 4
    off += elen

    off += 1  # is_bigendian
    step, = struct.unpack_from('<I', data, off); off += 4

    dlen, = struct.unpack_from('<I', data, off); off += 4
    raw = np.frombuffer(data[off:off+dlen], dtype=np.uint16).reshape(height, width)
    return stamp_ns, raw


def decode_scan(data: bytes):
    """
    Decode sensor_msgs/LaserScan from CDR bytes.
    Returns (stamp_ns, angles_rad np.array, ranges_m np.array).
    """
    off = 4
    sec,  = struct.unpack_from('<I', data, off); off += 4
    nsec, = struct.unpack_from('<I', data, off); off += 4
    stamp_ns = sec * 10**9 + nsec

    # frame_id
    slen, = struct.unpack_from('<I', data, off); off += 4
    off += slen

    angle_min, = struct.unpack_from('<f', data, off); off += 4
    angle_max, = struct.unpack_from('<f', data, off); off += 4
    angle_inc, = struct.unpack_from('<f', data, off); off += 4
    off += 4 + 4 + 4 + 4  # time_incr, scan_time, range_min, range_max

    nranges, = struct.unpack_from('<I', data, off); off += 4
    ranges = np.frombuffer(data[off:off + nranges*4], dtype=np.float32).copy()
    off += nranges * 4

    angles = angle_min + np.arange(nranges) * angle_inc
    return stamp_ns, angles, ranges


def decode_odom(data: bytes):
    """
    Decode nav_msgs/Odometry from CDR bytes.
    Returns (stamp_ns, x, y, yaw).
    """
    off = 4
    sec,  = struct.unpack_from('<I', data, off); off += 4
    nsec, = struct.unpack_from('<I', data, off); off += 4
    stamp_ns = sec * 10**9 + nsec

    # frame_id + child_frame_id
    for _ in range(2):
        slen, = struct.unpack_from('<I', data, off); off += 4
        off += slen

    # pose.pose.position
    x, = struct.unpack_from('<d', data, off); off += 8
    y, = struct.unpack_from('<d', data, off); off += 8
    off += 8  # z

    # pose.pose.orientation (quaternion → yaw)
    qx, = struct.unpack_from('<d', data, off); off += 8
    qy, = struct.unpack_from('<d', data, off); off += 8
    qz, = struct.unpack_from('<d', data, off); off += 8
    qw, = struct.unpack_from('<d', data, off); off += 8
    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    return stamp_ns, x, y, yaw


# ── Detection Functions ─────────────────────────────────────────────────────────

def detect_depth_blobs(depth_img: np.ndarray):
    """
    Find person candidates from a 16UC1 depth image.
    Returns list of (cx_m, cy_m, depth_m) in camera frame.
    cx_m, cy_m are horizontal/vertical offsets from optical axis in meters.
    """
    depth_m = depth_img.astype(np.float32) * DEPTH_SCALE
    mask = ((depth_m >= DEPTH_MIN_M) & (depth_m <= DEPTH_MAX_M)).astype(np.uint8)

    labeled, n_blobs = scipy_label(mask)
    detections = []
    for i in range(1, n_blobs + 1):
        blob_mask = (labeled == i)
        area = blob_mask.sum()
        if not (BLOB_MIN_PX <= area <= BLOB_MAX_PX):
            continue
        ys, xs = np.where(blob_mask)
        cx_px = xs.mean()
        cy_px = ys.mean()
        d_m   = depth_m[blob_mask].mean()
        cx_m  = (cx_px - ASTRA_CX) * d_m / ASTRA_FX
        cy_m  = (cy_px - ASTRA_CY) * d_m / ASTRA_FY
        detections.append((cx_m, cy_m, d_m))
    return detections


def detect_lidar_clusters(angles, ranges):
    """
    Cluster valid LiDAR points with DBSCAN.
    Returns list of (x_m, y_m) cluster centroids in laser frame.
    """
    valid = (ranges > LIDAR_MIN_R) & (ranges < LIDAR_MAX_R) & np.isfinite(ranges)
    if valid.sum() < LIDAR_MIN:
        return []
    pts = np.column_stack([
        ranges[valid] * np.cos(angles[valid]),
        ranges[valid] * np.sin(angles[valid])
    ])
    labels = DBSCAN(eps=LIDAR_EPS, min_samples=LIDAR_MIN).fit_predict(pts)
    centroids = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = pts[labels == lbl]
        centroids.append(cluster.mean(axis=0))
    return centroids


def fuse_detections(depth_dets, lidar_dets):
    """
    Fuse depth and LiDAR detections.
    Depth gives (cx_m, ~, depth_m) → approximate robot-frame (depth_m, cx_m).
    LiDAR gives (x_m, y_m) in laser frame ≈ robot frame.
    Returns list of (x_m, y_m) best estimates.
    """
    fused = []
    # LiDAR points first (more reliable for XY)
    for (lx, ly) in lidar_dets:
        fused.append((lx, ly))
    # Depth-only blobs that have no LiDAR counterpart within 0.4m
    for (cx_m, _, d_m) in depth_dets:
        x_approx = d_m
        y_approx = -cx_m
        matched = any(
            math.hypot(x_approx - fx, y_approx - fy) < 0.4
            for (fx, fy) in fused
        )
        if not matched:
            fused.append((x_approx, y_approx))
    return fused


# ── Tracker ────────────────────────────────────────────────────────────────────

class Track:
    _next_id = 0

    def __init__(self, x, y, t_ns):
        self.id = Track._next_id; Track._next_id += 1
        self.xs = [x]; self.ys = [y]; self.ts = [t_ns]
        self.missed = 0

    def update(self, x, y, t_ns):
        self.xs.append(x); self.ys.append(y); self.ts.append(t_ns)
        self.missed = 0

    @property
    def last_pos(self):
        return self.xs[-1], self.ys[-1]

    def to_robot_frame(self, robot_x, robot_y, robot_yaw):
        """Convert absolute world positions (after accumulation) to robot-relative."""
        cos_y, sin_y = math.cos(-robot_yaw), math.sin(-robot_yaw)
        rel_xs, rel_ys = [], []
        for wx, wy in zip(self.xs, self.ys):
            dx, dy = wx - robot_x, wy - robot_y
            rel_xs.append(cos_y*dx - sin_y*dy)
            rel_ys.append(sin_y*dx + cos_y*dy)
        return rel_xs, rel_ys


def match_detections(tracks, detections, t_ns):
    """Greedy nearest-neighbour assignment. Returns updated tracks + new tracks."""
    unmatched_dets = list(range(len(detections)))
    for track in tracks:
        if not unmatched_dets:
            break
        tx, ty = track.last_pos
        best_i, best_d = None, MATCH_DIST
        for i in unmatched_dets:
            dx, dy = detections[i][0] - tx, detections[i][1] - ty
            d = math.hypot(dx, dy)
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            track.update(*detections[best_i], t_ns)
            unmatched_dets.remove(best_i)
        else:
            track.missed += 1

    # Spawn new tracks for unmatched detections
    new_tracks = [Track(*detections[i], t_ns) for i in unmatched_dets]
    # Kill tracks that missed too many frames
    alive = [t for t in tracks if t.missed <= 5] + new_tracks
    return alive


# ── Window Builder ─────────────────────────────────────────────────────────────

def build_windows_from_track(xs, ys, ts_ns):
    """
    Given a track's position history (in robot-relative coords at final frame),
    build sliding windows matching the THÖR-MAGNI format:
      X: (N, 40) — [rel_x, rel_y, dx, dy] × T
      y: (N, 2)  — [vx, vy]
    """
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    ts = np.array(ts_ns, dtype=np.float64) / 1e9  # → seconds

    n = len(xs)
    if n < MIN_TRACK:
        return None, None

    # Smooth with Savitzky-Golay (same spirit as the THÖR-MAGNI cleaner)
    wl = min(11, n if n % 2 == 1 else n - 1)
    if wl >= 5:
        xs = savgol_filter(xs, wl, 3)
        ys = savgol_filter(ys, wl, 3)

    # Finite-difference velocity
    dt = np.diff(ts)
    dt = np.where(dt < 1e-6, 1e-6, dt)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    # Speed filter
    speed = np.hypot(vx, vy)
    if speed.max() > MAX_SPEED * 3:
        return None, None

    # Frame displacements
    dx = np.concatenate([[0.0], np.diff(xs)])
    dy = np.concatenate([[0.0], np.diff(ys)])

    X_list, y_list = [], []
    for start in range(0, n - T - 1, STRIDE):
        window_x  = xs[start:start+T]
        window_y  = ys[start:start+T]
        window_dx = dx[start:start+T]
        window_dy = dy[start:start+T]

        target_vx = vx[start+T-1]
        target_vy = vy[start+T-1]

        if not (np.isfinite(window_x).all() and np.isfinite(target_vx)):
            continue

        features = np.column_stack([window_x, window_y, window_dx, window_dy]).flatten()
        X_list.append(features)
        y_list.append([target_vx, target_vy])

    if not X_list:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def process_bag(bag_dir: Path, output_dir: Path, min_track_len: int):
    print(f"\n[05_process_rosbag] Processing: {bag_dir.name}")

    # ── Load all messages ───────────────────────────────────────────────────────
    print("  Loading /odom …")
    odom_frames = []
    for ts, data in iter_messages(bag_dir, "/odom"):
        _, rx, ry, ryaw = decode_odom(data)
        odom_frames.append((ts, rx, ry, ryaw))
    if not odom_frames:
        print("  [WARN] No /odom messages found — skipping bag.")
        return None, None

    print(f"  Loaded {len(odom_frames):,} odom frames")

    def get_robot_pose(ts_ns):
        """Linear interpolate robot pose at timestamp ts_ns."""
        if ts_ns <= odom_frames[0][0]:
            return odom_frames[0][1:]
        if ts_ns >= odom_frames[-1][0]:
            return odom_frames[-1][1:]
        for i in range(len(odom_frames) - 1):
            t0, t1 = odom_frames[i][0], odom_frames[i+1][0]
            if t0 <= ts_ns <= t1:
                alpha = (ts_ns - t0) / (t1 - t0)
                r0, r1 = odom_frames[i][1:], odom_frames[i+1][1:]
                return (
                    r0[0] + alpha*(r1[0]-r0[0]),
                    r0[1] + alpha*(r1[1]-r0[1]),
                    r0[2] + alpha*(r1[2]-r0[2]),
                )
        return odom_frames[-1][1:]

    # ── Load scan messages ──────────────────────────────────────────────────────
    print("  Loading /scan …")
    scan_frames = []
    for ts, data in iter_messages(bag_dir, "/scan"):
        _, angles, ranges = decode_scan(data)
        scan_frames.append((ts, angles, ranges))
    print(f"  Loaded {len(scan_frames):,} scan frames")

    # ── Load depth messages (subsample: every 3rd for speed) ───────────────────
    print("  Loading /camera/depth/image_raw …")
    depth_frames = []
    for i, (ts, data) in enumerate(iter_messages(bag_dir, "/camera/depth/image_raw")):
        if i % 3 == 0:
            _, img = decode_image(data)
            depth_frames.append((ts, img))
    print(f"  Loaded {len(depth_frames):,} depth frames (subsampled ×3)")

    # ── Resample to 10 Hz using scan as clock ──────────────────────────────────
    if not scan_frames:
        print("  [WARN] No /scan messages — skipping.")
        return None, None

    t_start = scan_frames[0][0]
    t_end   = scan_frames[-1][0]
    target_interval_ns = int(1e9 / HZ)
    target_times = list(range(t_start, t_end, target_interval_ns))
    print(f"  Resampling to {HZ} Hz → {len(target_times)} frames")

    # Index depth by timestamp for quick lookup
    depth_ts = np.array([f[0] for f in depth_frames]) if depth_frames else np.array([])
    scan_ts  = np.array([f[0] for f in scan_frames])

    # ── Track across frames ────────────────────────────────────────────────────
    tracks: list[Track] = []
    completed_tracks: list[Track] = []

    for frame_ts in target_times:
        # Nearest scan
        si = int(np.argmin(np.abs(scan_ts - frame_ts)))
        _, angles, ranges = scan_frames[si]
        lidar_dets = detect_lidar_clusters(angles, ranges)

        # Nearest depth
        depth_dets = []
        if len(depth_ts) > 0:
            di = int(np.argmin(np.abs(depth_ts - frame_ts)))
            if abs(depth_ts[di] - frame_ts) < 2 * target_interval_ns:
                _, depth_img = depth_frames[di]
                depth_dets = detect_depth_blobs(depth_img)

        # Fuse
        robot_x, robot_y, robot_yaw = get_robot_pose(frame_ts)
        raw_dets = fuse_detections(depth_dets, lidar_dets)

        # Transform detections from sensor frame to world frame
        cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
        world_dets = []
        for (sx, sy) in raw_dets:
            wx = robot_x + cos_y*sx - sin_y*sy
            wy = robot_y + sin_y*sx + cos_y*sy
            world_dets.append((wx, wy))

        # Update tracker
        # Mark dead tracks
        dead = [t for t in tracks if t.missed > 8]
        completed_tracks.extend(dead)
        tracks = [t for t in tracks if t.missed <= 8]
        tracks = match_detections(tracks, world_dets, frame_ts)

    # Flush remaining tracks
    completed_tracks.extend(tracks)

    print(f"  Total tracks recorded: {len(completed_tracks)}")

    # ── Build windows from completed tracks ────────────────────────────────────
    X_all, y_all = [], []
    n_used = 0
    for track in completed_tracks:
        if len(track.xs) < min_track_len:
            continue
        # Convert to robot-relative coords using final robot pose (simple approach)
        # Use the robot pose at the last timestamp of the track
        rx, ry, ryaw = get_robot_pose(track.ts[-1])
        rel_xs, rel_ys = track.to_robot_frame(rx, ry, ryaw)
        X, y = build_windows_from_track(rel_xs, rel_ys, track.ts)
        if X is not None:
            X_all.append(X)
            y_all.append(y)
            n_used += 1

    if not X_all:
        print("  [WARN] No valid windows extracted — check sensor data.")
        return None, None

    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)

    print(f"  Tracks used: {n_used} / {len(completed_tracks)}")
    print(f"  Windows: X={X_combined.shape}, y={y_combined.shape}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = bag_dir.name.replace("domain_adapt_", "")
    X_path = output_dir / f"X_adapt_{suffix}.npy"
    y_path = output_dir / f"y_adapt_{suffix}.npy"
    np.save(str(X_path), X_combined)
    np.save(str(y_path), y_combined)
    print(f"  Saved → {X_path.name}, {y_path.name}")

    return X_combined, y_combined


def main():
    parser = argparse.ArgumentParser(description="Process domain adaptation rosbags")
    parser.add_argument("--bag", required=True, nargs="+",
                        help="Path(s) to ROS2 bag directory/directories")
    parser.add_argument("--output-dir", default="dataset/domain_adaptation",
                        help="Output directory for X_adapt.npy / y_adapt.npy")
    parser.add_argument("--min-track-len", type=int, default=MIN_TRACK,
                        help=f"Minimum track length in frames (default: {MIN_TRACK})")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir   = project_root / args.output_dir

    X_bags, y_bags = [], []
    for bag_str in args.bag:
        bag_path = Path(bag_str)
        if not bag_path.is_absolute():
            bag_path = project_root / bag_path
        if not bag_path.exists():
            print(f"[WARN] Bag not found: {bag_path} — skipping")
            continue
        X, y = process_bag(bag_path, output_dir, args.min_track_len)
        if X is not None:
            X_bags.append(X)
            y_bags.append(y)

    if not X_bags:
        print("\n[ERROR] No valid data extracted from any bag.")
        return

    # Concatenate all bags into combined files for fine-tuning
    X_all = np.concatenate(X_bags, axis=0)
    y_all = np.concatenate(y_bags, axis=0)

    np.save(str(output_dir / "X_adapt.npy"), X_all)
    np.save(str(output_dir / "y_adapt.npy"), y_all)

    print(f"\n✓ Combined domain adaptation dataset:")
    print(f"  X: {X_all.shape}  (should be (N, 40))")
    print(f"  y: {y_all.shape}  (should be (N, 2))")
    print(f"  Saved to: {output_dir}/")
    print(f"\nNext step: python3 training/finetune.py")


if __name__ == "__main__":
    main()
