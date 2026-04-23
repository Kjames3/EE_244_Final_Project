import zipfile
import pandas as pd
import numpy as np
import os
import io

zip_path = "dataset/THOR_MAGNI.zip"
output_dir = "dataset/thor_magni_processed"
os.makedirs(output_dir, exist_ok=True)

# Physical plausibility limits
MAX_SPEED_MS     = 3.0   # m/s — fast human sprint upper bound
MAX_POSITION_JUMP_M = 0.5  # m per 100ms — impossible if larger

def parse_thor_magni_csv(z, filepath):
    with z.open(filepath) as f:
        content = f.read().decode('utf-8')
    lines = content.split('\n')
    
    header_row = None
    for i, line in enumerate(lines):
        if line.startswith('"Frame"') or line.startswith('Frame'):
            header_row = i
            break
    if header_row is None:
        return None, None
    
    metadata = {}
    for line in lines[:header_row]:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            metadata[parts[0].strip('"')] = parts[1].strip('"')
    
    data_str = '\n'.join(lines[header_row:])
    df = pd.read_csv(io.StringIO(data_str), header=0, low_memory=False)
    df['Frame'] = pd.to_numeric(df['Frame'], errors='coerce')
    df['Time']  = pd.to_numeric(df['Time'],  errors='coerce')
    df = df.dropna(subset=['Frame', 'Time']).reset_index(drop=True)
    return df, metadata

def get_body_names(df):
    bodies = set()
    for col in df.columns:
        if ' - ' in col and (' X' in col or ' Y' in col or ' Z' in col):
            bodies.add(col.split(' - ')[0].strip())
    return list(bodies)

def extract_centroid(df, body_name):
    x_cols = [c for c in df.columns if c.startswith(body_name) and c.endswith(' X')]
    y_cols = [c for c in df.columns if c.startswith(body_name) and c.endswith(' Y')]
    if not x_cols or not y_cols:
        return None, None
    x = df[x_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    y = df[y_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    return x, y

def clean_and_compute_velocity(x, y, dt=0.01):
    """
    Detect position gaps caused by MoCap occlusion and null out
    velocity at those frames before differentiating.
    """
    x = x.values.copy().astype(float)
    y = y.values.copy().astype(float)
    
    # Convert mm to m
    x_m = x / 1000.0
    y_m = y / 1000.0
    
    # Detect occlusion gaps — frames where position jumps impossibly
    dx = np.abs(np.diff(x_m, prepend=x_m[0]))
    dy = np.abs(np.diff(y_m, prepend=y_m[0]))
    gap_mask = (dx > MAX_POSITION_JUMP_M) | (dy > MAX_POSITION_JUMP_M)
    
    # Also mask NaN positions
    nan_mask = np.isnan(x_m) | np.isnan(y_m)
    
    # Forward fill NaNs for gradient computation, then mask bad frames
    x_filled = pd.Series(x_m).ffill().bfill().values
    y_filled = pd.Series(y_m).ffill().bfill().values
    
    vx = np.gradient(x_filled, dt)
    vy = np.gradient(y_filled, dt)
    
    # Zero out velocity at gap and NaN frames
    bad_frames = gap_mask | nan_mask
    vx[bad_frames] = np.nan
    vy[bad_frames] = np.nan
    
    # Final speed clip as safety net
    speed = np.sqrt(np.nan_to_num(vx)**2 + np.nan_to_num(vy)**2)
    excessive = speed > MAX_SPEED_MS
    vx[excessive] = np.nan
    vy[excessive] = np.nan
    
    return vx, vy, x_m, y_m, bad_frames

def downsample_stride(arr, stride=10):
    return arr[::stride]

# ── Main extraction loop ──────────────────────────────────────────
print("Starting cleaned feature extraction...\n")
summary = []
total_dropped = 0
total_kept = 0

with zipfile.ZipFile(zip_path, 'r') as z:
    csv_files = [f for f in z.namelist()
                if 'CSVs_Scenarios' in f and f.endswith('.csv')]

    for csv_file in csv_files:
        fname = csv_file.split('/')[-1].replace('.csv', '')
        print(f"Processing: {fname}")

        df, metadata = parse_thor_magni_csv(z, csv_file)
        if df is None:
            continue

        bodies     = get_body_names(df)
        pedestrians = [b for b in bodies if 'Robot' not in b]
        robot_name  = next((b for b in bodies if 'Robot' in b), None)

        robot_x, robot_y = None, None
        if robot_name:
            rx, ry = extract_centroid(df, robot_name)
            if rx is not None:
                robot_x = rx.values / 1000.0
                robot_y = ry.values / 1000.0

        sequence_records = []

        for body in pedestrians:
            x, y = extract_centroid(df, body)
            if x is None:
                continue

            vx, vy, x_m, y_m, bad_mask = clean_and_compute_velocity(x, y)

            # Relative position to robot
            if robot_x is not None:
                rel_x = x_m - robot_x
                rel_y = y_m - robot_y
            else:
                rel_x = x_m
                rel_y = y_m

            speed = np.sqrt(np.nan_to_num(vx)**2 + np.nan_to_num(vy)**2)

            # Downsample everything to 10Hz
            time_ds  = downsample_stride(df['Time'].values)
            rel_x_ds = downsample_stride(rel_x)
            rel_y_ds = downsample_stride(rel_y)
            vx_ds    = downsample_stride(vx)
            vy_ds    = downsample_stride(vy)
            speed_ds = downsample_stride(speed)
            bad_ds   = downsample_stride(bad_mask)

            for i in range(len(time_ds)):
                sequence_records.append({
                    'sequence': fname,
                    'body':     body,
                    'time':     time_ds[i],
                    'rel_x':    rel_x_ds[i],
                    'rel_y':    rel_y_ds[i],
                    'vx':       vx_ds[i],
                    'vy':       vy_ds[i],
                    'speed':    speed_ds[i],
                    'bad_frame': bad_ds[i],
                })

        if sequence_records:
            seq_df = pd.DataFrame(sequence_records)

            before = len(seq_df)
            # Drop bad frames and NaN velocities
            seq_df = seq_df[~seq_df['bad_frame']]
            seq_df = seq_df.dropna(subset=['vx', 'vy'])
            seq_df = seq_df.drop(columns=['bad_frame'])
            after  = len(seq_df)

            total_dropped += (before - after)
            total_kept    += after

            out_path = os.path.join(output_dir, f"{fname}_features.csv")
            seq_df.to_csv(out_path, index=False)

            summary.append({
                'sequence':      fname,
                'n_bodies':      len(pedestrians),
                'rows_written':  after,
                'rows_dropped':  before - after,
                'pct_kept':      round(after / before * 100, 1),
            })
            print(f"  Kept: {after:,} | Dropped: {before - after:,} "
                  f"({round((before-after)/before*100,1)}% removed)")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "extraction_summary.csv"), index=False)

print(f"\n{'='*50}")
print(f"Complete. {len(summary)} sequences processed.")
print(f"Total rows kept:    {total_kept:,}")
print(f"Total rows dropped: {total_dropped:,}")
print(f"Overall keep rate:  {round(total_kept/(total_kept+total_dropped)*100,1)}%")