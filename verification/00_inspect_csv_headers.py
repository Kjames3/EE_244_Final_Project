import zipfile
import pandas as pd
import io

zip_path = "dataset/THOR_MAGNI.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    # Read more rows to find the real header
    scenario_files = [f for f in z.namelist() 
                     if 'CSVs_Scenarios/Scenario_1' in f 
                     and f.endswith('.csv')]
    
    print(f"All Scenario_1 files:")
    for f in scenario_files:
        print(f"  {f}")
    
    print("\n=== First 25 rows raw ===")
    with z.open(scenario_files[0]) as f:
        # Read raw lines first to understand structure
        content = f.read().decode('utf-8')
        lines = content.split('\n')
        for i, line in enumerate(lines[:25]):
            print(f"Row {i}: {line[:120]}")  # First 120 chars per line
    
    print("\n=== Lidar PCD file list (first 10) ===")
    pcd_files = [f for f in z.namelist() 
                if 'Lidar_Sample/Files' in f and f.endswith('.pcd')]
    for f in pcd_files[:10]:
        print(f"  {f}")
    
    print(f"\nTotal PCD files: {len(pcd_files)}")