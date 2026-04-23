import zipfile
import pandas as pd
import io

zip_path = "dataset/THOR_MAGNI.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    
    # Check one CSV from each scenario
    print("=== CSVs_Scenarios Sample ===")
    scenario_files = [f for f in z.namelist() 
                     if 'CSVs_Scenarios/Scenario_1' in f and f.endswith('.csv')]
    
    if scenario_files:
        with z.open(scenario_files[0]) as f:
            df = pd.read_csv(f, nrows=5)
            print(f"File: {scenario_files[0]}")
            print(f"Columns: {list(df.columns)}")
            print(df.head(3))
    
    print("\n=== Lidar_Sample Files Sample ===")
    lidar_csvs = [f for f in z.namelist() 
                 if 'Lidar_Sample/Files' in f and f.endswith('.csv')]
    
    if lidar_csvs:
        with z.open(lidar_csvs[0]) as f:
            df = pd.read_csv(f, nrows=5)
            print(f"File: {lidar_csvs[0]}")
            print(f"Columns: {list(df.columns)}")
            print(df.head(3))
    
    print("\n=== TSVs_RAWET Sample ===")
    tsv_files = [f for f in z.namelist() 
                if 'TSVs_RAWET/Files' in f and f.endswith('.tsv')]
    
    if tsv_files:
        with z.open(tsv_files[0]) as f:
            df = pd.read_csv(f, sep='\t', nrows=5)
            print(f"File: {tsv_files[0]}")
            print(f"Columns: {list(df.columns)}")
            print(df.head(3))