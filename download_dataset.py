import os
import requests
import rosbag2_py
import pandas as pd
from rclpy.serialization import deserialize_message

def downsample_bag(input_path, output_path, target_hz=10, source_hz=100):
    stride = source_hz // target_hz
    
    # Storage config is required for ROS2 Humble
    storage_options_r = rosbag2_py.StorageOptions(
        uri=input_path,
        storage_id='sqlite3'
    )

    storage_options_w = rosbag2_py.StorageOptions(
        uri=output_path,
        storage_id='sqlite3'
    )

    converter_options = rosbag2_py.ConverterOptions('', '')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options_r, converter_options)
    
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options_w, converter_options)
    
    # Copy topic metadata to writer first
    topics = reader.get_all_topics_and_types()
    for topic in topics:
        writer.create_topic(topic)
    
    counters = {}
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in counters:
            counters[topic] = 0
        if counters[topic] % stride == 0:
            writer.write(topic, data, timestamp)
        counters[topic] += 1
    
def sync_check(depth_ts, lidar_ts, mocap_ts, tolerance_ms=20):
    tolerance_ns = tolerance_ms * 1e6
    return (abs(depth_ts - lidar_ts) < tolerance_ns and
            abs(depth_ts - mocap_ts) < tolerance_ns and
            abs(lidar_ts - mocap_ts) < tolerance_ns)

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(f"\r {percent:.2f}% ({downloaded//1024//1024}MB / {total//1024//1024}MB)",
                                end="", flush=True)
    print("Download complete.")

def download_thor_magni(output_dir="dataset"):
    record_id = "10837562"
    url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Querying Zenodo API: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    files = data.get("files", [])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for file_info in files:
        file_url = file_info.get("links", {}).get("self")
        if not file_url:
            print("No download link found for file:", file_info.get("key"))
            continue
            
        file_name = file_info.get("key", file_info.get("filename", "unknown"))
        out_path = os.path.join(output_dir, file_name)
        if not os.path.exists(out_path):
            download_file(file_url, out_path)
        else:
            print(f"File {file_name} already exists. Skipping.")

if __name__ == '__main__':
    download_thor_magni()


