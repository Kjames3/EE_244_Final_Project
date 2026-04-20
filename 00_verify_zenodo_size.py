import requests

record_id = "13865754"
url = f"https://zenodo.org/api/records/{record_id}"
response = requests.get(url)
data = response.json()

total_bytes = sum(f.get('size', 0) for f in data.get('files', []))
print(f"Total download size: {total_bytes / 1e9:.2f} GB")
print(f"\nFiles in this record:")
for f in data.get('files', []):
    size_gb = f.get('size', 0) / 1e9
    print(f"  {f.get('key', 'unknown')}: {size_gb:.2f} GB")