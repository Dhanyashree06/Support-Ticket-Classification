from pathlib import Path
import os

base_dir = Path(r"c:\Users\Dhanya Shree\OneDrive\Documents\Downloads\Support Ticket Classification")
static_dir = base_dir / "static"
index_file = static_dir / "index.html"

print(f"Base dir: {base_dir} (exists: {base_dir.exists()})")
print(f"Static dir: {static_dir} (exists: {static_dir.exists()})")
print(f"Index file: {index_file} (exists: {index_file.exists()})")

print("\nListing static dir:")
if static_dir.exists():
    for f in static_dir.iterdir():
        print(f"  - {f.name}")
else:
    print("Static dir does not exist")
