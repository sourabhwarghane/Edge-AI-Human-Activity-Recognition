from pathlib import Path
import pandas as pd

data_dir = Path("data/raw")
activities = ["standing", "walking", "sitting", "falling"]

total_files = 0

for activity in activities:
    files = list((data_dir / activity).glob("*.csv"))
    print(f"{activity}: {len(files)} files")
    total_files += len(files)

    for file in files:
        df = pd.read_csv(file)
        if len(df) != 20:
            print(f"Warning: {file.name} has {len(df)} frames")

print(f"\nTotal sequences: {total_files}")
