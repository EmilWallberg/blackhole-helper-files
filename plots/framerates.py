import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set your data directory
type = "zoom"

DATA_DIR = f"C:/Users/wilbj602/Documents/blackhole_{type}/"

# Search for matching CSV files
csv_files = glob.glob(os.path.join(DATA_DIR, "frame_rates_*.csv"))

# Dictionary to organize data
# Format: {BlackholeType: {rayCount: [fps_values]}}
data = {}

for file in csv_files:
    # Extract metadata from filename
    basename = os.path.basename(file)
    try:
        _,_, bh_type, ray_count = basename[:-4].split("_")
        ray_count = int(ray_count)
    except ValueError:
        print(f"Skipping file with unexpected name: {basename}")
        continue

    # Read FPS data from CSV
    try:
        df = pd.read_csv(file)
        if 'FrameRate' not in df.columns:
            print(f"Skipping {basename}: no 'FrameRate' column")
            continue
        fps_values = df['FrameRate'].dropna().values
        ms_values = 1000 / fps_values
    except Exception as e:
        print(f"Error reading {basename}: {e}")
        continue

    # Store in dictionary
    data.setdefault(bh_type, {}).setdefault(ray_count, []).extend(ms_values)

# Plotting
for bh_type, ray_dict in data.items():
    num_plots = len(ray_dict)
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axs = [axs]  # Ensure axs is iterable

    for ax, (ray_count, fps_list) in zip(axs, sorted(ray_dict.items())):
        sns.boxplot(data=[fps_list], ax=ax, showfliers=False)
        ax.set_title(f"{ray_count} rays")
        ax.set_xticks([])  # No x-axis label needed for single box
        ax.set_xlabel("")
        ax.grid(True)

    # fig.suptitle(f"FPS Distribution for {bh_type}", fontsize=16)
    fig.text(0.04, 0.5, 'Time (ms)', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    plt.savefig(f"ms_boxplot_{type}_{bh_type}.png")
    plt.show()
