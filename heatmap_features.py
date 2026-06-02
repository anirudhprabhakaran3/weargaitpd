import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as ndimage


def load_data(folder, columns):
    # Data dictionary: data["control"]["hc100"]["tug"], data["pd"]["nls002"]["selfpace_mat"]
    data = {}

    # Recursively search for csv files
    search_pattern = os.path.join(folder, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    for file_path in csv_files:
        # Extract file and folder names
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Get cohort ("control" or "pd") from folder name
        cohort = folder_name.lower()

        # Create cohort if it does not exist
        if cohort not in data:
            data[cohort] = {}

        # Extract subject and task
        try:
            subject, task = file_name_no_ext.lower().split("_", 1)
        except ValueError:
            continue

        # Create subject if it does not exist
        if subject not in data[cohort]:
            data[cohort][subject] = {}

        # Load task data with selected columns
        data[cohort][subject][task] = pd.read_csv(
            file_path, usecols=lambda c: c in columns
        )

    return data


def clean_data(data, columns):
    for cohort, subjects in list(data.items()):
        for subject, tasks in list(subjects.items()):
            for task, df in list(tasks.items()):
                # Find missing columns
                missing_cols = [c for c in columns if c not in df.columns]

                # Remove task for subject if incomplete data
                if missing_cols:
                    del tasks[task]
                    continue

                # Clean 'Time' column - remove 'sec' and convert to float
                if "Time" in df.columns and df["Time"].dtype == "O":
                    df["Time"] = (
                        df["Time"].str.replace(" sec", "", regex=False).astype(float)
                    )

                # Drop rows with NaN
                clean_df = df[columns].dropna()

                # Update task if still has data
                if not clean_df.empty:
                    tasks[task] = clean_df
                # Task has no data - delete
                else:
                    del tasks[task]

            # Remove subject if they have no tasks left
            if not tasks:
                del subjects[subject]


def compute_global_bounds(data, margin=0.05):
    """Compute the global min and max for CoP X and Y across all data."""
    all_x = []
    all_y = []

    for cohort in data.keys():
        for subject in data[cohort].keys():
            for task in data[cohort][subject].keys():
                df = data[cohort][subject][task]

                try:
                    # Left foot
                    mask_l = (df["LTotalForce"] > 0) & df["LCoP_X"].notna()
                    if mask_l.any():
                        all_x.append(df.loc[mask_l, "LCoP_X"].values)
                        all_y.append(df.loc[mask_l, "LCoP_Y"].values)

                    # Right foot
                    mask_r = (df["RTotalForce"] > 0) & df["RCoP_X"].notna()
                    if mask_r.any():
                        all_x.append(df.loc[mask_r, "RCoP_X"].values)
                        all_y.append(df.loc[mask_r, "RCoP_Y"].values)
                except KeyError:
                    continue

    if not all_x:
        return 0, 1, 0, 1

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    return x_min, x_max, y_min, y_max


def generate_sample_heatmap(df, xedges, yedges, grid, sigma):
    """Generate a 2-channel heatmap (Left, Right) for a single sample."""
    density_map = np.zeros((2, grid, grid), dtype=np.float32)

    try:
        # Left foot (Channel 0)
        mask_l = (df["LTotalForce"] > 0) & df["LCoP_X"].notna()
        if mask_l.any():
            x_l = df.loc[mask_l, "LCoP_X"].values
            y_l = df.loc[mask_l, "LCoP_Y"].values
            H_l, _, _ = np.histogram2d(x_l, y_l, bins=[xedges, yedges])
            H_l = np.clip(H_l.T, 0, np.percentile(H_l[H_l > 0], 99.99))
            density_l = ndimage.gaussian_filter(H_l, sigma=sigma) / len(x_l)
            density_map[0] = density_l.astype(np.float32)

        # Right foot (Channel 1)
        mask_r = (df["RTotalForce"] > 0) & df["RCoP_X"].notna()
        if mask_r.any():
            x_r = df.loc[mask_r, "RCoP_X"].values
            y_r = df.loc[mask_r, "RCoP_Y"].values
            H_r, _, _ = np.histogram2d(x_r, y_r, bins=[xedges, yedges])
            H_r = np.clip(H_r.T, 0, np.percentile(H_r[H_r > 0], 99.99))
            density_r = ndimage.gaussian_filter(H_r, sigma=sigma) / len(x_r)
            density_map[1] = density_r.astype(np.float32)
    except KeyError:
        pass  # Return empty zeros map if columns are missing

    return density_map


def build_heatmap_dataset(data, grid, sigma):
    """Build the full dataset of heatmaps and labels."""
    print("Computing global bounds...")
    x_min, x_max, y_min, y_max = compute_global_bounds(data)
    print(f"Global X bounds: [{x_min:.4f}, {x_max:.4f}]")
    print(f"Global Y bounds: [{y_min:.4f}, {y_max:.4f}]")

    xedges = np.linspace(x_min, x_max, grid + 1)
    yedges = np.linspace(y_min, y_max, grid + 1)

    X = []
    y = []
    metadata = []  # Keep track of subject and task

    print("Generating heatmaps...")
    for cohort in data.keys():
        label = 0 if cohort == "control" else 1

        for subject in tqdm(data[cohort].keys(), desc=f"{cohort} subjects"):
            for task in data[cohort][subject].keys():
                df = data[cohort][subject][task]
                heatmap = generate_sample_heatmap(
                    df, xedges, yedges, grid=grid, sigma=sigma
                )

                # Check if it's completely empty due to missing data.
                # We skip empty ones to avoid corrupting training with blank images.
                if np.sum(heatmap) > 0:
                    X.append(heatmap)
                    y.append(label)
                    metadata.append({"subject": subject, "task": task})

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), metadata


if __name__ == "__main__":
    # Load and clean data
    data_folder = "./download_data/"
    columns = ["Time"] + [
        f"{foot}{feature}"
        for foot in ("L", "R")
        for feature in (
            *(f"Pressure{i}" for i in range(1, 17)),
            "TotalForce",
            "CoP_X",
            "CoP_Y",
        )
    ]
    print("Loading dataset...")
    print("This might take a couple of minutes...")
    data = load_data(data_folder, columns)
    print("Cleaning data...")
    clean_data(data, columns)

    # Generate heatmaps
    grid_size = 64  # Size of the heatmap grid
    sigma = 1.5  # Controls gaussian blur
    X_heatmaps, y_labels, heatmap_metadata = build_heatmap_dataset(
        data, grid=grid_size, sigma=sigma
    )
    print(f"Generated dataset:")
    print(f"X_heatmaps shape: {X_heatmaps.shape} (N, Channels, H, W)")
    print(f"y_labels shape: {y_labels.shape}")

    # Save the dataset
    output_file = "./heatmap_dataset.npz"
    np.savez_compressed(
        output_file, X=X_heatmaps, y=y_labels, metadata=heatmap_metadata
    )
    print(f"Dataset saved to {output_file}")
