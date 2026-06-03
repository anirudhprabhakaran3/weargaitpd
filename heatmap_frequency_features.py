import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as ndimage
from scipy.signal import welch


# ==========================================
# SHARED UTILITIES
# ==========================================


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

                # Clean 'Time' column - remove 'sec' and convert to float safely
                if "Time" in df.columns:
                    try:
                        df["Time"] = (
                            df["Time"]
                            .astype(str)
                            .str.replace(" sec", "", regex=False)
                            .astype(float)
                        )
                    except ValueError:
                        pass

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


# ==========================================
# HEATMAP FEATURES
# ==========================================


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

        for subject in tqdm(data[cohort].keys(), desc=f"Heatmaps: {cohort} subjects"):
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


# ==========================================
# PSD FREQUENCY FEATURES
# ==========================================


def _interpolate_cop(
    df, time_col="Time", cols=["LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y"], fs=100.0
):
    """Helper function to resample CoP data uniformly."""
    if df.empty or time_col not in df.columns:
        return df

    # Safely force the time column to numeric to prevent string math errors
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    df = (
        df.dropna(subset=[time_col])
        .drop_duplicates(subset=[time_col])
        .sort_values(time_col)
    )
    if df.empty:
        return df
    t_min, t_max = df[time_col].min(), df[time_col].max()
    num_points = int(np.floor((t_max - t_min) * fs)) + 1
    df_uniform = pd.DataFrame(
        {time_col: np.linspace(t_min, t_min + (num_points - 1) / fs, num_points)}
    )
    # Merge and interpolate
    existing_cols = [c for c in cols if c in df.columns]
    df_joined = pd.merge_asof(
        df_uniform,
        df[[time_col] + existing_cols],
        on=time_col,
        direction="nearest",
        tolerance=0.5 / fs,
    )
    df_joined[existing_cols] = (
        df_joined[existing_cols]
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
    )
    return df_joined


def compute_welch_psd(signal, fs=100.0, nfft=1000):
    """Computes Welch's PSD safely, keeping frequency bins fixed across different signal lengths."""
    nperseg = min(nfft, len(signal))
    noverlap = nperseg // 2 if nperseg > 0 else 0
    f_dummy = np.fft.rfftfreq(nfft, 1 / fs)

    if len(signal) == 0 or np.std(signal) < 1e-6:
        return f_dummy, np.zeros_like(f_dummy)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f, Pxx = welch(
            signal,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend="linear",
            scaling="density",
        )
    return f, Pxx


def build_psd_dataset(
    data, cols=["LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y"], fs=100.0, freq_lim=(0.1, 5.0)
):
    X = []
    y = []
    metadata = []

    # Pre-calculate masks and feature column names to ensure consistency
    f_dummy = np.fft.rfftfreq(int(10 * fs), 1 / fs)
    mask = (f_dummy >= freq_lim[0]) & (f_dummy <= freq_lim[1])
    f_selected = f_dummy[mask]

    print(
        f"Extracting {len(f_selected)} frequency bins per sensor in the range {freq_lim[0]} to {freq_lim[1]} Hz"
    )

    feature_names = []
    for col in cols:
        for freq in f_selected:
            feature_names.append(f"{col}_{freq:.2f}Hz")

    print("Generating PSD features...")
    for cohort in data.keys():
        label = 0 if cohort == "control" else 1

        for subject in tqdm(data[cohort].keys(), desc=f"PSD: {cohort} subjects"):
            for task in data[cohort][subject].keys():
                df = data[cohort][subject][task]
                df_interp = _interpolate_cop(df, cols=cols, fs=fs)

                sample_features = []
                is_valid = False

                for col in cols:
                    if col in df_interp.columns:
                        signal = df_interp[col].dropna().values
                        f, Pxx = compute_welch_psd(signal, fs=fs, nfft=int(10 * fs))
                        Pxx_selected = Pxx[mask]
                        sample_features.extend(Pxx_selected)
                        if np.sum(Pxx_selected) > 0:
                            is_valid = True
                    else:
                        # Missing column, pad with zeros
                        sample_features.extend(np.zeros(len(f_selected)))

                # We skip samples that are completely flat or empty
                if is_valid:
                    X.append(sample_features)
                    y.append(label)
                    metadata.append((subject, task))

    # Convert to DataFrame
    df_features = pd.DataFrame(X, columns=feature_names)
    df_features.insert(0, "Label", y)
    df_features.insert(0, "Task", [m[1] for m in metadata])
    df_features.insert(0, "Subject", [m[0] for m in metadata])

    return df_features


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    data_folder = "./download_data/"

    # We load the union of columns needed for both Heatmap and PSD extraction
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

    # -----------------------------------------------------
    # 1. Generate Heatmaps
    # -----------------------------------------------------
    print("\n--- Starting Heatmap Generation ---")
    grid_size = 64
    sigma = 1.5
    X_heatmaps, y_labels, heatmap_metadata = build_heatmap_dataset(
        data, grid=grid_size, sigma=sigma
    )
    print(f"Generated heatmap dataset:")
    print(f"X_heatmaps shape: {X_heatmaps.shape} (N, Channels, H, W)")
    print(f"y_labels shape: {y_labels.shape}")

    heatmap_output_file = "./heatmap_dataset.npz"
    np.savez_compressed(
        heatmap_output_file, X=X_heatmaps, y=y_labels, metadata=heatmap_metadata
    )
    print(f"Heatmap Dataset saved to {heatmap_output_file}")

    # -----------------------------------------------------
    # 2. Generate PSD Features
    # -----------------------------------------------------
    print("\n--- Starting PSD Feature Generation ---")
    df_dataset = build_psd_dataset(data)

    print(f"Generated PSD dataset shape: {df_dataset.shape}")

    psd_output_file = "./psd_dataset.csv"
    df_dataset.to_csv(psd_output_file, index=False)
    print(f"PSD Dataset saved to {psd_output_file}")
