import numpy as np
import pandas as pd
import antropy as ant
import nolds
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import glob

left_pressure_cols = [f"LPressure{i}" for i in range(1, 17)]
right_pressure_cols = [f"RPressure{i}" for i in range(1, 17)]
all_pressure_cols = left_pressure_cols + right_pressure_cols

TASK_EVENT_MAP = {
    "SelfPace": ["Walk", "Turn"],
    "SelfPace_mat": ["Walk", "Turn"],
    "SelfPace_matTURN": ["Walk", "Turn"],
    "HurriedPace": ["Walk", "Turn"],
    "HurriedPace_mat": ["Walk", "Turn"],
    "TandemGait": ["TandemWalk"],
    "TUG": ["Walk", "Turn", "SitToStand", "TurnToSit"],
    "Balance": [
        "EC_FeetShoWidth",
        "EC_FeetTogether",
        "EO_FeetShoWidth",
        "EO_FeetTogether",
        "EO_LFootFront",
        "EO_RFootFront",
    ],
}

# Feature extraction functions


def calc_hjorth_mobility(window_data):
    var_raw = np.var(window_data)

    if var_raw < 1e-5:
        return 0.0

    var_diff = np.var(np.diff(window_data))
    return np.sqrt(var_diff / var_raw)


def calc_lempel_ziv(window_data, threshold=5.0):
    bin_string = "".join((window_data > threshold).astype(int).astype(str))
    try:
        return ant.lziv_complexity(bin_string, normalize=True)
    except Exception:
        return 0.0


def calc_sample_entropy(window_data, scale=1):
    try:
        if scale > 1:
            window_data = np.mean(
                window_data[: len(window_data) - len(window_data) % scale].reshape(
                    -1, scale
                ),
                axis=1,
            )
        return ant.sample_entropy(window_data)
    except Exception:
        return 0.0


def calc_markov_spectral_gap(window_data, t_low, t_high):
    if t_low >= t_high:
        return 0.0

    states = np.zeros_like(window_data)
    states[window_data > t_low] = 1
    states[window_data > t_high] = 2

    t_matrix = np.zeros((3, 3))
    for t in range(len(states) - 1):
        t_matrix[int(states[t]), int(states[t + 1])] += 1

    row_sums = t_matrix.sum(axis=1, keepdims=True)
    t_matrix = np.divide(
        t_matrix, row_sums, out=np.zeros_like(t_matrix), where=row_sums != 0
    )

    try:
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(t_matrix)))[::-1]
        return eigenvalues[0] - eigenvalues[1]
    except Exception:
        return 0.0


def calc_katz_fd(window_data):
    if np.var(window_data) < 1e-5:
        return 0.0

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            return ant.katz_fd(window_data)
    except Exception:
        return 0.0


def calc_dfa_alpha(window_data):
    if np.var(window_data) < 1e-5:
        return 0.5

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nolds.dfa(window_data)
    except Exception:
        return 0.5


def calc_hurst_exponent(window_data):
    try:
        return nolds.hurst_rs(window_data)
    except Exception:
        return 0.5


def process_patient_walk_uncollapsed(
    gait_matrix, sampling_rate=100, window_duration=5.12, overlap=0.5
):
    """Slices data and returns a raw list of window dictionaries without aggregating."""
    num_timestamps, num_sensors = gait_matrix.shape
    window_size = int(window_duration * sampling_rate)
    step_size = int(window_size * (1 - overlap))

    # Pre-calculate dynamic Markov thresholds for this specific event matrix
    sensor_thresholds = {}
    for sensor_idx in range(num_sensors):
        active_pressure = gait_matrix[:, sensor_idx][gait_matrix[:, sensor_idx] > 0.1]
        if len(active_pressure) > 0:
            t_low = np.percentile(active_pressure, 10)
            t_high = np.percentile(active_pressure, 80)
        else:
            t_low, t_high = 1.0, 10.0
        sensor_thresholds[sensor_idx] = (t_low, t_high)

    raw_windows_list = []
    start_idx = 0

    while start_idx + window_size <= num_timestamps:
        end_idx = start_idx + window_size
        window_chunk = gait_matrix[start_idx:end_idx, :]
        current_window_features = {}

        for sensor_idx in range(num_sensors):
            sensor_data = window_chunk[:, sensor_idx]
            prefix = f"sensor_{sensor_idx}"

            t_low, t_high = sensor_thresholds[sensor_idx]

            current_window_features[f"{prefix}_hjorth"] = calc_hjorth_mobility(
                sensor_data
            )
            current_window_features[f"{prefix}_lzc"] = calc_lempel_ziv(sensor_data)
            current_window_features[f"{prefix}_sampen_s1"] = calc_sample_entropy(
                sensor_data, scale=1
            )
            current_window_features[f"{prefix}_sampen_s3"] = calc_sample_entropy(
                sensor_data, scale=3
            )
            current_window_features[f"{prefix}_markov"] = calc_markov_spectral_gap(
                sensor_data, t_low, t_high
            )
            current_window_features[f"{prefix}_katz"] = calc_katz_fd(sensor_data)
            current_window_features[f"{prefix}_dfa"] = calc_dfa_alpha(sensor_data)
            current_window_features[f"{prefix}_hurst"] = calc_hurst_exponent(
                sensor_data
            )

        raw_windows_list.append(current_window_features)
        start_idx += step_size

    return raw_windows_list


def extract_patient_features_compressed(df, file_path):
    """Builds the 'Giant Bucket' of windows across all events and compresses to 1024 features."""
    base_name = os.path.basename(file_path).replace(".csv", "")
    try:
        subject_id, task_name = base_name.split("_", 1)
    except ValueError:
        task_name = "Unknown"
        subject_id = base_name

    print(f"Processing {subject_id} - {task_name}")

    # Check for ghost files
    missing_cols = [col for col in all_pressure_cols if col not in df.columns]
    if missing_cols:
        print(
            f"⚠️ Subject {subject_id}: Missing {len(missing_cols)} pressure columns. Dropped."
        )
        return None

    df[all_pressure_cols] = (
        df[all_pressure_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    if df[all_pressure_cols].sum().sum() == 0:
        print(f"⚠️ Subject {subject_id}: Pressure data is completely empty. Dropped.")
        return None

    patient_features = {"subject_id": subject_id}
    events_to_process = TASK_EVENT_MAP.get(task_name, ["Walk"])

    # The Giant Bucket
    master_window_list = []

    for event in events_to_process:
        event_df = df[df["GeneralEvent"] == event]
        if event_df.empty:
            continue

        gait_matrix = event_df[all_pressure_cols].to_numpy()

        # 5.12s window duration based on your snippet
        raw_event_windows = process_patient_walk_uncollapsed(
            gait_matrix, sampling_rate=100, window_duration=5.12, overlap=0.5
        )

        master_window_list.extend(raw_event_windows)

    # If no valid windows were generated across any event
    if not master_window_list:
        print(
            f"⚠️ Subject {subject_id}: Walk duration too short to form a single window. Dropped."
        )
        return None

    # The Global Compression
    feature_keys = master_window_list[0].keys()

    for key in feature_keys:
        global_timeline_values = [window[key] for window in master_window_list]

        patient_features[f"global_{key}_mean"] = np.mean(global_timeline_values)
        patient_features[f"global_{key}_std"] = np.std(global_timeline_values)
        patient_features[f"global_{key}_min"] = np.min(global_timeline_values)
        patient_features[f"global_{key}_max"] = np.max(global_timeline_values)

    return patient_features


def process_single_file(file_path, label):
    """
    A lightweight wrapper that loads one file, extracts it, and adds the label.
    This is the function we will send to the parallel CPU cores.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        features = extract_patient_features_compressed(df, file_path)

        if features is not None:
            features["Label"] = label
            return features
    except Exception as e:
        print(f"❌ Critical Error processing {file_path}: {e}")

    return None


if __name__ == "__main__":
    control_files = glob.glob("download_data/Control/*.csv")
    pd_files = glob.glob("download_data/PD/*.csv")

    all_tasks = [(f, 0) for f in control_files] + [(f, 1) for f in pd_files]
    print(
        f"🚀 Launching Parallel Extraction across {os.cpu_count()} CPU Cores for {len(all_tasks)} files..."
    )
    results = Parallel(n_jobs=-1, verbose=50)(
        delayed(process_single_file)(file_path, label)
        for file_path, label in tqdm(all_tasks, desc="Extracting Patients")
    )

    valid_results = [r for r in results if r is not None]
    final_feature_df = pd.DataFrame(valid_results)
    output_filename = "stochastic_fractal_universal_1024.csv"
    final_feature_df.to_csv(output_filename, index=False)
    print(
        f"\n✅ Parallel Pipeline Complete! Universal dataset saved to '{output_filename}'."
    )
    print(f"Final Data Shape: {final_feature_df.shape}")
