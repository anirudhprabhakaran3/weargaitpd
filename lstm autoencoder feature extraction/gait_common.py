"""
gait_common.py
--------------
Shared utilities for the Parkinson's / Control gait-sensor pipeline.

Expected repository layout (the path you described):

    ./download_data/
        Control/
            HC100_Balance.csv
            HC100_HurriedPace.csv
            ... (8 task files per subject)
        PD/
            PD001_Balance.csv
            ...

Every CSV is one (subject, task) recording sampled at 100 Hz. The first 11
columns are metadata (time, event labels, foot contact/pressure, walkway).
The remaining 286 columns are numeric IMU channels from 14 body-worn sensors
(accelerometer, free-acceleration, gyroscope, magnetometer, velocity/orientation
increments and roll/pitch/yaw per sensor).

All three scripts (01_data_analysis, 02_autoencoder_reduction,
03_latent_analysis) import this module, so keep the four files in one folder.
"""

from __future__ import annotations
import os
import re
import glob
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_ROOT = "download_data"          # root containing Control/ and PD/
GROUPS = ("Control", "PD")           # sub-folder names -> class labels
SAMPLE_RATE_HZ = 100.0               # 0.01 s per row

# --- Sequence preprocessing (WearGait-PD / Anderson et al., 2025, §II.B) ----
# NaNs in IMU streams are wireless dropout. The dataset's own pipeline fills
# them with cubic-spline interpolation along time, with forward/backward fill
# at the recording boundaries -- a sequence-aware fill, NOT a global mean/median
# (which would ignore temporal structure and mix subjects). Acc/gyro channels
# are then low-pass filtered (4th-order zero-lag Butterworth, 20 Hz).
GAP_FILL = True
APPLY_BUTTERWORTH = True
FILTER_CUTOFF_HZ = 20.0
FILTER_ORDER = 4

# The tasks to analyse. ONLY files whose task matches this list are loaded;
# anything else found in the folders is ignored. (Includes the instrumented-mat
# variants, which carry the turn-rich recordings.)
TASKS = [
    "Balance",
    "HurriedPace",
    "HurriedPace_mat",
    "SelfPace",
    "SelfPace_mat",
    "SelfPace_matTURN",
    "TUG",
    "TandemGait",
]
ALLOWED_TASKS = set(TASKS)

# Non-sensor columns. Walkway_* are pipe-delimited text in the *_mat files and
# are NOT usable as plain numeric features, so they live here and are excluded
# from the sensor block.
META_COLS = [
    "Time", "GeneralEvent", "ClinicalEvent",
    "L Foot Contact", "R Foot Contact",
    "L Foot Pressure", "R Foot Pressure",
    "Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot",
]


# --------------------------------------------------------------------------- #
# Discovery & parsing
# --------------------------------------------------------------------------- #

def parse_filename(path: str):
    """'HC100_SelfPace_matTURN.csv' -> ('HC100', 'SelfPace_matTURN')."""
    base = os.path.splitext(os.path.basename(path))[0]
    subject, _, task = base.partition("_")   # split on the FIRST underscore
    return subject, task


def is_mat_task(task: str) -> bool:
    """True for the instrumented-mat variants (HurriedPace_mat, SelfPace_mat, SelfPace_matTURN)."""
    return "_mat" in task.lower()


def discover_recordings(data_root: str = DATA_ROOT,
                        allowed_tasks=ALLOWED_TASKS) -> pd.DataFrame:
    """
    Scan download_data/{Control,PD}/*.csv and return an index DataFrame:
        columns = [path, subject, group, task]
    Only files whose task is in `allowed_tasks` (the TASKS list) are kept;
    any other file in the folders is ignored.
    """
    rows = []
    for group in GROUPS:
        folder = os.path.join(data_root, group)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            subject, task = parse_filename(path)
            if allowed_tasks is not None and task not in allowed_tasks:
                continue
            rows.append(dict(path=path, subject=subject, group=group, task=task))
    df = pd.DataFrame(rows, columns=["path", "subject", "group", "task"])
    if df.empty:
        raise FileNotFoundError(
            f"No CSVs matching TASKS found under '{data_root}/(Control|PD)'. "
            f"Run these scripts from the directory that CONTAINS '{data_root}'."
        )
    return df


def _time_to_seconds(series: pd.Series) -> pd.Series:
    """'0.01 sec' -> 0.01  (robust to already-numeric input)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return (
        series.astype(str)
        .str.replace("sec", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def load_recording(path: str) -> pd.DataFrame:
    """Load one CSV, normalise the Time column to float seconds."""
    df = pd.read_csv(path, low_memory=False)
    if "Time" in df.columns:
        df["Time"] = _time_to_seconds(df["Time"])
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Numeric IMU channels = every column not in META_COLS."""
    return [c for c in df.columns if c not in META_COLS]


def sensor_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (float32 array, column names) of the numeric sensor block."""
    cols = sensor_columns(df)
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return X, cols


# --------------------------------------------------------------------------- #
# Sequence-aware preprocessing  (matches the WearGait-PD source pipeline)
# --------------------------------------------------------------------------- #

def gap_fill_sequence(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Fill NaN gaps PER RECORDING, along the time axis (Anderson et al., 2025):
        * interior gaps  -> cubic-spline interpolation
        * recording edges-> forward then backward fill
    Channels that are entirely NaN for this recording (a sensor never worn /
    full dropout) cannot be interpolated and are left as NaN here; they are
    resolved later at the task level (dropped if dead everywhere, else filled
    with the training median). Operates on a copy and returns the modified df.
    """
    df = df.copy()
    block = df[cols].apply(pd.to_numeric, errors="coerce")
    # cubic needs >=4 valid points per column; fall back to linear if it raises
    try:
        block = block.interpolate(method="cubic", axis=0)
    except Exception:
        block = block.interpolate(method="linear", axis=0)
    block = block.ffill().bfill()          # boundary fill
    df[cols] = block
    return df


def butter_lowpass(df: pd.DataFrame, cols: list[str],
                   fs: float = SAMPLE_RATE_HZ,
                   cutoff: float = FILTER_CUTOFF_HZ,
                   order: int = FILTER_ORDER) -> pd.DataFrame:
    """
    Zero-lag (filtfilt) Butterworth low-pass on acceleration & gyroscope
    channels only, as in the source pipeline. Channels still containing NaN
    (fully-missing in this recording) or too-short recordings are skipped.
    """
    from scipy.signal import butter, filtfilt
    df = df.copy()
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    target = [c for c in cols if ("Acc" in c or "Gyr" in c)]
    X = df[target].to_numpy(dtype=float)
    ok = ~np.isnan(X).any(axis=0)          # only fully-valid columns
    padlen = 3 * max(len(a), len(b))
    if X.shape[0] > padlen and ok.any():
        Xf = X.copy()
        Xf[:, ok] = filtfilt(b, a, X[:, ok], axis=0)
        df[target] = Xf
    return df


def preprocess_recording(df: pd.DataFrame,
                         gap_fill: bool = GAP_FILL,
                         butter: bool = APPLY_BUTTERWORTH) -> pd.DataFrame:
    """Apply the sequence-aware fill (+ optional low-pass) to one recording."""
    cols = sensor_columns(df)
    if gap_fill:
        df = gap_fill_sequence(df, cols)
    if butter:
        df = butter_lowpass(df, cols)
    return df


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sensor_group(col: str) -> str:
    """
    Map a channel name to its IMU location, e.g.
    'L_LatShank_Gyr_X' -> 'L_LatShank', 'LowerBack_Acc_Y' -> 'LowerBack'.
    Used only for descriptive grouping in the reports.
    """
    m = re.match(r"([A-Za-z]+_?[A-Za-z]*?)_(Acc|FreeAcc|Gyr|Mag|VelInc|OriInc|Roll|Pitch|Yaw)", col)
    return m.group(1) if m else col


def pick_device(verbose: bool = True):
    """
    Choose the best available torch device, cross-platform:
      CUDA (NVIDIA) -> MPS (Apple Silicon GPU) -> CPU.
    Imported lazily so this module has no hard torch dependency for scripts
    (e.g. 01_data_analysis.py) that never call it.
    """
    import torch
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"                       # Apple Silicon GPU (Metal)
    else:
        dev = "cpu"
    if verbose:
        print(f"Using device: {dev}")
    return dev
