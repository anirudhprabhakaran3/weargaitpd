"""
04c_lstm_inference.py
=====================
Extract per-timestep BiLSTM latent features for one or more IMU CSV files
using the models trained by 02c_lstm_autoencoder_reduction.py.

You point this at an IMU CSV (or a folder of them). For each file it:
  1. Parses the task from the filename (HC100_TUG.csv -> task=TUG).
  2. Loads the matching saved model + scaler + channel list from
     ./<task>/ (the script's working directory; override with --models-dir).
  3. Applies the SAME preprocessing as training (cubic-spline NaN fill,
     Butterworth low-pass, channel keep-list, scaler, +/- clip).
  4. Runs the BiLSTM encoder over the full recording end-to-end (long files
     are processed in encode-chunks).
  5. Writes one output file per input recording with one row per timestep:
       Time, GeneralEvent (if present), ClinicalEvent (if present),
       z0..z{L-1}, recon_error

Expected layout (working directory):
    ./04c_lstm_inference.py
    ./gait_common.py
    ./Balance/                  <- one folder per task, accessible as ./<task>/
        lstm_autoencoder.pt
        scaler.joblib
        kept_channels.json
    ./TUG/
    ./HurriedPace/
    ... (one folder per trained task)

Usage:
    python 04c_lstm_inference.py path/to/one_file.csv
    python 04c_lstm_inference.py path/to/folder/                  (recurses)
    python 04c_lstm_inference.py path/to/folder/ --out my_latents/
    python 04c_lstm_inference.py file.csv --task TUG              (override task)
    python 04c_lstm_inference.py file.csv --models-dir other_dir/ (override layout)
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

import gait_common as gc


DEFAULT_MODELS_DIR = "."
DEFAULT_OUT_DIR = "outputs/04c_inference"


# Mirror of the training-time class. Kept here so this script is standalone.
class BiLSTMAutoencoder(nn.Module):
    def __init__(self, n_channels, latent_dim, lstm_hidden, lstm_layers):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.encoder = nn.LSTM(input_size=n_channels, hidden_size=lstm_hidden,
                               num_layers=lstm_layers, batch_first=True,
                               bidirectional=True,
                               dropout=0.1 if lstm_layers > 1 else 0.0)
        self.to_latent = nn.Linear(2 * lstm_hidden, latent_dim)
        self.from_latent = nn.Linear(latent_dim, lstm_hidden)
        self.decoder = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden,
                               num_layers=lstm_layers, batch_first=True,
                               bidirectional=False,
                               dropout=0.1 if lstm_layers > 1 else 0.0)
        self.to_output = nn.Linear(lstm_hidden, n_channels)

    def encode(self, x):
        h, _ = self.encoder(x)
        return self.to_latent(h)

    def decode(self, z):
        d, _ = self.decoder(self.from_latent(z))
        return self.to_output(d)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# --------------------------- per-task asset loader ------------------------- #
def load_task_assets(task, models_dir, device):
    """Returns (model, scaler, kept_channels, clip_sigma) for a given task."""
    task_dir = os.path.join(models_dir, task)
    ckpt_path = os.path.join(task_dir, "lstm_autoencoder.pt")
    scaler_path = os.path.join(task_dir, "scaler.joblib")
    cfg_path = os.path.join(task_dir, "kept_channels.json")
    for p in (ckpt_path, scaler_path, cfg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing model asset for task '{task}': {p}")

    with open(cfg_path) as f:
        meta = json.load(f)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = BiLSTMAutoencoder(
        n_channels=ckpt["n_channels"], latent_dim=ckpt["latent_dim"],
        lstm_hidden=ckpt["lstm_hidden"], lstm_layers=ckpt["lstm_layers"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)
    kept = meta["kept_channels"]
    clip_sigma = meta.get("clip_sigma")
    return model, scaler, kept, clip_sigma


# --------------------------- preprocessing one file ------------------------ #
def preprocess_one(path, kept_channels):
    """Same pipeline as training: load -> gap fill -> low-pass -> select kept channels."""
    df = gc.load_recording(path)
    df = gc.preprocess_recording(df)
    cols_present = gc.sensor_columns(df)
    missing = [c for c in kept_channels if c not in cols_present]
    if missing:
        raise ValueError(
            f"File '{path}' is missing {len(missing)} channels expected by the model "
            f"(first few: {missing[:3]}). Was the model trained on a different schema?")
    X = df[kept_channels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    # any residual NaN (whole-recording-missing channel) -> 0 (== training median after standardising)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # collect frame-level metadata
    meta = pd.DataFrame({
        "Time": df["Time"].to_numpy() if "Time" in df else np.arange(len(df)) / gc.SAMPLE_RATE_HZ,
        "GeneralEvent": df.get("GeneralEvent", pd.Series(["NA"] * len(df))).astype(str).to_numpy(),
        "ClinicalEvent": df.get("ClinicalEvent", pd.Series(["NA"] * len(df))).astype(str).to_numpy(),
    })
    return X, meta


@torch.no_grad()
def encode_recording(model, Xs, device, chunk=4000):
    """Encode one recording timestep by timestep; long files in chunks."""
    n = len(Xs)
    Z = np.zeros((n, model.latent_dim), dtype=np.float32)
    err = np.zeros(n, dtype=np.float32)
    for cs in range(0, n, chunk):
        ce = min(cs + chunk, n)
        xb = torch.from_numpy(Xs[cs:ce]).unsqueeze(0).to(device)
        recon, z = model(xb)
        Z[cs:ce] = z.squeeze(0).cpu().numpy()
        err[cs:ce] = ((recon - xb) ** 2).mean(dim=2).squeeze(0).cpu().numpy()
    return Z, err


# --------------------------- main: process files --------------------------- #
def gather_files(path):
    """Accept a single CSV or a folder (recursive)."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    raise FileNotFoundError(f"Path not found: {path}")


def process_one(path, models_dir, out_dir, device, task_override, encode_chunk,
                _cache):
    subject, parsed_task = gc.parse_filename(path)
    task = task_override or parsed_task
    if task not in gc.ALLOWED_TASKS:
        print(f"  [skip] {os.path.basename(path)}: task '{task}' not in TASKS whitelist.")
        return None

    if task not in _cache:
        _cache[task] = load_task_assets(task, models_dir, device)
    model, scaler, kept, clip_sigma = _cache[task]

    X, meta = preprocess_one(path, kept)
    Xs = scaler.transform(X).astype(np.float32)
    if clip_sigma:
        np.clip(Xs, -clip_sigma, clip_sigma, out=Xs)
    Z, err = encode_recording(model, Xs, device, chunk=encode_chunk)

    out = meta.copy()
    for j in range(Z.shape[1]):
        out[f"z{j}"] = Z[:, j]
    out["recon_error"] = err

    base = os.path.splitext(os.path.basename(path))[0]
    out_subdir = gc.ensure_dir(os.path.join(out_dir, task))
    out_path = os.path.join(out_subdir, f"{base}_latent.parquet")
    try:
        out.to_parquet(out_path, index=False)
    except Exception:
        out_path = os.path.join(out_subdir, f"{base}_latent.csv")
        out.to_csv(out_path, index=False)

    print(f"  [{task}] {os.path.basename(path)}  ->  {out_path}  "
          f"({len(out)} frames, mean recon err {err.mean():.4f})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract BiLSTM latent features from IMU CSVs.")
    ap.add_argument("input", help="CSV file or folder of CSV files")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                    help="folder containing one subfolder per task "
                         "(each with lstm_autoencoder.pt, scaler.joblib, "
                         "kept_channels.json). Default: current directory, "
                         "so ./<task>/ is used.")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help="output folder for *_latent.parquet files (default: %(default)s)")
    ap.add_argument("--task", default=None,
                    help="override task name (otherwise parsed from filename)")
    ap.add_argument("--encode-chunk", type=int, default=4000,
                    help="frames per encoding chunk for long files (default: 4000)")
    args = ap.parse_args()

    device = gc.pick_device()
    files = gather_files(args.input)
    if not files:
        raise SystemExit("No CSV files found.")
    out_dir = gc.ensure_dir(args.out)

    print(f"Encoding {len(files)} file(s) -> {out_dir}")
    cache = {}
    written = []
    for f in files:
        try:
            p = process_one(f, args.models_dir, out_dir, device, args.task,
                            args.encode_chunk, cache)
            if p:
                written.append(p)
        except Exception as e:
            print(f"  [error] {f}: {e}")

    print(f"\nDone. Wrote {len(written)} latent file(s).")


if __name__ == "__main__":
    main()
