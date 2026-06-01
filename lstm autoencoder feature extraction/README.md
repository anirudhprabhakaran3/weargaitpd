# BiLSTM IMU Feature Extractor

Extracts **64-dimensional per-timestep latent features** from WearGait-PD IMU
recordings using a pre-trained bidirectional LSTM autoencoder. One model per
movement task; the task is auto-detected from each filename.

The output is an ordered sequence of latent vectors per recording — designed
to be fed to a downstream classifier (e.g. PD-vs-Control).

---

## What's in this bundle

```
.
├── 04c_lstm_inference.py     # entry-point script - you run this
├── gait_common.py            # preprocessing helpers (imported by the script)
├── requirements.txt
├── README.md                 # this file
├── Balance/                  # one folder per trained task
│   ├── lstm_autoencoder.pt   #   trained model weights + architecture config
│   ├── scaler.joblib         #   fitted per-channel StandardScaler
│   └── kept_channels.json    #   channel list, latent dim, clip sigma
├── HurriedPace/
├── HurriedPace_mat/
├── SelfPace/
├── SelfPace_mat/
├── SelfPace_matTURN/
├── TUG/
└── TandemGait/
```

All three files in each task folder are required for inference. Don't rename
them.

---

## Setup

```bash
pip install -r requirements.txt
```

Python **3.9 or newer**. PyTorch will pick CUDA (NVIDIA GPU), Apple MPS, or
CPU automatically depending on what's available — no config needed.

---

## Quick start

Run from the same directory the `./Balance/`, `./TUG/`, ... folders live in.

**One file:**
```bash
python 04c_lstm_inference.py path/to/HC123_TUG.csv
```

**A folder (recurses into subfolders, processes every `.csv`):**
```bash
python 04c_lstm_inference.py path/to/data_folder/
```

**Custom output location:**
```bash
python 04c_lstm_inference.py path/to/folder/ --out my_latents/
```

By default, outputs go to `./outputs/04c_inference/<task>/<basename>_latent.parquet`.

---

## Input requirements

CSV files following the WearGait-PD schema. Required:

- A `Time` column (seconds; uniform 100 Hz sampling assumed).
- The **286 IMU channels** the model was trained on (accelerometer,
  free-acceleration, gyroscope, magnetometer, roll/pitch/yaw, velocity and
  orientation increments). The exact channel list and order are saved in
  each task's `kept_channels.json`.

Optional (carried through to output if present): `GeneralEvent`,
`ClinicalEvent`.

**Task detection.** The script parses the task from the filename. Example:
`HC123_TUG.csv` → task `TUG`. The recognised tasks match the folder names
above. If a filename doesn't match the convention, force the task:

```bash
python 04c_lstm_inference.py file.csv --task TUG
```

If a file uses an unrecognised task, it's skipped with a clear message.

If a file is missing channels the model expects, the script raises an error
rather than silently producing wrong outputs.

---

## Output format

One **Parquet file per input recording**, written to
`outputs/04c_inference/<task>/<basename>_latent.parquet`. Each row is one
timestep, in time order:

| column           | what it is                                  |
| ---------------- | ------------------------------------------- |
| `Time`           | timestamp in seconds                        |
| `GeneralEvent`   | event label if present in input, else `"NA"` |
| `ClinicalEvent`  | event label if present in input, else `"NA"` |
| `z0` … `z63`     | the 64 latent features for this timestep    |
| `recon_error`    | per-timestep reconstruction MSE (sanity check) |

So a 5,000-frame recording yields a Parquet file with 5,000 rows and 67
numeric columns (plus the metadata).

---

## Using the latents in your downstream model

Load with pandas:

```python
import pandas as pd
df = pd.read_parquet("outputs/04c_inference/TUG/HC123_TUG_latent.parquet")
zcols = [c for c in df.columns if c.startswith("z")]
latents = df[zcols].to_numpy()    # shape (T, 64), in time order
```

You now have a `(T, 64)` ordered sequence per recording. Two common ways to
consume it:

**1. As a time series for a temporal classifier (preferred).** Feed the
sequence directly to an LSTM/TCN/Transformer. This is the natural fit — the
autoencoder was specifically designed to produce context-aware per-timestep
features, and a sequence classifier exploits the time axis the autoencoder
already encoded.

**2. Aggregated per recording into a fixed-size vector.** Compute mean, std,
and range across the time axis for each of the 64 dims (~192 features), then
feed to a tabular classifier (logistic regression, random forest, etc.).
Simpler, but it discards dynamics — and dynamics are exactly where
Parkinsonian signal lives, so option 1 will usually score higher.

---

## Three things to know before training a classifier on these features

These come up often enough to be worth stating upfront.

**Subject-level splits are non-negotiable.** Each subject contributes
thousands of latent rows. If you pour them into one pile and do a random
80/20 split, the classifier learns to recognise individuals (gait is
distinctive per person) and you'll see a near-perfect AUC that collapses on
new patients. Use scikit-learn's `GroupKFold` with subject ID as the group:

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X, y, groups=subject_ids):
    ...
```

**The scaler matters as much as the model weights.** The model was trained
on standardised data. The inference script applies the saved `scaler.joblib`
automatically — don't try to load the `.pt` file directly without it; raw
IMU values would be scaled wrong and the latents would be garbage.

**Channel order is fixed.** The 286 channels must appear with the same
names as in `kept_channels.json`. If your data source renames or reorders
columns, the script will refuse rather than silently producing wrong
outputs. Fix the input, don't disable the check.

---

## Command-line reference

```
python 04c_lstm_inference.py INPUT [options]

positional:
  INPUT                 CSV file or folder (recursive)

options:
  --models-dir DIR      Folder containing the per-task subfolders
                        (default: current directory, so ./<task>/ is used).
  --out DIR             Output directory
                        (default: outputs/04c_inference/).
  --task NAME           Override task auto-detection
                        (e.g. TUG, Balance, HurriedPace, ...).
  --encode-chunk N      Frames per encoding chunk for very long recordings
                        (default: 4000). Lower if memory-constrained.
```

---

## Troubleshooting

**`Missing model asset for task 'X': ./X/lstm_autoencoder.pt`**
Your working directory doesn't contain a folder named `X/` with the three
model files. Either `cd` to the directory that does, use `--models-dir`, or
make sure the task folder name matches what's parsed from the filename
(case-sensitive).

**`File 'foo.csv' is missing K channels expected by the model`**
Your CSV doesn't have all 286 channels the model was trained on. Check the
file against the channel list in `<task>/kept_channels.json`.

**Out-of-memory errors on long recordings**
Lower `--encode-chunk` to 2000 or 1000 — this trades a tiny bit of
context-awareness at chunk boundaries for substantially less memory.

**`task 'foo' not in TASKS whitelist`**
The filename parses to a task the script doesn't recognise (only the eight
trained tasks are accepted). Use `--task <known_task>` to force it, or
rename the file.

---

## What this model is and isn't

It's an unsupervised dimensionality reduction: 286 IMU channels per timestep
→ 64 latent numbers per timestep, with each latent informed by its temporal
neighborhood (forward and backward LSTM hidden states). The compression
ratio is about 4.5:1 per timestep, with temporal context baked in.

It is **not** a PD classifier itself. The autoencoder was trained
unsupervised — it never saw diagnostic labels. Its job is to produce a
useful representation; your downstream classifier turns that representation
into predictions.
