import synapseclient
import synapseutils
from synapseclient.operations import get as syn_get, FileOptions
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

PAT = os.environ.get("PAT", "invalidAccessToken")
PROJECT_ID = "syn55052683"


def download_filtered_data(syn, parent_id, download_dir, patients=None, tasks=None):
    print(f"Scanning Synapse folder/project: {parent_id}...")

    walked = synapseutils.walk(syn, parent_id)
    download_count = 0

    for dirpath, _, filenames in walked:
        path_str = dirpath[0]

        if "MAT file" in path_str:
            continue

        if "CONTROL PARTICIPANTS" in path_str:
            target_dir = os.path.join(download_dir, "Control")
        elif "PD PARTICIPANTS" in path_str:
            target_dir = os.path.join(download_dir, "PD")
        else:
            target_dir = download_dir

        for filename, file_id in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            if patients:
                if not any(patient_id in filename for patient_id in patients):
                    continue

            if tasks:
                if "_" in filename:
                    actual_task = filename.split("_", 1)[1].replace(".csv", "")
                    if actual_task not in tasks:
                        continue
                else:
                    continue

            os.makedirs(target_dir, exist_ok=True)

            print(f"Downloading to {target_dir}: {filename}")
            syn_get(
                synapse_id=file_id,
                file_options=FileOptions(
                    download_file=True, download_location=target_dir
                ),
            )
            download_count += 1

    print(f"\n✅ Done! {download_count} files downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download WearGaitPD CSV data by patient or task."
    )

    parser.add_argument(
        "--patients", nargs="+", help="List of patient IDs (e.g., HC100 WPD001)"
    )
    parser.add_argument("--tasks", nargs="+", help="List of tasks (e.g., Balance TUG)")

    args = parser.parse_args()

    print("Authenticating with Synapse...")
    syn = synapseclient.Synapse()
    syn.login(authToken=PAT)
    print("Authentication complete.")

    download_filtered_data(
        syn=syn,
        parent_id=PROJECT_ID,
        download_dir="./download_data",
        patients=args.patients,
        tasks=args.tasks,
    )
