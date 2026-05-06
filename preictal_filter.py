"""
Pre-ictal spike filter.

Takes an output directory and a seizure times JSON, then creates
*_results_preictal.json files containing only spikes before the seizure onset.

Seizure times JSON format:
{
    "2810": 300.0,
    "2810_1": 250.5,
    ...
}

Keys are folder names (mouse codes). Values are seizure start times in seconds.

Usage:
    python preictal_filter.py --output-dir SNr_output --seizure-times seizure_times_snr.json
    python preictal_filter.py --output-dir PPN_output --seizure-times seizure_times_ppn.json
"""

import argparse
import glob
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Filter spikes to pre-ictal only")
    parser.add_argument(
        "--output-dir", "-d", required=True,
        help="Root directory containing per-dataset subdirectories"
    )
    parser.add_argument(
        "--seizure-times", "-s", required=True,
        help="JSON file mapping folder names to seizure start times (seconds)"
    )
    args = parser.parse_args()

    with open(args.seizure_times) as f:
        seizure_times = json.load(f)

    for subdir in sorted(os.listdir(args.output_dir)):
        full = os.path.join(args.output_dir, subdir)
        if not os.path.isdir(full):
            continue

        if subdir not in seizure_times:
            print(f"  Skipping {subdir}: no seizure time in JSON")
            continue

        seizure_start = float(seizure_times[subdir])

        json_files = sorted(glob.glob(os.path.join(full, "*_results.json")))
        if not json_files:
            print(f"  Skipping {subdir}: no results JSON found")
            continue

        results_path = json_files[0]
        with open(results_path) as f:
            results = json.load(f)

        total = len(results["spikes"])
        preictal_spikes = [
            s for s in results["spikes"]
            if s["time_seconds"] < seizure_start
        ]
        removed = total - len(preictal_spikes)

        # Build the preictal results
        preictal_results = dict(results)
        preictal_results["spikes"] = preictal_spikes
        preictal_results["summary"] = dict(results.get("summary", {}))
        preictal_results["summary"]["total_spikes"] = len(preictal_spikes)
        preictal_results["summary"]["seizure_start_seconds"] = seizure_start
        preictal_results["summary"]["filter"] = "preictal"

        # Write *_results_preictal.json alongside the original
        base = os.path.basename(results_path).replace("_results.json", "_results_preictal.json")
        out_path = os.path.join(full, base)
        with open(out_path, "w") as f:
            json.dump(preictal_results, f, indent=2)

        print(f"  {subdir}: {len(preictal_spikes)}/{total} spikes kept "
              f"({removed} removed, seizure at {seizure_start}s) → {base}")


if __name__ == "__main__":
    main()
