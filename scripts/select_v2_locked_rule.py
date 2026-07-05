"""
select_v2_locked_rule.py

Applies the locked V2 calibration selection rule.

Important:
    This script does not regenerate the MSLS split.
    It uses the existing calibration sweep and existing split file.

Locked rule:
    1. Maximise calibration R@1.
    2. If tied, choose smaller candidate_pool.
    3. If still tied, choose smaller rho.

R@5 and R@10 are reported only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

SWEEP_PATH = OUTPUT_DIR / "msls_calibration_sweep.csv"
SPLIT_PATH = OUTPUT_DIR / "msls_calibration_split_indices.npz"
OUT_PATH = OUTPUT_DIR / "calibrated_v2_selection_locked_rule.json"


def main():
    if not SWEEP_PATH.exists():
        raise FileNotFoundError(f"Missing calibration sweep: {SWEEP_PATH}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing fixed split file: {SPLIT_PATH}")

    df = pd.read_csv(SWEEP_PATH)
    split = np.load(SPLIT_PATH)

    calibration_indices = split["calibration_indices"]
    evaluation_indices = split["evaluation_indices"]

    print("Using existing fixed MSLS split:")
    print("Calibration queries:", len(calibration_indices))
    print("Held-out evaluation queries:", len(evaluation_indices))
    print("Seed:", int(split["seed"]))
    print("Calibration fraction:", float(split["calibration_fraction"]))

    selected = (
        df.sort_values(
            by=["R@1", "candidate_pool", "rho"],
            ascending=[False, True, True],
        )
        .iloc[0]
        .to_dict()
    )

    selection = {
        "selection_source": "MSLS-val calibration split only",
        "selection_rule": "max R@1, then smaller candidate_pool, then smaller rho; R@5/R@10 reported only",
        "seed": int(split["seed"]),
        "calibration_fraction": float(split["calibration_fraction"]),
        "n_calibration_queries": int(len(calibration_indices)),
        "n_heldout_queries": int(len(evaluation_indices)),
        "uncertainty_k": int(selected["uncertainty_k"]),
        "selected_rho": float(selected["rho"]),
        "selected_candidate_pool": int(selected["candidate_pool"]),
        "calibration_R@1": float(selected["R@1"]),
        "calibration_R@5": float(selected["R@5"]),
        "calibration_R@10": float(selected["R@10"]),
        "calibration_count@1": int(selected["count@1"]),
        "calibration_mean_alpha": float(selected["mean_alpha"]),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(selection, f, indent=2)

    print("\nSelected setting under locked rule:")
    print(json.dumps(selection, indent=2))

    print("\nSaved:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
