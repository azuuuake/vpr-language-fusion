"""
calibrate_v2_msls.py

Select V2 hyperparameters using only a calibration split from MSLS-val.

Selection rule:
1. Highest R@1 on MSLS calibration split
2. If tied, highest R@5
3. If tied, highest R@10
4. If tied, smaller candidate_pool
5. If tied, smaller rho
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    precision_weighted_fusion,
)


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

SEED = 42
CALIBRATION_FRACTION = 0.30
UNCERTAINTY_K = 10

RHOS = [1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 20, 30]
CANDIDATE_POOLS = [10, 20, 50, 100]


def recall_at_k(retrieved_indices, positive, k):
    correct = 0

    for i in range(retrieved_indices.shape[0]):
        if positive[i, retrieved_indices[i, :k]].any():
            correct += 1

    recall = 100.0 * correct / retrieved_indices.shape[0]
    return recall, correct


def run_v2_subset(
    visual_sim,
    language_sim,
    positive,
    query_indices,
    rho,
    candidate_pool,
):
    visual_q = visual_sim[query_indices]
    language_q = language_sim[query_indices]
    positive_q = positive[query_indices]

    top10_idx = np.argsort(-visual_q, axis=1)[:, :UNCERTAINTY_K]
    visual_top10 = np.take_along_axis(visual_q, top10_idx, axis=1)
    language_top10 = np.take_along_axis(language_q, top10_idx, axis=1)

    su = visual_uncertainty(visual_top10)
    lu = language_uncertainty(language_top10)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    pool_idx = np.argsort(-visual_q, axis=1)[:, :candidate_pool]
    visual_pool = np.take_along_axis(visual_q, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_q, pool_idx, axis=1)

    fused_pool, alpha = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    order = np.argsort(-fused_pool, axis=1)
    retrieved = np.take_along_axis(pool_idx, order, axis=1)

    r1, c1 = recall_at_k(retrieved, positive_q, 1)
    r5, c5 = recall_at_k(retrieved, positive_q, 5)
    r10, c10 = recall_at_k(retrieved, positive_q, 10)

    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "count@1": c1,
        "count@5": c5,
        "count@10": c10,
        "mean_alpha": float(alpha.mean()),
        "mean_SU": float(su.mean()),
        "mean_LU": float(lu.mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visual_sim = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    language_sim = np.load(EMBED_DIR / "msls_val_lang_sim_matrix.npy")
    positive = np.load(EMBED_DIR / "msls_val_positive_matrix.npy").astype(bool)

    n_queries = visual_sim.shape[0]

    rng = np.random.default_rng(SEED)
    all_indices = np.arange(n_queries)
    rng.shuffle(all_indices)

    n_calibration = int(round(CALIBRATION_FRACTION * n_queries))

    calibration_indices = np.sort(all_indices[:n_calibration])
    evaluation_indices = np.sort(all_indices[n_calibration:])

    print("MSLS-val total queries:", n_queries)
    print("Calibration queries:", len(calibration_indices))
    print("Held-out evaluation queries:", len(evaluation_indices))
    print("Seed:", SEED)

    rows = []

    for rho in RHOS:
        for candidate_pool in CANDIDATE_POOLS:
            result = run_v2_subset(
                visual_sim=visual_sim,
                language_sim=language_sim,
                positive=positive,
                query_indices=calibration_indices,
                rho=rho,
                candidate_pool=candidate_pool,
            )

            row = {
                "dataset": "MSLS-val",
                "split": "calibration",
                "seed": SEED,
                "calibration_fraction": CALIBRATION_FRACTION,
                "n_calibration_queries": len(calibration_indices),
                "n_heldout_queries": len(evaluation_indices),
                "uncertainty_k": UNCERTAINTY_K,
                "rho": rho,
                "candidate_pool": candidate_pool,
            }

            row.update(result)
            rows.append(row)

    df = pd.DataFrame(rows)

    selected = (
        df.sort_values(
            by=["R@1", "R@5", "R@10", "candidate_pool", "rho"],
            ascending=[False, False, False, True, True],
        )
        .iloc[0]
        .to_dict()
    )

    csv_path = OUTPUT_DIR / "msls_calibration_sweep.csv"
    df.to_csv(csv_path, index=False)

    selection = {
        "selection_source": "MSLS-val calibration split only",
        "selection_rule": "max R@1, then max R@5, then max R@10, then smaller candidate_pool, then smaller rho",
        "seed": SEED,
        "calibration_fraction": CALIBRATION_FRACTION,
        "n_total_msls_queries": int(n_queries),
        "n_calibration_queries": int(len(calibration_indices)),
        "n_heldout_queries": int(len(evaluation_indices)),
        "uncertainty_k": int(UNCERTAINTY_K),
        "selected_rho": float(selected["rho"]),
        "selected_candidate_pool": int(selected["candidate_pool"]),
        "calibration_R@1": float(selected["R@1"]),
        "calibration_R@5": float(selected["R@5"]),
        "calibration_R@10": float(selected["R@10"]),
        "calibration_count@1": int(selected["count@1"]),
        "calibration_mean_alpha": float(selected["mean_alpha"]),
    }

    json_path = OUTPUT_DIR / "calibrated_v2_selection.json"

    with open(json_path, "w") as f:
        json.dump(selection, f, indent=2)

    split_path = OUTPUT_DIR / "msls_calibration_split_indices.npz"

    np.savez(
        split_path,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        seed=SEED,
        calibration_fraction=CALIBRATION_FRACTION,
    )

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    print(split_path)

    print("\nTop calibration settings:")
    show = df.sort_values(
        by=["R@1", "R@5", "R@10", "candidate_pool", "rho"],
        ascending=[False, False, False, True, True],
    ).head(12)

    print(
        show[
            [
                "rho",
                "candidate_pool",
                "R@1",
                "R@5",
                "R@10",
                "count@1",
                "mean_alpha",
            ]
        ].to_string(index=False)
    )

    print("\nSELECTED SETTING:")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
