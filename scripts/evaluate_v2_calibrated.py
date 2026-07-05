"""
evaluate_v2_calibrated.py

Final evaluation using calibration-selected V2 hyperparameters.

Hyperparameters are loaded from:
    results/redesign_v2/calibrated_v2_selection.json

MSLS-val:
    evaluate only on held-out evaluation queries.

AmsterTime and Nordland-clean:
    evaluate full datasets.

Outputs:
    results/redesign_v2/calibrated_final_results.csv
    results/redesign_v2/calibrated_final_bootstrap.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from su_metric import (
    sort_scores_desc as old_sort_scores_desc,
    SU_normalised as old_SU_normalised,
    confidence_gated_fusion as old_confidence_gated_fusion,
)

from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    precision_weighted_fusion,
)


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

BOOTSTRAPS = 5000
SEED = 42


DATASETS = {
    "amstertime": {
        "pretty": "AmsterTime",
        "split": "full",
        "visual": "amstertime_visual_sim_matrix.npy",
        "language": "amstertime_lang_sim_matrix.npy",
        "positive": "amstertime_positive_matrix.npy",
    },
    "msls_val_heldout": {
        "pretty": "MSLS-val held-out",
        "split": "heldout",
        "visual": "msls_val_visual_sim_matrix.npy",
        "language": "msls_val_lang_sim_matrix.npy",
        "positive": "msls_val_positive_matrix.npy",
    },
    "nordland_clean": {
        "pretty": "Nordland-clean",
        "split": "full",
        "visual": "nordland_clean_visual_sim_matrix.npy",
        "language": "nordland_clean_lang_sim_matrix.npy",
        "positive": "nordland_clean_positive_matrix.npy",
    },
}


def correct_at_k(retrieved_indices, positive, k):
    n_queries = retrieved_indices.shape[0]
    correct = np.zeros(n_queries, dtype=bool)

    for i in range(n_queries):
        correct[i] = positive[i, retrieved_indices[i, :k]].any()

    return correct


def recall_percent(correct):
    return 100.0 * correct.mean(), int(correct.sum())


def bootstrap_diff(a_correct, b_correct, n_boot=BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a_correct)

    a = a_correct.astype(float)
    b = b_correct.astype(float)

    point = 100.0 * (a.mean() - b.mean())

    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = 100.0 * (a[idx].mean() - b[idx].mean())

    low, high = np.percentile(diffs, [2.5, 97.5])

    return point, low, high


def build_retrievals(
    visual_sim,
    language_sim,
    uncertainty_k,
    candidate_pool,
    rho,
):
    # B1 visual-only
    retrieved_b1 = np.argsort(-visual_sim, axis=1)

    # B2 fixed fusion over full database
    fixed_sim = 0.5 * visual_sim + 0.5 * language_sim
    retrieved_b2 = np.argsort(-fixed_sim, axis=1)

    # Top-k visual candidates for uncertainty and V1
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :uncertainty_k]
    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

    # V1 old sigmoid + threshold
    visual_topk_v1 = old_sort_scores_desc(visual_topk)
    su_norm_v1 = old_SU_normalised(visual_topk_v1)

    fused_v1_topk, alpha_v1 = old_confidence_gated_fusion(
        visual_sims=visual_topk_v1,
        lang_sims=language_topk,
        su_norm=su_norm_v1,
        tau=1.0,
        lang_threshold=0.05,
    )

    order_v1 = np.argsort(-fused_v1_topk, axis=1)
    retrieved_v1 = np.take_along_axis(topk_idx, order_v1, axis=1)

    # V2 calibrated method
    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    pool_idx = np.argsort(-visual_sim, axis=1)[:, :candidate_pool]
    visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

    fused_v2_pool, alpha_v2 = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    order_v2 = np.argsort(-fused_v2_pool, axis=1)
    retrieved_v2 = np.take_along_axis(pool_idx, order_v2, axis=1)

    return {
        "B1 visual-only": {
            "retrieved": retrieved_b1,
            "mean_alpha": None,
        },
        "B2 fixed fusion": {
            "retrieved": retrieved_b2,
            "mean_alpha": 0.5,
        },
        "V1 sigmoid+threshold": {
            "retrieved": retrieved_v1,
            "mean_alpha": float(alpha_v1.mean()),
        },
        "V2 calibrated precision": {
            "retrieved": retrieved_v2,
            "mean_alpha": float(alpha_v2.mean()),
            "mean_SU": float(su.mean()),
            "mean_LU": float(lu.mean()),
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection_path = OUTPUT_DIR / "calibrated_v2_selection_locked_rule.json"
    split_path = OUTPUT_DIR / "msls_calibration_split_indices.npz"

    with open(selection_path, "r") as f:
        selection = json.load(f)

    rho = float(selection["selected_rho"])
    candidate_pool = int(selection["selected_candidate_pool"])
    uncertainty_k = int(selection["uncertainty_k"])

    split_data = np.load(split_path)
    msls_eval_indices = split_data["evaluation_indices"]

    print("Using frozen calibrated setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)
    print("MSLS held-out queries:", len(msls_eval_indices))

    result_rows = []
    bootstrap_rows = []

    comparisons = [
        ("V2 calibrated precision", "B1 visual-only"),
        ("V2 calibrated precision", "B2 fixed fusion"),
        ("V2 calibrated precision", "V1 sigmoid+threshold"),
    ]

    for dataset_key, cfg in DATASETS.items():
        pretty = cfg["pretty"]
        split = cfg["split"]

        print("\n" + "=" * 80)
        print(pretty)
        print("=" * 80)

        visual_sim = np.load(EMBED_DIR / cfg["visual"])
        language_sim = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if split == "heldout":
            visual_sim = visual_sim[msls_eval_indices]
            language_sim = language_sim[msls_eval_indices]
            positive = positive[msls_eval_indices]

        retrievals = build_retrievals(
            visual_sim=visual_sim,
            language_sim=language_sim,
            uncertainty_k=uncertainty_k,
            candidate_pool=candidate_pool,
            rho=rho,
        )

        correct = {}

        for method, obj in retrievals.items():
            correct[method] = {}

            row = {
                "Dataset": pretty,
                "Split": split,
                "Method": method,
                "n_queries": visual_sim.shape[0],
                "rho": rho if method == "V2 calibrated precision" else None,
                "uncertainty_k": uncertainty_k if method == "V2 calibrated precision" else None,
                "candidate_pool": candidate_pool if method == "V2 calibrated precision" else None,
                "mean_alpha": obj.get("mean_alpha"),
                "mean_SU": obj.get("mean_SU"),
                "mean_LU": obj.get("mean_LU"),
            }

            for k in [1, 5, 10]:
                c = correct_at_k(obj["retrieved"], positive, k)
                correct[method][k] = c
                recall, count = recall_percent(c)
                row[f"R@{k}"] = recall
                row[f"count@{k}"] = count

            result_rows.append(row)

        for k in [1, 5, 10]:
            for a, b in comparisons:
                point, low, high = bootstrap_diff(correct[a][k], correct[b][k])

                bootstrap_rows.append({
                    "Dataset": pretty,
                    "Split": split,
                    "Metric": f"R@{k}",
                    "Comparison": f"{a} - {b}",
                    "Difference_pp": point,
                    "CI95_low": low,
                    "CI95_high": high,
                    "Significant": (low > 0) or (high < 0),
                    "rho": rho,
                    "uncertainty_k": uncertainty_k,
                    "candidate_pool": candidate_pool,
                    "n_boot": BOOTSTRAPS,
                })

    results_df = pd.DataFrame(result_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    results_path = OUTPUT_DIR / "calibrated_final_results.csv"
    bootstrap_path = OUTPUT_DIR / "calibrated_final_bootstrap.csv"

    results_df.to_csv(results_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(bootstrap_path)

    print("\nFINAL RESULTS")
    print(
        results_df[
            [
                "Dataset",
                "Split",
                "Method",
                "R@1",
                "R@5",
                "R@10",
                "count@1",
                "mean_alpha",
            ]
        ].to_string(index=False)
    )

    print("\nBOOTSTRAP")
    print(bootstrap_df.to_string(index=False))


if __name__ == "__main__":
    main()
