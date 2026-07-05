"""
bootstrap_v2_final_pool20.py

Final paired bootstrap for redesigned V2 method.

Final V2 setting:
    uncertainty_k = 10
    candidate_pool = 20
    rho = 6.0

Compares:
    V2 vs B1 visual-only
    V2 vs B2 fixed fusion
    V2 vs V1 sigmoid+threshold
"""

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

DATASETS = {
    "amstertime": {
        "pretty": "AmsterTime",
        "visual": "amstertime_visual_sim_matrix.npy",
        "language": "amstertime_lang_sim_matrix.npy",
        "positive": "amstertime_positive_matrix.npy",
    },
    "msls_val": {
        "pretty": "MSLS-val",
        "visual": "msls_val_visual_sim_matrix.npy",
        "language": "msls_val_lang_sim_matrix.npy",
        "positive": "msls_val_positive_matrix.npy",
    },
    "nordland_clean": {
        "pretty": "Nordland-clean",
        "visual": "nordland_clean_visual_sim_matrix.npy",
        "language": "nordland_clean_lang_sim_matrix.npy",
        "positive": "nordland_clean_positive_matrix.npy",
    },
}

UNCERTAINTY_K = 10
CANDIDATE_POOL = 20
RHO = 6.0
BOOTSTRAPS = 5000
SEED = 42


def correct_at_k(retrieved_indices, positive, k):
    n_queries = retrieved_indices.shape[0]
    correct = np.zeros(n_queries, dtype=bool)

    for i in range(n_queries):
        correct[i] = positive[i, retrieved_indices[i, :k]].any()

    return correct


def bootstrap_diff(a_correct, b_correct, n_boot=BOOTSTRAPS, seed=SEED):
    """
    Paired bootstrap difference in recall percentage points.
    Difference = A - B
    """
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


def build_retrievals(visual_sim, language_sim):
    # B1 visual-only
    retrieved_b1 = np.argsort(-visual_sim, axis=1)

    # B2 fixed fusion over full database
    fixed_sim = 0.5 * visual_sim + 0.5 * language_sim
    retrieved_b2 = np.argsort(-fixed_sim, axis=1)

    # Top-10 visual candidates for V1
    top10_idx = np.argsort(-visual_sim, axis=1)[:, :UNCERTAINTY_K]
    visual_top10 = np.take_along_axis(visual_sim, top10_idx, axis=1)
    language_top10 = np.take_along_axis(language_sim, top10_idx, axis=1)

    # V1 old sigmoid+threshold method
    visual_top10_v1 = old_sort_scores_desc(visual_top10)
    su_norm_v1 = old_SU_normalised(visual_top10_v1)

    fused_v1_top10, alpha_v1 = old_confidence_gated_fusion(
        visual_sims=visual_top10_v1,
        lang_sims=language_top10,
        su_norm=su_norm_v1,
        tau=1.0,
        lang_threshold=0.05,
    )

    order_v1 = np.argsort(-fused_v1_top10, axis=1)
    retrieved_v1 = np.take_along_axis(top10_idx, order_v1, axis=1)

    # V2 final method:
    # uncertainty from top-10, reranking over top-20 candidate pool
    su = visual_uncertainty(visual_top10)
    lu = language_uncertainty(language_top10)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    pool_idx = np.argsort(-visual_sim, axis=1)[:, :CANDIDATE_POOL]
    visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

    fused_v2_pool, alpha_v2 = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=RHO,
    )

    order_v2 = np.argsort(-fused_v2_pool, axis=1)
    retrieved_v2 = np.take_along_axis(pool_idx, order_v2, axis=1)

    return {
        "B1 visual-only": retrieved_b1,
        "B2 fixed fusion": retrieved_b2,
        "V1 sigmoid+threshold": retrieved_v1,
        "V2 conservative precision": retrieved_v2,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    comparisons = [
        ("V2 conservative precision", "B1 visual-only"),
        ("V2 conservative precision", "B2 fixed fusion"),
        ("V2 conservative precision", "V1 sigmoid+threshold"),
    ]

    for dataset_key, cfg in DATASETS.items():
        pretty = cfg["pretty"]

        print("=" * 80)
        print(pretty)
        print("=" * 80)

        visual_sim = np.load(EMBED_DIR / cfg["visual"])
        language_sim = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        retrievals = build_retrievals(visual_sim, language_sim)

        correct = {}

        for method, retrieved in retrievals.items():
            correct[method] = {}
            for k in [1, 5, 10]:
                correct[method][k] = correct_at_k(retrieved, positive, k)

        for k in [1, 5, 10]:
            for a, b in comparisons:
                point, low, high = bootstrap_diff(
                    correct[a][k],
                    correct[b][k],
                    n_boot=BOOTSTRAPS,
                    seed=SEED,
                )

                rows.append({
                    "Dataset": pretty,
                    "Metric": f"R@{k}",
                    "Comparison": f"{a} - {b}",
                    "Difference_pp": point,
                    "CI95_low": low,
                    "CI95_high": high,
                    "Significant": (low > 0) or (high < 0),
                    "rho": RHO,
                    "uncertainty_k": UNCERTAINTY_K,
                    "candidate_pool": CANDIDATE_POOL,
                    "n_boot": BOOTSTRAPS,
                })

    df = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "bootstrap_v2_final_pool20.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
