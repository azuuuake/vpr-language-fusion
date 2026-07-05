
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


def correct_at_1(retrieved_indices, positive):
    correct = np.zeros(retrieved_indices.shape[0], dtype=bool)

    for i in range(retrieved_indices.shape[0]):
        correct[i] = positive[i, retrieved_indices[i, 0]]

    return correct


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

    print("Using locked calibrated setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)

    decile_rows = []
    query_rows = []

    for key, cfg in DATASETS.items():
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

        topk_idx = np.argsort(-visual_sim, axis=1)[:, :uncertainty_k]
        visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
        language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

        # Visual-only within top-k
        visual_order = np.argsort(-visual_topk, axis=1)
        visual_retrieved = np.take_along_axis(topk_idx, visual_order, axis=1)
        visual_correct = correct_at_1(visual_retrieved, positive)

        # Language-only reranking within visual top-k
        language_order = np.argsort(-language_topk, axis=1)
        language_retrieved = np.take_along_axis(topk_idx, language_order, axis=1)
        language_correct = correct_at_1(language_retrieved, positive)

        # V2 fusion
        su = visual_uncertainty(visual_topk)
        lu = language_uncertainty(language_topk)

        su_norm = normalise_uncertainty(su)
        lu_norm = normalise_uncertainty(lu)

        pool_idx = np.argsort(-visual_sim, axis=1)[:, :candidate_pool]
        visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
        language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

        fused_pool, alpha = precision_weighted_fusion(
            visual_sims=visual_pool,
            language_sims=language_pool,
            su_norm=su_norm,
            lu_norm=lu_norm,
            rho=rho,
        )

        v2_order = np.argsort(-fused_pool, axis=1)
        v2_retrieved = np.take_along_axis(pool_idx, v2_order, axis=1)
        v2_correct = correct_at_1(v2_retrieved, positive)

        df = pd.DataFrame({
            "Dataset": pretty,
            "Split": split,
            "query_id": np.arange(len(lu)),
            "LU": lu,
            "SU": su,
            "alpha": alpha,
            "visual_correct@1": visual_correct,
            "language_correct@1": language_correct,
            "v2_correct@1": v2_correct,
        })

        # Decile 1 = lowest LU = most confident language
        # Decile 10 = highest LU = most uncertain language
        df["LU_decile"] = pd.qcut(
            df["LU"].rank(method="first"),
            q=10,
            labels=False,
        ) + 1

        for decile in range(1, 11):
            sub = df[df["LU_decile"] == decile]

            visual_r1 = 100.0 * sub["visual_correct@1"].mean()
            language_r1 = 100.0 * sub["language_correct@1"].mean()
            v2_r1 = 100.0 * sub["v2_correct@1"].mean()

            rescued = int((~sub["visual_correct@1"] & sub["v2_correct@1"]).sum())
            broken = int((sub["visual_correct@1"] & ~sub["v2_correct@1"]).sum())

            decile_rows.append({
                "Dataset": pretty,
                "Split": split,
                "LU_decile": decile,
                "n_queries": len(sub),
                "mean_LU": float(sub["LU"].mean()),
                "mean_alpha": float(sub["alpha"].mean()),
                "visual_R@1": visual_r1,
                "language_only_R@1": language_r1,
                "v2_R@1": v2_r1,
                "v2_gain_vs_visual_pp": v2_r1 - visual_r1,
                "rescued_by_v2": rescued,
                "broken_by_v2": broken,
                "net_rescue": rescued - broken,
            })

        query_rows.append(df)

    decile_df = pd.DataFrame(decile_rows)
    query_df = pd.concat(query_rows, ignore_index=True)

    decile_path = OUTPUT_DIR / "lu_decile_validation.csv"
    query_path = OUTPUT_DIR / "lu_query_level_validation.csv"

    decile_df.to_csv(decile_path, index=False)
    query_df.to_csv(query_path, index=False)

    print("\nSaved:")
    print(decile_path)
    print(query_path)

    for dataset in decile_df["Dataset"].unique():
        print("\n" + "=" * 80)
        print(dataset)
        print("=" * 80)

        show = decile_df[decile_df["Dataset"] == dataset][
            [
                "LU_decile",
                "n_queries",
                "mean_LU",
                "mean_alpha",
                "visual_R@1",
                "language_only_R@1",
                "v2_R@1",
                "v2_gain_vs_visual_pp",
                "rescued_by_v2",
                "broken_by_v2",
                "net_rescue",
            ]
        ]

        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
