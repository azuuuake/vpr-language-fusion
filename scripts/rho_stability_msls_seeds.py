
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/content/vpr-language-fusion")
sys.path.insert(0, str(REPO_ROOT / "src"))

from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    precision_weighted_fusion,
)


EMBED_DIR = Path("/content/drive/MyDrive/vpr_research/embeddings")
OUT_DIR = REPO_ROOT / "results" / "redesign_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 7, 21, 42, 100]
CALIBRATION_FRACTION = 0.30

RHO_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
CANDIDATE_POOL_GRID = [10, 20, 50, 100]
UNCERTAINTY_K = 10


def split_indices(n_queries, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_queries)

    n_cal = int(round(CALIBRATION_FRACTION * n_queries))
    cal_idx = np.sort(idx[:n_cal])
    eval_idx = np.sort(idx[n_cal:])

    return cal_idx, eval_idx


def recall_at_k(retrieved, positive, k):
    correct = []
    for i in range(retrieved.shape[0]):
        correct.append(positive[i, retrieved[i, :k]].any())
    return 100.0 * np.mean(correct), int(np.sum(correct))


def v2_retrieval(visual, language, rho, candidate_pool, uncertainty_k):
    topk_idx = np.argsort(-visual, axis=1)[:, :uncertainty_k]

    visual_topk = np.take_along_axis(visual, topk_idx, axis=1)
    language_topk = np.take_along_axis(language, topk_idx, axis=1)

    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    pool_idx = np.argsort(-visual, axis=1)[:, :candidate_pool]

    visual_pool = np.take_along_axis(visual, pool_idx, axis=1)
    language_pool = np.take_along_axis(language, pool_idx, axis=1)

    fused_pool, alpha = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    order = np.argsort(-fused_pool, axis=1)
    retrieved = np.take_along_axis(pool_idx, order, axis=1)

    return retrieved, float(alpha.mean())


def evaluate_setting(visual, language, positive, rho, candidate_pool, uncertainty_k):
    retrieved, mean_alpha = v2_retrieval(
        visual=visual,
        language=language,
        rho=rho,
        candidate_pool=candidate_pool,
        uncertainty_k=uncertainty_k,
    )

    r1, c1 = recall_at_k(retrieved, positive, 1)
    r5, c5 = recall_at_k(retrieved, positive, 5)
    r10, c10 = recall_at_k(retrieved, positive, 10)

    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "count@1": c1,
        "count@5": c5,
        "count@10": c10,
        "mean_alpha": mean_alpha,
    }


def main():
    visual = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    language = np.load(EMBED_DIR / "msls_val_lang_sim_matrix.npy")
    positive = np.load(EMBED_DIR / "msls_val_positive_matrix.npy").astype(bool)

    n_queries = visual.shape[0]

    print("MSLS total queries:", n_queries)
    print("Calibration fraction:", CALIBRATION_FRACTION)
    print("Seeds:", SEEDS)
    print("Rho grid:", RHO_GRID)
    print("Candidate pool grid:", CANDIDATE_POOL_GRID)
    print("Uncertainty K:", UNCERTAINTY_K)
    print("Selection rule: max calibration R@1, then smaller candidate_pool, then smaller rho")

    all_grid_rows = []
    selected_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print("Seed:", seed)
        print("=" * 90)

        cal_idx, eval_idx = split_indices(n_queries, seed)

        visual_cal = visual[cal_idx]
        language_cal = language[cal_idx]
        positive_cal = positive[cal_idx]

        visual_eval = visual[eval_idx]
        language_eval = language[eval_idx]
        positive_eval = positive[eval_idx]

        grid_rows = []

        for candidate_pool in CANDIDATE_POOL_GRID:
            for rho in RHO_GRID:
                cal_metrics = evaluate_setting(
                    visual=visual_cal,
                    language=language_cal,
                    positive=positive_cal,
                    rho=rho,
                    candidate_pool=candidate_pool,
                    uncertainty_k=UNCERTAINTY_K,
                )

                row = {
                    "seed": seed,
                    "split": "calibration",
                    "n_calibration": len(cal_idx),
                    "n_evaluation": len(eval_idx),
                    "rho": rho,
                    "candidate_pool": candidate_pool,
                    "uncertainty_k": UNCERTAINTY_K,
                    **cal_metrics,
                }

                grid_rows.append(row)
                all_grid_rows.append(row)

        grid_df = pd.DataFrame(grid_rows)

        selected = (
            grid_df
            .sort_values(
                by=["R@1", "candidate_pool", "rho"],
                ascending=[False, True, True],
            )
            .iloc[0]
        )

        selected_rho = float(selected["rho"])
        selected_pool = int(selected["candidate_pool"])

        eval_metrics = evaluate_setting(
            visual=visual_eval,
            language=language_eval,
            positive=positive_eval,
            rho=selected_rho,
            candidate_pool=selected_pool,
            uncertainty_k=UNCERTAINTY_K,
        )

        selected_row = {
            "seed": seed,
            "n_calibration": len(cal_idx),
            "n_evaluation": len(eval_idx),
            "selected_rho": selected_rho,
            "selected_candidate_pool": selected_pool,
            "uncertainty_k": UNCERTAINTY_K,
            "calibration_R@1": float(selected["R@1"]),
            "calibration_R@5": float(selected["R@5"]),
            "calibration_R@10": float(selected["R@10"]),
            "heldout_R@1": eval_metrics["R@1"],
            "heldout_R@5": eval_metrics["R@5"],
            "heldout_R@10": eval_metrics["R@10"],
            "heldout_count@1": eval_metrics["count@1"],
            "heldout_count@5": eval_metrics["count@5"],
            "heldout_count@10": eval_metrics["count@10"],
            "heldout_mean_alpha": eval_metrics["mean_alpha"],
        }

        selected_rows.append(selected_row)

        print("Selected rho:", selected_rho)
        print("Selected candidate_pool:", selected_pool)
        print(
            "Calibration R@1/R@5/R@10:",
            selected_row["calibration_R@1"],
            selected_row["calibration_R@5"],
            selected_row["calibration_R@10"],
        )
        print(
            "Held-out R@1/R@5/R@10:",
            selected_row["heldout_R@1"],
            selected_row["heldout_R@5"],
            selected_row["heldout_R@10"],
        )

    all_grid_df = pd.DataFrame(all_grid_rows)
    selected_df = pd.DataFrame(selected_rows)

    grid_path = OUT_DIR / "rho_stability_grid_all_seeds.csv"
    selected_path = OUT_DIR / "rho_stability_selected_by_seed.csv"

    all_grid_df.to_csv(grid_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    print("\n" + "=" * 90)
    print("RHO STABILITY SUMMARY")
    print("=" * 90)
    print(selected_df.to_string(index=False))

    print("\nSelected rho counts:")
    print(selected_df["selected_rho"].value_counts().sort_index().to_string())

    print("\nSelected candidate_pool counts:")
    print(selected_df["selected_candidate_pool"].value_counts().sort_index().to_string())

    print("\nHeld-out performance summary:")
    print(selected_df[["heldout_R@1", "heldout_R@5", "heldout_R@10"]].describe().to_string())

    print("\nSaved:")
    print(grid_path)
    print(selected_path)


if __name__ == "__main__":
    main()
