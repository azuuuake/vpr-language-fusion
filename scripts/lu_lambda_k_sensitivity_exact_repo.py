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

LU_LAMBDAS = [0.3, 0.5, 0.7]
LU_K_VALUES = [5, 10, 20]

BASE_SU_K = 10
BASE_RHO = 3.0
BASE_CANDIDATE_POOL = 10


def ensure_msls_split():
    split_path = OUT_DIR / "msls_calibration_split_indices.npz"

    if split_path.exists():
        return split_path

    visual = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    n_queries = visual.shape[0]

    seed = 42
    calibration_fraction = 0.30

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_queries)

    n_calibration = int(round(calibration_fraction * n_queries))
    calibration_indices = np.sort(indices[:n_calibration])
    evaluation_indices = np.sort(indices[n_calibration:])

    np.savez(
        split_path,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        seed=seed,
        calibration_fraction=calibration_fraction,
        n_queries=n_queries,
        n_calibration=len(calibration_indices),
        n_evaluation=len(evaluation_indices),
    )

    return split_path


def recall_at_k(retrieved, positive, k):
    correct = []
    for i in range(retrieved.shape[0]):
        correct.append(positive[i, retrieved[i, :k]].any())
    return 100.0 * np.mean(correct), int(np.sum(correct))


def v2_retrieval_exact_lu_sensitivity(
    visual,
    language,
    rho,
    candidate_pool,
    su_k,
    lu_k,
    lu_lambda,
):
    # SU fixed at original K=10 and lambda=0.5.
    su_idx = np.argsort(-visual, axis=1)[:, :su_k]
    visual_su_topk = np.take_along_axis(visual, su_idx, axis=1)
    su = visual_uncertainty(visual_su_topk, lam=0.5)

    # LU varied with exact repo function.
    lu_idx = np.argsort(-visual, axis=1)[:, :lu_k]
    language_lu_topk = np.take_along_axis(language, lu_idx, axis=1)
    lu = language_uncertainty(language_lu_topk, lam=lu_lambda)

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

    return retrieved, alpha, su, lu


def main():
    split_path = ensure_msls_split()
    split = np.load(split_path)
    eval_indices = split["evaluation_indices"]

    print("EXACT LU sensitivity diagnostic")
    print("rho fixed:", BASE_RHO)
    print("candidate_pool fixed:", BASE_CANDIDATE_POOL)
    print("SU K fixed:", BASE_SU_K)
    print("SU lambda fixed: 0.5")
    print("LU lambdas:", LU_LAMBDAS)
    print("LU K values:", LU_K_VALUES)
    print("Formula: exact uncertainty_v2.language_uncertainty")
    print("Note: this is a sensitivity check, not a hyperparameter search.")

    rows = []

    for key, cfg in DATASETS.items():
        visual = np.load(EMBED_DIR / cfg["visual"])
        language = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual = visual[eval_indices]
            language = language[eval_indices]
            positive = positive[eval_indices]

        # Baseline: lambda=0.5, K=10.
        base_ret, base_alpha, base_su, base_lu = v2_retrieval_exact_lu_sensitivity(
            visual=visual,
            language=language,
            rho=BASE_RHO,
            candidate_pool=BASE_CANDIDATE_POOL,
            su_k=BASE_SU_K,
            lu_k=10,
            lu_lambda=0.5,
        )

        base_r1, base_c1 = recall_at_k(base_ret, positive, 1)
        base_r5, base_c5 = recall_at_k(base_ret, positive, 5)
        base_r10, base_c10 = recall_at_k(base_ret, positive, 10)

        for lu_k in LU_K_VALUES:
            for lu_lambda in LU_LAMBDAS:
                ret, alpha, su, lu = v2_retrieval_exact_lu_sensitivity(
                    visual=visual,
                    language=language,
                    rho=BASE_RHO,
                    candidate_pool=BASE_CANDIDATE_POOL,
                    su_k=BASE_SU_K,
                    lu_k=lu_k,
                    lu_lambda=lu_lambda,
                )

                r1, c1 = recall_at_k(ret, positive, 1)
                r5, c5 = recall_at_k(ret, positive, 5)
                r10, c10 = recall_at_k(ret, positive, 10)

                rows.append({
                    "Dataset": cfg["pretty"],
                    "LU_lambda": lu_lambda,
                    "LU_K": lu_k,
                    "is_baseline_lambda05_K10": bool(lu_lambda == 0.5 and lu_k == 10),
                    "R@1": r1,
                    "R@5": r5,
                    "R@10": r10,
                    "count@1": c1,
                    "count@5": c5,
                    "count@10": c10,
                    "mean_alpha": float(alpha.mean()),
                    "mean_LU": float(lu.mean()),
                    "delta_R@1_vs_lambda05_K10": r1 - base_r1,
                    "delta_R@5_vs_lambda05_K10": r5 - base_r5,
                    "delta_R@10_vs_lambda05_K10": r10 - base_r10,
                })

    df = pd.DataFrame(rows)

    out_path = OUT_DIR / "lu_lambda_k_sensitivity_exact_repo.csv"
    df.to_csv(out_path, index=False)

    print("\nEXACT LU LAMBDA/K SENSITIVITY")
    print(df.to_string(index=False))

    print("\nSENSITIVITY RANGE BY DATASET")
    summary = (
        df
        .groupby("Dataset")
        .agg(
            R1_min=("R@1", "min"),
            R1_max=("R@1", "max"),
            R1_range=("R@1", lambda x: x.max() - x.min()),
            R5_min=("R@5", "min"),
            R5_max=("R@5", "max"),
            R5_range=("R@5", lambda x: x.max() - x.min()),
            R10_min=("R@10", "min"),
            R10_max=("R@10", "max"),
            R10_range=("R@10", lambda x: x.max() - x.min()),
            max_abs_delta_R1=("delta_R@1_vs_lambda05_K10", lambda x: np.max(np.abs(x))),
            max_abs_delta_R5=("delta_R@5_vs_lambda05_K10", lambda x: np.max(np.abs(x))),
            max_abs_delta_R10=("delta_R@10_vs_lambda05_K10", lambda x: np.max(np.abs(x))),
        )
        .reset_index()
    )

    summary_path = OUT_DIR / "lu_lambda_k_sensitivity_exact_repo_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))

    print("\nBaseline rows check:")
    print(df[df["is_baseline_lambda05_K10"]].to_string(index=False))

    print("\nSaved:")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
