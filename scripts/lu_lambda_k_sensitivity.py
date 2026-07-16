
import json
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


def ratio_spread(scores):
    """
    Higher value = more uncertain / less separation.
    Uses top-1/top-2 ratio-style margin after descending sort.
    """
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    s1 = sorted_scores[:, 0]
    s2 = sorted_scores[:, 1]
    eps = 1e-12

    # If top1 and top2 are very close, uncertainty is high.
    margin_ratio = (s1 - s2) / (np.abs(s1) + eps)
    return 1.0 - np.clip(margin_ratio, 0.0, 1.0)


def similarity_distribution(scores):
    """
    Higher value = more uncertain / flatter distribution.
    Uses normalised entropy over softmaxed scores.
    """
    x = scores - np.max(scores, axis=1, keepdims=True)
    exp_x = np.exp(x)
    p = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)

    entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
    max_entropy = np.log(scores.shape[1])

    return entropy / max_entropy


def custom_uncertainty(scores, lam):
    rs = ratio_spread(scores)
    sd = similarity_distribution(scores)
    return lam * rs + (1.0 - lam) * sd


def v2_retrieval_lu_sensitivity(
    visual,
    language,
    rho,
    candidate_pool,
    su_k,
    lu_k,
    lu_lambda,
):
    # SU kept fixed using the original implementation and base K=10.
    su_idx = np.argsort(-visual, axis=1)[:, :su_k]
    visual_su_topk = np.take_along_axis(visual, su_idx, axis=1)
    su = visual_uncertainty(visual_su_topk)

    # LU varied by K and lambda.
    lu_idx = np.argsort(-visual, axis=1)[:, :lu_k]
    language_lu_topk = np.take_along_axis(language, lu_idx, axis=1)
    lu = custom_uncertainty(language_lu_topk, lam=lu_lambda)

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


def v2_retrieval_baseline(visual, language, rho, candidate_pool, uncertainty_k):
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

    return retrieved, alpha, su, lu


def main():
    split_path = ensure_msls_split()
    split = np.load(split_path)
    eval_indices = split["evaluation_indices"]

    print("LU sensitivity diagnostic")
    print("rho fixed:", BASE_RHO)
    print("candidate_pool fixed:", BASE_CANDIDATE_POOL)
    print("SU K fixed:", BASE_SU_K)
    print("LU lambdas:", LU_LAMBDAS)
    print("LU K values:", LU_K_VALUES)
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

        # Baseline using exact repo implementation
        base_ret, base_alpha, base_su, base_lu = v2_retrieval_baseline(
            visual=visual,
            language=language,
            rho=BASE_RHO,
            candidate_pool=BASE_CANDIDATE_POOL,
            uncertainty_k=BASE_SU_K,
        )

        base_r1, base_c1 = recall_at_k(base_ret, positive, 1)
        base_r5, base_c5 = recall_at_k(base_ret, positive, 5)
        base_r10, base_c10 = recall_at_k(base_ret, positive, 10)

        rows.append({
            "Dataset": cfg["pretty"],
            "Variant": "baseline_repo_LU",
            "LU_lambda": 0.5,
            "LU_K": BASE_SU_K,
            "R@1": base_r1,
            "R@5": base_r5,
            "R@10": base_r10,
            "count@1": base_c1,
            "count@5": base_c5,
            "count@10": base_c10,
            "mean_alpha": float(base_alpha.mean()),
            "mean_LU": float(base_lu.mean()),
            "delta_R@1_vs_baseline": 0.0,
            "delta_R@5_vs_baseline": 0.0,
            "delta_R@10_vs_baseline": 0.0,
        })

        for lu_k in LU_K_VALUES:
            for lu_lambda in LU_LAMBDAS:
                ret, alpha, su, lu = v2_retrieval_lu_sensitivity(
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
                    "Variant": "custom_LU_sensitivity",
                    "LU_lambda": lu_lambda,
                    "LU_K": lu_k,
                    "R@1": r1,
                    "R@5": r5,
                    "R@10": r10,
                    "count@1": c1,
                    "count@5": c5,
                    "count@10": c10,
                    "mean_alpha": float(alpha.mean()),
                    "mean_LU": float(lu.mean()),
                    "delta_R@1_vs_baseline": r1 - base_r1,
                    "delta_R@5_vs_baseline": r5 - base_r5,
                    "delta_R@10_vs_baseline": r10 - base_r10,
                })

    df = pd.DataFrame(rows)

    out_path = OUT_DIR / "lu_lambda_k_sensitivity.csv"
    df.to_csv(out_path, index=False)

    print("\nLU LAMBDA/K SENSITIVITY")
    print(df.to_string(index=False))

    print("\nSENSITIVITY RANGE BY DATASET")
    sens = df[df["Variant"] == "custom_LU_sensitivity"].copy()

    summary = (
        sens
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
            max_abs_delta_R1=("delta_R@1_vs_baseline", lambda x: np.max(np.abs(x))),
            max_abs_delta_R5=("delta_R@5_vs_baseline", lambda x: np.max(np.abs(x))),
            max_abs_delta_R10=("delta_R@10_vs_baseline", lambda x: np.max(np.abs(x))),
        )
        .reset_index()
    )

    summary_path = OUT_DIR / "lu_lambda_k_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))

    print("\nSaved:")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
