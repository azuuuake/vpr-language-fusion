
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


def load_locked_setting():
    path = OUT_DIR / "calibrated_v2_selection_locked_rule.json"
    if path.exists():
        with open(path, "r") as f:
            s = json.load(f)
        return float(s["selected_rho"]), int(s["selected_candidate_pool"]), int(s["uncertainty_k"])
    return 3.0, 10, 10


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


def correct_at_1(retrieved, positive):
    out = np.zeros(retrieved.shape[0], dtype=bool)
    for i in range(retrieved.shape[0]):
        out[i] = positive[i, retrieved[i, 0]]
    return out


def r1(correct):
    return 100.0 * correct.mean()


def compute_su_lu_alpha(visual, language, rho, candidate_pool, uncertainty_k):
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

    return su, lu, alpha, 1.0 - alpha, retrieved


def main():
    rho, candidate_pool, uncertainty_k = load_locked_setting()
    split_path = ensure_msls_split()
    split = np.load(split_path)
    eval_indices = split["evaluation_indices"]

    print("Frozen setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)

    rows = []
    query_rows = []

    for key, cfg in DATASETS.items():
        visual = np.load(EMBED_DIR / cfg["visual"])
        language = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual = visual[eval_indices]
            language = language[eval_indices]
            positive = positive[eval_indices]

        su, lu, alpha, lang_weight, retrieved_v2 = compute_su_lu_alpha(
            visual=visual,
            language=language,
            rho=rho,
            candidate_pool=candidate_pool,
            uncertainty_k=uncertainty_k,
        )

        correct_v2 = correct_at_1(retrieved_v2, positive)

        n = len(su)
        n_low = int(np.ceil(0.10 * n))

        low_su_idx = set(np.argsort(su)[:n_low])
        low_lang_weight_idx = set(np.argsort(lang_weight)[:n_low])
        low_lu_idx = set(np.argsort(lu)[:n_low])

        overlap_su_lang = low_su_idx & low_lang_weight_idx
        overlap_lu_lang = low_lu_idx & low_lang_weight_idx

        low_su_mask = np.array([i in low_su_idx for i in range(n)])
        low_lang_mask = np.array([i in low_lang_weight_idx for i in range(n)])
        overlap_mask = np.array([i in overlap_su_lang for i in range(n)])

        row = {
            "Dataset": cfg["pretty"],
            "n_queries": n,
            "n_low_10pct": n_low,

            "overlap_lowSU_lowLanguageWeight_count": len(overlap_su_lang),
            "overlap_lowSU_lowLanguageWeight_pct_of_bucket": 100.0 * len(overlap_su_lang) / n_low,

            "overlap_lowLU_lowLanguageWeight_count": len(overlap_lu_lang),
            "overlap_lowLU_lowLanguageWeight_pct_of_bucket": 100.0 * len(overlap_lu_lang) / n_low,

            "V2_R@1_full": r1(correct_v2),
            "V2_R@1_low_SU_10pct": r1(correct_v2[low_su_mask]),
            "V2_R@1_low_language_weight_10pct": r1(correct_v2[low_lang_mask]),
            "V2_R@1_overlap_lowSU_lowLanguageWeight": r1(correct_v2[overlap_mask]) if overlap_mask.sum() > 0 else np.nan,

            "mean_SU_full": float(np.mean(su)),
            "mean_SU_low_language_weight_10pct": float(np.mean(su[low_lang_mask])),
            "mean_language_weight_full": float(np.mean(lang_weight)),
            "mean_language_weight_low_SU_10pct": float(np.mean(lang_weight[low_su_mask])),

            "rho": rho,
            "candidate_pool": candidate_pool,
            "uncertainty_k": uncertainty_k,
        }

        rows.append(row)

        for i in range(n):
            query_rows.append({
                "Dataset": cfg["pretty"],
                "query_index": i,
                "SU": float(su[i]),
                "LU": float(lu[i]),
                "alpha_visual_weight": float(alpha[i]),
                "language_weight_1_minus_alpha": float(lang_weight[i]),
                "is_low_SU_10pct": i in low_su_idx,
                "is_low_language_weight_10pct": i in low_lang_weight_idx,
                "is_low_LU_10pct": i in low_lu_idx,
                "V2_correct_at_1": bool(correct_v2[i]),
            })

    summary = pd.DataFrame(rows)
    query_df = pd.DataFrame(query_rows)

    summary_path = OUT_DIR / "su_vs_alpha_overlap_summary.csv"
    query_path = OUT_DIR / "su_vs_alpha_overlap_query_level.csv"

    summary.to_csv(summary_path, index=False)
    query_df.to_csv(query_path, index=False)

    print("\nSU vs 1-alpha overlap summary")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(summary_path)
    print(query_path)


if __name__ == "__main__":
    main()
