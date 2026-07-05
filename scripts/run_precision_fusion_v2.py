"""
run_precision_fusion_v2.py

Runs the redesigned V2 method:

    Visual uncertainty: SU
    Language uncertainty: LU
    Fusion: precision-weighted Bayesian-style fusion

This script compares:

    B1 Visual-only
    B2 Fixed fusion
    V1 Old sigmoid + hard language threshold gate
    V2 Precision-weighted fusion

Outputs:
    results/redesign_v2/precision_fusion_v2_results_<dataset>.csv
    results/redesign_v2/per_query_<dataset>.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

# V1 old method imports
from su_metric import (
    sort_scores_desc as old_sort_scores_desc,
    SU_normalised as old_SU_normalised,
    confidence_gated_fusion as old_confidence_gated_fusion,
)

# V2 new method imports
from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    precision_weighted_fusion,
)


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"


FILE_MAP = {
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


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def compute_recall_from_indices(
    retrieved_indices: np.ndarray,
    positive_matrix: np.ndarray,
    k: int,
) -> tuple[float, int, np.ndarray]:
    """
    Compute recall@k from retrieved indices.

    Returns:
        recall percentage
        correct count
        per-query boolean correctness
    """
    n_queries = retrieved_indices.shape[0]
    correct = np.zeros(n_queries, dtype=bool)

    for i in range(n_queries):
        correct[i] = positive_matrix[i, retrieved_indices[i, :k]].any()

    count = int(correct.sum())
    recall = 100.0 * count / n_queries

    return recall, count, correct


def compute_recall_from_sim(
    sim_matrix: np.ndarray,
    positive_matrix: np.ndarray,
    k: int,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """
    Compute recall@k directly from a full similarity matrix.

    Returns:
        recall percentage
        correct count
        per-query boolean correctness
        retrieved indices
    """
    retrieved = np.argsort(-sim_matrix, axis=1)

    recall, count, correct = compute_recall_from_indices(
        retrieved_indices=retrieved,
        positive_matrix=positive_matrix,
        k=k,
    )

    return recall, count, correct, retrieved


def compute_r1_r5_r10_from_indices(
    retrieved_indices: np.ndarray,
    positive_matrix: np.ndarray,
) -> dict:
    out = {}

    for k in [1, 5, 10]:
        recall, count, correct = compute_recall_from_indices(
            retrieved_indices=retrieved_indices,
            positive_matrix=positive_matrix,
            k=k,
        )
        out[f"R@{k}"] = recall
        out[f"count@{k}"] = count
        if k == 1:
            out["correct@1"] = correct

    return out


def compute_r1_r5_r10_from_sim(
    sim_matrix: np.ndarray,
    positive_matrix: np.ndarray,
) -> dict:
    retrieved = np.argsort(-sim_matrix, axis=1)
    out = compute_r1_r5_r10_from_indices(retrieved, positive_matrix)
    out["retrieved"] = retrieved
    return out


def run_dataset(
    dataset: str,
    top_k: int = 10,
    candidate_pool: int = 20,
    old_tau: float = 1.0,
    old_lang_threshold: float = 0.05,
    fixed_alpha: float = 0.5,
    rho: float = 6.0,
) -> None:
    if dataset not in FILE_MAP:
        raise ValueError(f"Unknown dataset: {dataset}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = FILE_MAP[dataset]
    pretty = cfg["pretty"]

    print("=" * 80)
    print(f"Dataset: {pretty}")
    print("=" * 80)

    visual_path = EMBED_DIR / cfg["visual"]
    language_path = EMBED_DIR / cfg["language"]
    positive_path = EMBED_DIR / cfg["positive"]

    visual_sim = load_matrix(visual_path)
    language_sim = load_matrix(language_path)
    positive = load_matrix(positive_path).astype(bool)

    if visual_sim.shape != language_sim.shape:
        raise ValueError("Visual and language matrices have different shapes.")

    if visual_sim.shape != positive.shape:
        raise ValueError("Visual and positive matrices have different shapes.")

    n_queries = visual_sim.shape[0]

    print("Visual sim shape:  ", visual_sim.shape)
    print("Language sim shape:", language_sim.shape)
    print("Positive shape:    ", positive.shape)

    # ---------------------------------------------------------------------
    # B1 visual-only
    # ---------------------------------------------------------------------
    b1 = compute_r1_r5_r10_from_sim(visual_sim, positive)

    # ---------------------------------------------------------------------
    # B2 fixed fusion
    # ---------------------------------------------------------------------
    fixed_sim = fixed_alpha * visual_sim + (1.0 - fixed_alpha) * language_sim
    b2 = compute_r1_r5_r10_from_sim(fixed_sim, positive)

    # ---------------------------------------------------------------------
    # Select visual candidates.
    #
    # SU/LU are computed from top_k=10 candidates for consistency with V1.
    # V2 then reranks a wider candidate_pool=20 to allow R@10 improvement.
    # ---------------------------------------------------------------------
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :top_k]
    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

    pool_idx = np.argsort(-visual_sim, axis=1)[:, :candidate_pool]
    visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

    # ---------------------------------------------------------------------
    # V1 old method: sigmoid + hard language threshold gate
    # ---------------------------------------------------------------------
    visual_topk_v1 = old_sort_scores_desc(visual_topk)
    su_norm_v1 = old_SU_normalised(visual_topk_v1)

    fused_v1_topk, alpha_v1 = old_confidence_gated_fusion(
        visual_sims=visual_topk_v1,
        lang_sims=language_topk,
        su_norm=su_norm_v1,
        tau=old_tau,
        lang_threshold=old_lang_threshold,
    )

    rerank_v1_order = np.argsort(-fused_v1_topk, axis=1)
    retrieved_v1 = np.take_along_axis(topk_idx, rerank_v1_order, axis=1)
    v1 = compute_r1_r5_r10_from_indices(retrieved_v1, positive)

    # ---------------------------------------------------------------------
    # V2 new method: SU + LU + precision-weighted fusion
    # ---------------------------------------------------------------------
    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    fused_v2_pool, alpha_v2 = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    rerank_v2_order = np.argsort(-fused_v2_pool, axis=1)
    retrieved_v2 = np.take_along_axis(pool_idx, rerank_v2_order, axis=1)
    v2 = compute_r1_r5_r10_from_indices(retrieved_v2, positive)

    # ---------------------------------------------------------------------
    # Build result table
    # ---------------------------------------------------------------------
    rows = []

    def add_row(method_name: str, result: dict, extra: dict | None = None):
        row = {
            "Dataset": pretty,
            "Method": method_name,
            "R@1": result["R@1"],
            "R@5": result["R@5"],
            "R@10": result["R@10"],
            "count@1": result["count@1"],
            "count@5": result["count@5"],
            "count@10": result["count@10"],
            "n_queries": n_queries,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    add_row("B1 visual-only", b1)
    add_row("B2 fixed fusion", b2, {"fixed_alpha": fixed_alpha})

    add_row(
        "V1 sigmoid+threshold",
        v1,
        {
            "old_tau": old_tau,
            "old_lang_threshold": old_lang_threshold,
            "language_active": int((alpha_v1 < 1.0).sum()),
            "mean_alpha": float(alpha_v1.mean()),
            "std_alpha": float(alpha_v1.std()),
        },
    )

    add_row(
        "V2 conservative precision",
        v2,
        {
            "language_active": int((alpha_v2 < 0.999).sum()),
            "mean_alpha": float(alpha_v2.mean()),
            "std_alpha": float(alpha_v2.std()),
            "rho": rho,
            "candidate_pool": candidate_pool,
            "uncertainty_k": top_k,
            "mean_SU": float(su.mean()),
            "std_SU": float(su.std()),
            "mean_LU": float(lu.mean()),
            "std_LU": float(lu.std()),
        },
    )

    df = pd.DataFrame(rows)

    print()
    print("RESULTS")
    print("-" * 80)
    print(
        df[
            [
                "Dataset",
                "Method",
                "R@1",
                "R@5",
                "R@10",
                "count@1",
                "language_active",
                "mean_alpha",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------------------
    # Save CSV and per-query arrays
    # ---------------------------------------------------------------------
    csv_path = OUTPUT_DIR / f"precision_fusion_v2_results_{dataset}.csv"
    df.to_csv(csv_path, index=False)

    npz_path = OUTPUT_DIR / f"per_query_{dataset}.npz"
    np.savez(
        npz_path,
        b1_correct=b1["correct@1"],
        b2_correct=b2["correct@1"],
        v1_correct=v1["correct@1"],
        v2_correct=v2["correct@1"],
        su=su,
        lu=lu,
        su_norm=su_norm,
        lu_norm=lu_norm,
        alpha_v1=alpha_v1,
        alpha_v2=alpha_v2,
        topk_idx=topk_idx,
        retrieved_v1=retrieved_v1,
        retrieved_v2=retrieved_v2,
    )

    print()
    print("Saved:")
    print(csv_path)
    print(npz_path)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amstertime", "msls_val", "nordland_clean", "all"],
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_pool", type=int, default=20)
    parser.add_argument("--old_tau", type=float, default=1.0)
    parser.add_argument("--old_lang_threshold", type=float, default=0.05)
    parser.add_argument("--fixed_alpha", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=6.0)

    args = parser.parse_args()

    if args.dataset == "all":
        for dataset in ["amstertime", "msls_val", "nordland_clean"]:
            run_dataset(
                dataset=dataset,
                top_k=args.top_k,
                candidate_pool=args.candidate_pool,
                old_tau=args.old_tau,
                old_lang_threshold=args.old_lang_threshold,
                fixed_alpha=args.fixed_alpha,
                rho=args.rho,
            )
    else:
        run_dataset(
            dataset=args.dataset,
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            old_tau=args.old_tau,
            old_lang_threshold=args.old_lang_threshold,
            fixed_alpha=args.fixed_alpha,
            rho=args.rho,
        )


if __name__ == "__main__":
    main()
