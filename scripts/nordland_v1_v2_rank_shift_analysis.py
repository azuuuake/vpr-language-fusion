
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/content/vpr-language-fusion")
sys.path.insert(0, str(REPO_ROOT / "src"))

# Old V1 method
from su_metric import (
    sort_scores_desc as old_sort_scores_desc,
    SU_normalised as old_SU_normalised,
    confidence_gated_fusion as old_confidence_gated_fusion,
)

# New V2 method
from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    precision_weighted_fusion,
)


EMBED_DIR = Path("/content/drive/MyDrive/vpr_research/embeddings")
OUT_DIR = REPO_ROOT / "results" / "redesign_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_locked_setting():
    path = OUT_DIR / "calibrated_v2_selection_locked_rule.json"
    if path.exists():
        with open(path, "r") as f:
            s = json.load(f)
        return float(s["selected_rho"]), int(s["selected_candidate_pool"]), int(s["uncertainty_k"])
    return 3.0, 10, 10


def v1_retrieval(visual, language, top_k=10, tau=1.0, lang_threshold=0.05):
    topk_idx = np.argsort(-visual, axis=1)[:, :top_k]

    visual_topk = np.take_along_axis(visual, topk_idx, axis=1)
    language_topk = np.take_along_axis(language, topk_idx, axis=1)

    visual_topk_sorted = old_sort_scores_desc(visual_topk)
    su_norm = old_SU_normalised(visual_topk_sorted)

    fused_topk, alpha = old_confidence_gated_fusion(
        visual_sims=visual_topk_sorted,
        lang_sims=language_topk,
        su_norm=su_norm,
        tau=tau,
        lang_threshold=lang_threshold,
    )

    order = np.argsort(-fused_topk, axis=1)
    retrieved = np.take_along_axis(topk_idx, order, axis=1)

    return retrieved, alpha


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

    return retrieved, alpha, su, lu


def best_positive_rank(retrieved_row, positive_row):
    """
    Return 1-indexed best rank of any positive DB item.
    If no positive appears in retrieved_row, return >len(retrieved_row).
    """
    for rank, db_idx in enumerate(retrieved_row, start=1):
        if positive_row[db_idx]:
            return rank
    return len(retrieved_row) + 1


def bucket_rank(rank, max_rank=10):
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if 4 <= rank <= 5:
        return "rank_4_5"
    if 6 <= rank <= 10:
        return "rank_6_10"
    return "outside_top10"


def main():
    rho, candidate_pool, uncertainty_k = load_locked_setting()

    print("Frozen V2 setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)
    print("V1 setting: top_k=10, tau=1.0, lang_threshold=0.05")

    visual = np.load(EMBED_DIR / "nordland_clean_visual_sim_matrix.npy")
    language = np.load(EMBED_DIR / "nordland_clean_lang_sim_matrix.npy")
    positive = np.load(EMBED_DIR / "nordland_clean_positive_matrix.npy").astype(bool)

    v1, alpha_v1 = v1_retrieval(
        visual=visual,
        language=language,
        top_k=10,
        tau=1.0,
        lang_threshold=0.05,
    )

    v2, alpha_v2, su, lu = v2_retrieval(
        visual=visual,
        language=language,
        rho=rho,
        candidate_pool=candidate_pool,
        uncertainty_k=uncertainty_k,
    )

    rows = []

    for qid in range(positive.shape[0]):
        v1_rank = best_positive_rank(v1[qid], positive[qid])
        v2_rank = best_positive_rank(v2[qid], positive[qid])

        rows.append({
            "query_id": qid,
            "v1_best_positive_rank_in_top10": v1_rank,
            "v2_best_positive_rank_in_top10": v2_rank,
            "v1_bucket": bucket_rank(v1_rank),
            "v2_bucket": bucket_rank(v2_rank),
            "v1_R@1_correct": v1_rank == 1,
            "v2_R@1_correct": v2_rank == 1,
            "v1_R@5_correct": v1_rank <= 5,
            "v2_R@5_correct": v2_rank <= 5,
            "v1_R@10_correct": v1_rank <= 10,
            "v2_R@10_correct": v2_rank <= 10,
            "rank_shift_v2_minus_v1": v2_rank - v1_rank,
            "alpha_v1": float(alpha_v1[qid]),
            "alpha_v2": float(alpha_v2[qid]),
            "SU": float(su[qid]),
            "LU": float(lu[qid]),
        })

    df = pd.DataFrame(rows)

    # Key subset from supervisor:
    # V1 correct at R@1 but V2 not correct at R@1
    demoted = df[(df["v1_R@1_correct"]) & (~df["v2_R@1_correct"])].copy()

    # Opposite subset:
    # V2 correct at R@1 but V1 not correct at R@1
    recovered = df[(~df["v1_R@1_correct"]) & (df["v2_R@1_correct"])].copy()

    summary_rows = []

    def add_bucket_summary(name, sub):
        if len(sub) == 0:
            return

        bucket_counts = sub["v2_bucket"].value_counts().to_dict()

        row = {
            "Subset": name,
            "n_queries": len(sub),
            "v2_rank2_count": int(bucket_counts.get("rank_2", 0)),
            "v2_rank3_count": int(bucket_counts.get("rank_3", 0)),
            "v2_rank4_5_count": int(bucket_counts.get("rank_4_5", 0)),
            "v2_rank2_5_count": int(
                bucket_counts.get("rank_2", 0)
                + bucket_counts.get("rank_3", 0)
                + bucket_counts.get("rank_4_5", 0)
            ),
            "v2_rank6_10_count": int(bucket_counts.get("rank_6_10", 0)),
            "v2_outside_top10_count": int(bucket_counts.get("outside_top10", 0)),
            "v2_rank2_5_pct": 100.0 * int(
                bucket_counts.get("rank_2", 0)
                + bucket_counts.get("rank_3", 0)
                + bucket_counts.get("rank_4_5", 0)
            ) / len(sub),
            "v2_outside_top10_pct": 100.0 * int(bucket_counts.get("outside_top10", 0)) / len(sub),
            "mean_rank_shift_v2_minus_v1": float(sub["rank_shift_v2_minus_v1"].mean()),
            "median_rank_shift_v2_minus_v1": float(sub["rank_shift_v2_minus_v1"].median()),
            "mean_alpha_v1": float(sub["alpha_v1"].mean()),
            "mean_alpha_v2": float(sub["alpha_v2"].mean()),
            "mean_SU": float(sub["SU"].mean()),
            "mean_LU": float(sub["LU"].mean()),
        }

        summary_rows.append(row)

    add_bucket_summary("V1_R@1_correct_but_V2_R@1_wrong", demoted)
    add_bucket_summary("V2_R@1_correct_but_V1_R@1_wrong", recovered)

    # Global transition table
    transition = pd.crosstab(df["v1_bucket"], df["v2_bucket"])

    summary = pd.DataFrame(summary_rows)

    out_query = OUT_DIR / "nordland_v1_v2_rank_shift_query_level.csv"
    out_summary = OUT_DIR / "nordland_v1_v2_rank_shift_summary.csv"
    out_transition = OUT_DIR / "nordland_v1_v2_rank_shift_transition_table.csv"

    df.to_csv(out_query, index=False)
    summary.to_csv(out_summary, index=False)
    transition.to_csv(out_transition)

    print("\nOVERALL METRICS")
    print("V1 R@1:", 100.0 * df["v1_R@1_correct"].mean())
    print("V1 R@5:", 100.0 * df["v1_R@5_correct"].mean())
    print("V1 R@10:", 100.0 * df["v1_R@10_correct"].mean())
    print("V2 R@1:", 100.0 * df["v2_R@1_correct"].mean())
    print("V2 R@5:", 100.0 * df["v2_R@5_correct"].mean())
    print("V2 R@10:", 100.0 * df["v2_R@10_correct"].mean())

    print("\nKEY RANK-SHIFT SUMMARY")
    print(summary.to_string(index=False))

    print("\nV1 bucket vs V2 bucket transition table")
    print(transition.to_string())

    print("\nFirst 20 V1-correct/V2-wrong cases")
    show_cols = [
        "query_id",
        "v1_best_positive_rank_in_top10",
        "v2_best_positive_rank_in_top10",
        "v2_bucket",
        "rank_shift_v2_minus_v1",
        "alpha_v1",
        "alpha_v2",
        "SU",
        "LU",
    ]
    print(demoted[show_cols].head(20).to_string(index=False))

    print("\nSaved:")
    print(out_query)
    print(out_summary)
    print(out_transition)


if __name__ == "__main__":
    main()
