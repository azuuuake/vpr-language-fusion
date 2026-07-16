
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


def load_locked_setting():
    path = OUT_DIR / "calibrated_v2_selection_locked_rule.json"
    if path.exists():
        with open(path, "r") as f:
            s = json.load(f)
        return float(s["selected_rho"]), int(s["selected_candidate_pool"]), int(s["uncertainty_k"])
    return 3.0, 10, 10


def recall_at_k(retrieved, positive, k):
    correct = []
    for i in range(retrieved.shape[0]):
        correct.append(positive[i, retrieved[i, :k]].any())
    return 100.0 * np.mean(correct), int(np.sum(correct))


def metrics(name, retrieved, positive):
    r1, c1 = recall_at_k(retrieved, positive, 1)
    r5, c5 = recall_at_k(retrieved, positive, 5)
    r10, c10 = recall_at_k(retrieved, positive, 10)
    return {
        "Method": name,
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "count@1": c1,
        "count@5": c5,
        "count@10": c10,
        "n_queries": retrieved.shape[0],
    }


def full_db_retrieval(score_matrix):
    return np.argsort(-score_matrix, axis=1)


def v2_retrieval(visual_sim, language_sim, positive, rho, candidate_pool, uncertainty_k):
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :uncertainty_k]

    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

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

    order = np.argsort(-fused_pool, axis=1)
    retrieved = np.take_along_axis(pool_idx, order, axis=1)

    extra = {
        "mean_SU": float(np.mean(su)),
        "mean_LU": float(np.mean(lu)),
        "mean_alpha": float(np.mean(alpha)),
        "mean_language_weight": float(np.mean(1.0 - alpha)),
    }

    return retrieved, extra


def main():
    rho, candidate_pool, uncertainty_k = load_locked_setting()

    print("Frozen setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)

    visual = np.load(EMBED_DIR / "nordland_clean_visual_sim_matrix.npy")
    positive = np.load(EMBED_DIR / "nordland_clean_positive_matrix.npy").astype(bool)

    lang_original = np.load(EMBED_DIR / "nordland_clean_lang_sim_matrix.npy")
    lang_prompted = np.load(EMBED_DIR / "nordland_clean_prompted_lang_sim_matrix.npy")

    rows = []

    rows.append(metrics("B1_visual_only", full_db_retrieval(visual), positive))
    rows.append(metrics("Original_language_only", full_db_retrieval(lang_original), positive))
    rows.append(metrics("Prompted_language_only", full_db_retrieval(lang_prompted), positive))

    rows.append(metrics("Original_fixed_0.5", full_db_retrieval(0.5 * visual + 0.5 * lang_original), positive))
    rows.append(metrics("Prompted_fixed_0.5", full_db_retrieval(0.5 * visual + 0.5 * lang_prompted), positive))

    ret_orig, extra_orig = v2_retrieval(
        visual, lang_original, positive, rho, candidate_pool, uncertainty_k
    )
    ret_prompt, extra_prompt = v2_retrieval(
        visual, lang_prompted, positive, rho, candidate_pool, uncertainty_k
    )

    row_orig = metrics("Original_V2_precision_weighted", ret_orig, positive)
    row_orig.update(extra_orig)
    rows.append(row_orig)

    row_prompt = metrics("Prompted_V2_precision_weighted", ret_prompt, positive)
    row_prompt.update(extra_prompt)
    rows.append(row_prompt)

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "nordland_prompted_full_evaluation.csv"
    df.to_csv(out_csv, index=False)

    print("\nNORDLAND PROMPTED FULL EVALUATION")
    print(df.to_string(index=False))

    print("\nSaved:", out_csv)

    print("\nKEY DELTAS")
    orig_v2 = df[df["Method"] == "Original_V2_precision_weighted"].iloc[0]
    prompt_v2 = df[df["Method"] == "Prompted_V2_precision_weighted"].iloc[0]

    for k in ["R@1", "R@5", "R@10"]:
        print(f"Prompted V2 - Original V2 {k}: {prompt_v2[k] - orig_v2[k]:.2f} pp")

    orig_lang = df[df["Method"] == "Original_language_only"].iloc[0]
    prompt_lang = df[df["Method"] == "Prompted_language_only"].iloc[0]

    for k in ["R@1", "R@5", "R@10"]:
        print(f"Prompted language - Original language {k}: {prompt_lang[k] - orig_lang[k]:.2f} pp")


if __name__ == "__main__":
    main()
