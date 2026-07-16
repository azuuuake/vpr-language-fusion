
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path("/content/vpr-language-fusion")
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
    uncertainty_to_variance,
    precision_weighted_fusion,
)


EMBED_DIR = Path("/content/drive/MyDrive/vpr_research/embeddings")
OUT_DIR = REPO_ROOT / "results" / "redesign_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
BOOT_SEED = 42

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


def correct_at_k(retrieved, positive, k):
    out = np.zeros(retrieved.shape[0], dtype=bool)
    for i in range(retrieved.shape[0]):
        out[i] = positive[i, retrieved[i, :k]].any()
    return out


def query_uncertainties(visual, language, uncertainty_k):
    topk_idx = np.argsort(-visual, axis=1)[:, :uncertainty_k]

    visual_topk = np.take_along_axis(visual, topk_idx, axis=1)
    language_topk = np.take_along_axis(language, topk_idx, axis=1)

    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    sigma_v2 = uncertainty_to_variance(su_norm)
    sigma_l2 = uncertainty_to_variance(lu_norm)

    return su, lu, su_norm, lu_norm, sigma_v2, sigma_l2


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

    return retrieved


def v2_retrieval(visual, language, rho, candidate_pool, uncertainty_k):
    su, lu, su_norm, lu_norm, sigma_v2, sigma_l2 = query_uncertainties(
        visual, language, uncertainty_k
    )

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

    return retrieved


def build_candidate_features(visual, language, uncertainty_k, candidate_pool, rho):
    n_queries = visual.shape[0]

    su, lu, su_norm, lu_norm, sigma_v2, sigma_l2 = query_uncertainties(
        visual, language, uncertainty_k
    )

    pool_idx = np.argsort(-visual, axis=1)[:, :candidate_pool]

    visual_pool = np.take_along_axis(visual, pool_idx, axis=1)
    language_pool = np.take_along_axis(language, pool_idx, axis=1)

    _, alpha = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    visual_rank = np.tile(np.arange(candidate_pool), (n_queries, 1))
    visual_rank_norm = visual_rank / max(candidate_pool - 1, 1)

    language_rank = np.argsort(np.argsort(-language_pool, axis=1), axis=1)
    language_rank_norm = language_rank / max(candidate_pool - 1, 1)

    visual_margin = visual_pool[:, [0]] - visual_pool
    language_margin = language_pool.max(axis=1, keepdims=True) - language_pool

    X = np.column_stack([
        visual_pool.ravel(),
        language_pool.ravel(),
        visual_rank_norm.ravel(),
        language_rank_norm.ravel(),
        visual_margin.ravel(),
        language_margin.ravel(),
        np.repeat(su, candidate_pool),
        np.repeat(lu, candidate_pool),
        np.repeat(su_norm, candidate_pool),
        np.repeat(lu_norm, candidate_pool),
        np.repeat(sigma_v2, candidate_pool),
        np.repeat(sigma_l2, candidate_pool),
        np.repeat(alpha, candidate_pool),
        np.repeat(1.0 - alpha, candidate_pool),
        (visual_pool * language_pool).ravel(),
        (language_pool - visual_pool).ravel(),
    ])

    return X, pool_idx


def train_logreg(msls_visual, msls_language, msls_positive, calibration_indices, uncertainty_k, candidate_pool, rho):
    train_visual = msls_visual[calibration_indices]
    train_language = msls_language[calibration_indices]
    train_positive = msls_positive[calibration_indices]

    X_train, pool_idx = build_candidate_features(
        train_visual, train_language, uncertainty_k, candidate_pool, rho
    )

    y_train = train_positive[
        np.arange(train_visual.shape[0])[:, None],
        pool_idx
    ].ravel().astype(int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
    )

    model.fit(X_train, y_train)

    return model


def logreg_retrieval(model, visual, language, uncertainty_k, candidate_pool, rho):
    X, pool_idx = build_candidate_features(
        visual, language, uncertainty_k, candidate_pool, rho
    )

    prob = model.predict_proba(X)[:, 1].reshape(visual.shape[0], candidate_pool)
    order = np.argsort(-prob, axis=1)
    retrieved = np.take_along_axis(pool_idx, order, axis=1)

    return retrieved


def paired_bootstrap_pvalue(a_correct, b_correct, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Difference = A - B in percentage points.
    Two-sided bootstrap p-value around zero.
    """
    rng = np.random.default_rng(seed)
    n = len(a_correct)

    observed = 100.0 * (a_correct.mean() - b_correct.mean())

    samples = np.empty(n_boot, dtype=np.float32)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = 100.0 * (a_correct[idx].mean() - b_correct[idx].mean())

    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))

    # Two-sided bootstrap p-value.
    p_lower = (np.sum(samples <= 0.0) + 1) / (n_boot + 1)
    p_upper = (np.sum(samples >= 0.0) + 1) / (n_boot + 1)
    p_two = float(min(1.0, 2.0 * min(p_lower, p_upper)))

    ci_sig = not (ci_low <= 0.0 <= ci_high)

    return observed, ci_low, ci_high, p_two, ci_sig


def benjamini_hochberg(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    qvals_ranked = np.empty(m, dtype=float)

    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * m / rank
        prev = min(prev, q)
        qvals_ranked[i] = prev

    qvals = np.empty(m, dtype=float)
    qvals[order] = np.minimum(qvals_ranked, 1.0)

    rejected = qvals <= alpha

    return qvals, rejected


def main():
    rho, candidate_pool, uncertainty_k = load_locked_setting()

    split_path = ensure_msls_split()
    split = np.load(split_path)

    calibration_indices = split["calibration_indices"]
    evaluation_indices = split["evaluation_indices"]

    print("Frozen setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)
    print("MSLS calibration:", len(calibration_indices))
    print("MSLS held-out:", len(evaluation_indices))
    print("Bootstrap samples:", N_BOOT)
    print("FDR method: Benjamini-Hochberg across main comparison family")

    msls_visual = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    msls_language = np.load(EMBED_DIR / "msls_val_lang_sim_matrix.npy")
    msls_positive = np.load(EMBED_DIR / "msls_val_positive_matrix.npy").astype(bool)

    logreg_model = train_logreg(
        msls_visual=msls_visual,
        msls_language=msls_language,
        msls_positive=msls_positive,
        calibration_indices=calibration_indices,
        uncertainty_k=uncertainty_k,
        candidate_pool=candidate_pool,
        rho=rho,
    )

    rows = []

    for key, cfg in DATASETS.items():
        pretty = cfg["pretty"]

        visual = np.load(EMBED_DIR / cfg["visual"])
        language = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual = visual[evaluation_indices]
            language = language[evaluation_indices]
            positive = positive[evaluation_indices]

        methods = {
            "B1_visual_only": np.argsort(-visual, axis=1),
            "B2_fixed_0.5": np.argsort(-(0.5 * visual + 0.5 * language), axis=1),
            "V1_sigmoid_threshold": v1_retrieval(visual, language),
            "V2_precision_weighted": v2_retrieval(visual, language, rho, candidate_pool, uncertainty_k),
            "LogReg_learned_reranker": logreg_retrieval(logreg_model, visual, language, uncertainty_k, candidate_pool, rho),
        }

        correct = {
            method: {
                k: correct_at_k(ret, positive, k)
                for k in [1, 5, 10]
            }
            for method, ret in methods.items()
        }

        comparisons = [
            ("V2_precision_weighted", "B1_visual_only"),
            ("V2_precision_weighted", "B2_fixed_0.5"),
            ("V2_precision_weighted", "V1_sigmoid_threshold"),
            ("V2_precision_weighted", "LogReg_learned_reranker"),
        ]

        for a, b in comparisons:
            for k in [1, 5, 10]:
                obs, lo, hi, p, ci_sig = paired_bootstrap_pvalue(
                    correct[a][k],
                    correct[b][k],
                )

                rows.append({
                    "Dataset": pretty,
                    "Metric": f"R@{k}",
                    "Comparison": f"{a} - {b}",
                    "Difference_pp": obs,
                    "CI_low": lo,
                    "CI_high": hi,
                    "bootstrap_p_two_sided": p,
                    "Significant_by_95CI_uncorrected": ci_sig,
                    "n_queries": visual.shape[0],
                    "n_boot": N_BOOT,
                    "rho": rho,
                    "candidate_pool": candidate_pool,
                    "uncertainty_k": uncertainty_k,
                })

    df = pd.DataFrame(rows)

    qvals, rejected = benjamini_hochberg(df["bootstrap_p_two_sided"].values, alpha=0.05)

    df["BH_FDR_q_value"] = qvals
    df["Significant_after_BH_FDR_0.05"] = rejected
    df["n_tests_in_family"] = len(df)

    out_path = OUT_DIR / "main_comparisons_bh_fdr_correction.csv"
    df.to_csv(out_path, index=False)

    print("\nFDR-CORRECTED MAIN COMPARISONS")
    print(df.to_string(index=False))

    print("\nSUMMARY")
    print("Number of tests:", len(df))
    print("Uncorrected significant:", int(df["Significant_by_95CI_uncorrected"].sum()))
    print("BH-FDR significant:", int(df["Significant_after_BH_FDR_0.05"].sum()))

    print("\nRows surviving BH-FDR:")
    survived = df[df["Significant_after_BH_FDR_0.05"]].copy()
    if len(survived) == 0:
        print("None")
    else:
        print(survived.to_string(index=False))

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
