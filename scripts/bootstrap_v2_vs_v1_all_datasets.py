
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


def load_locked_setting():
    path = OUT_DIR / "calibrated_v2_selection_locked_rule.json"

    if path.exists():
        with open(path, "r") as f:
            s = json.load(f)
        return float(s["selected_rho"]), int(s["selected_candidate_pool"]), int(s["uncertainty_k"])

    return 3.0, 10, 10


def recall_correct_at_k(retrieved, positive, k):
    out = np.zeros(retrieved.shape[0], dtype=bool)

    for i in range(retrieved.shape[0]):
        out[i] = positive[i, retrieved[i, :k]].any()

    return out


def metric_row(dataset, method, retrieved, positive, mean_alpha=np.nan):
    c1 = recall_correct_at_k(retrieved, positive, 1)
    c5 = recall_correct_at_k(retrieved, positive, 5)
    c10 = recall_correct_at_k(retrieved, positive, 10)

    return {
        "Dataset": dataset,
        "Method": method,
        "R@1": 100.0 * c1.mean(),
        "R@5": 100.0 * c5.mean(),
        "R@10": 100.0 * c10.mean(),
        "count@1": int(c1.sum()),
        "count@5": int(c5.sum()),
        "count@10": int(c10.sum()),
        "n_queries": retrieved.shape[0],
        "mean_alpha": mean_alpha,
    }


def v1_retrieval(visual, language, top_k=10, tau=1.0, lang_threshold=0.05):
    """
    Old V1:
    - Top-K visual candidates
    - SU_normalised from visual top-K
    - sigmoid gate
    - hard language threshold
    """
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
    """
    New calibrated V2:
    - SU + LU
    - precision-weighted fusion
    - frozen rho/candidate_pool/uncertainty_k
    """
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

    return retrieved, alpha


def paired_bootstrap_diff(a_correct, b_correct, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Difference = A - B in percentage points.
    """
    rng = np.random.default_rng(seed)
    n = len(a_correct)

    observed = 100.0 * (a_correct.mean() - b_correct.mean())

    samples = np.empty(n_boot, dtype=np.float32)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = 100.0 * (a_correct[idx].mean() - b_correct[idx].mean())

    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    sig = not (lo <= 0.0 <= hi)

    return observed, lo, hi, sig


def main():
    rho, candidate_pool, uncertainty_k = load_locked_setting()

    split_path = ensure_msls_split()
    split = np.load(split_path)
    eval_indices = split["evaluation_indices"]

    print("Frozen V2 setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)
    print("V1 setting: top_k=10, tau=1.0, lang_threshold=0.05")
    print("Bootstrap samples:", N_BOOT)

    metrics_rows = []
    boot_rows = []

    for key, cfg in DATASETS.items():
        pretty = cfg["pretty"]

        visual = np.load(EMBED_DIR / cfg["visual"])
        language = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual = visual[eval_indices]
            language = language[eval_indices]
            positive = positive[eval_indices]

        print("\n" + "=" * 80)
        print(pretty)
        print("=" * 80)

        b1 = np.argsort(-visual, axis=1)
        b2 = np.argsort(-(0.5 * visual + 0.5 * language), axis=1)

        v1, alpha_v1 = v1_retrieval(
            visual=visual,
            language=language,
            top_k=10,
            tau=1.0,
            lang_threshold=0.05,
        )

        v2, alpha_v2 = v2_retrieval(
            visual=visual,
            language=language,
            rho=rho,
            candidate_pool=candidate_pool,
            uncertainty_k=uncertainty_k,
        )

        method_retrieved = {
            "B1_visual_only_full_db": b1,
            "B2_fixed_0.5_full_db": b2,
            "V1_sigmoid_threshold_pool10": v1,
            "V2_precision_weighted_pool10": v2,
        }

        metrics_rows.append(metric_row(pretty, "B1_visual_only_full_db", b1, positive))
        metrics_rows.append(metric_row(pretty, "B2_fixed_0.5_full_db", b2, positive))
        metrics_rows.append(metric_row(pretty, "V1_sigmoid_threshold_pool10", v1, positive, float(alpha_v1.mean())))
        metrics_rows.append(metric_row(pretty, "V2_precision_weighted_pool10", v2, positive, float(alpha_v2.mean())))

        correct = {
            method: {
                k: recall_correct_at_k(ret, positive, k)
                for k in [1, 5, 10]
            }
            for method, ret in method_retrieved.items()
        }

        comparisons = [
            ("V2_precision_weighted_pool10", "V1_sigmoid_threshold_pool10"),
            ("V2_precision_weighted_pool10", "B1_visual_only_full_db"),
            ("V1_sigmoid_threshold_pool10", "B1_visual_only_full_db"),
        ]

        for a, b in comparisons:
            for k in [1, 5, 10]:
                obs, lo, hi, sig = paired_bootstrap_diff(correct[a][k], correct[b][k])

                boot_rows.append({
                    "Dataset": pretty,
                    "Metric": f"R@{k}",
                    "Comparison": f"{a} - {b}",
                    "Difference_pp": obs,
                    "CI_low": lo,
                    "CI_high": hi,
                    "Significant_95CI_excludes_0": sig,
                    "n_queries": visual.shape[0],
                    "n_boot": N_BOOT,
                    "rho": rho,
                    "candidate_pool": candidate_pool,
                    "uncertainty_k": uncertainty_k,
                    "v1_tau": 1.0,
                    "v1_lang_threshold": 0.05,
                    "v1_top_k": 10,
                })

        print("\nMETRICS")
        print(pd.DataFrame([r for r in metrics_rows if r["Dataset"] == pretty]).to_string(index=False))

        print("\nBOOTSTRAP COMPARISONS")
        print(pd.DataFrame([r for r in boot_rows if r["Dataset"] == pretty]).to_string(index=False))

    metrics_df = pd.DataFrame(metrics_rows)
    boot_df = pd.DataFrame(boot_rows)

    metrics_path = OUT_DIR / "v2_vs_v1_metrics_all_datasets.csv"
    boot_path = OUT_DIR / "v2_vs_v1_bootstrap_all_datasets.csv"

    metrics_df.to_csv(metrics_path, index=False)
    boot_df.to_csv(boot_path, index=False)

    print("\nSaved:")
    print(metrics_path)
    print(boot_path)

    print("\nFINAL METRICS TABLE")
    print(metrics_df.to_string(index=False))

    print("\nFINAL V2-vs-V1 BOOTSTRAP SUMMARY")
    show = boot_df[boot_df["Comparison"] == "V2_precision_weighted_pool10 - V1_sigmoid_threshold_pool10"]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
