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

from su_metric import (
    sort_scores_desc as old_sort_scores_desc,
    SU_normalised as old_SU_normalised,
    confidence_gated_fusion as old_confidence_gated_fusion,
)


EMBED_DIR = Path("/content/drive/MyDrive/vpr_research/embeddings")
OUT_DIR = REPO_ROOT / "results" / "redesign_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RHO_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
SEEDS = [1, 7, 21, 42, 100]

CALIBRATION_FRACTION = 0.30
GLOBAL_RHO = 3.0
CANDIDATE_POOL = 10
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


def v2_retrieval(visual, language, rho, candidate_pool=CANDIDATE_POOL, uncertainty_k=UNCERTAINTY_K):
    topk_idx = np.argsort(-visual, axis=1)[:, :uncertainty_k]

    visual_topk = np.take_along_axis(visual, topk_idx, axis=1)
    language_topk = np.take_along_axis(language, topk_idx, axis=1)

    su = visual_uncertainty(visual_topk, lam=0.5)
    lu = language_uncertainty(language_topk, lam=0.5)

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

    return retrieved, float(alpha.mean())


def evaluate_method(name, retrieved, positive, mean_alpha=np.nan):
    r1, c1 = recall_at_k(retrieved, positive, 1)
    r5, c5 = recall_at_k(retrieved, positive, 5)
    r10, c10 = recall_at_k(retrieved, positive, 10)

    return {
        "method": name,
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "count@1": c1,
        "count@5": c5,
        "count@10": c10,
        "mean_alpha": mean_alpha,
    }


def select_rho_on_calibration(visual_cal, language_cal, positive_cal):
    rows = []

    for rho in RHO_GRID:
        ret, mean_alpha = v2_retrieval(
            visual=visual_cal,
            language=language_cal,
            rho=rho,
        )

        r1, c1 = recall_at_k(ret, positive_cal, 1)
        r5, c5 = recall_at_k(ret, positive_cal, 5)
        r10, c10 = recall_at_k(ret, positive_cal, 10)

        rows.append({
            "rho": rho,
            "calibration_R@1": r1,
            "calibration_R@5": r5,
            "calibration_R@10": r10,
            "calibration_count@1": c1,
            "calibration_count@5": c5,
            "calibration_count@10": c10,
            "calibration_mean_alpha": mean_alpha,
        })

    df = pd.DataFrame(rows)

    # Selection rule:
    # maximize calibration R@1;
    # if tied, smaller rho.
    selected = (
        df
        .sort_values(by=["calibration_R@1", "rho"], ascending=[False, True])
        .iloc[0]
    )

    return float(selected["rho"]), df


def main():
    visual = np.load(EMBED_DIR / "nordland_clean_visual_sim_matrix.npy")
    language = np.load(EMBED_DIR / "nordland_clean_lang_sim_matrix.npy")
    positive = np.load(EMBED_DIR / "nordland_clean_positive_matrix.npy").astype(bool)

    n_queries = visual.shape[0]

    print("Nordland oracle rho diagnostic")
    print("n_queries:", n_queries)
    print("rho grid:", RHO_GRID)
    print("calibration fraction:", CALIBRATION_FRACTION)
    print("seeds:", SEEDS)
    print("candidate_pool fixed:", CANDIDATE_POOL)
    print("uncertainty_k fixed:", UNCERTAINTY_K)
    print("global rho:", GLOBAL_RHO)
    print("Selection rule: max calibration R@1, then smaller rho")
    print("Note: Nordland-calibrated rho is an oracle/diagnostic because it uses target-domain labels.")

    # Full-data diagnostic only. This is intentionally leaky and should not be used as main result.
    full_rows = []

    visual_ret_full = np.argsort(-visual, axis=1)
    full_rows.append(evaluate_method("B1_visual_only", visual_ret_full, positive))

    v1_ret_full, v1_alpha_full = v1_retrieval(visual, language)
    full_rows.append(evaluate_method("V1_sigmoid_threshold", v1_ret_full, positive, v1_alpha_full))

    global_ret_full, global_alpha_full = v2_retrieval(visual, language, rho=GLOBAL_RHO)
    full_rows.append(evaluate_method("V2_global_rho_3", global_ret_full, positive, global_alpha_full))

    selected_full_rho, full_grid_df = select_rho_on_calibration(visual, language, positive)
    oracle_ret_full, oracle_alpha_full = v2_retrieval(visual, language, rho=selected_full_rho)
    full_rows.append(evaluate_method(f"V2_full_data_oracle_rho_{selected_full_rho}", oracle_ret_full, positive, oracle_alpha_full))

    full_df = pd.DataFrame(full_rows)
    full_grid_df["scope"] = "full_data_oracle_leaky"

    print("\nFULL-DATA DIAGNOSTIC ONLY")
    print("Selected full-data oracle rho:", selected_full_rho)
    print(full_df.to_string(index=False))

    # Split-based target-domain oracle check.
    split_summary_rows = []
    split_metric_rows = []
    grid_rows = []

    for seed in SEEDS:
        cal_idx, eval_idx = split_indices(n_queries, seed)

        visual_cal = visual[cal_idx]
        language_cal = language[cal_idx]
        positive_cal = positive[cal_idx]

        visual_eval = visual[eval_idx]
        language_eval = language[eval_idx]
        positive_eval = positive[eval_idx]

        selected_rho, grid_df = select_rho_on_calibration(
            visual_cal=visual_cal,
            language_cal=language_cal,
            positive_cal=positive_cal,
        )

        grid_df["seed"] = seed
        grid_df["n_calibration"] = len(cal_idx)
        grid_df["n_evaluation"] = len(eval_idx)
        grid_rows.append(grid_df)

        methods = []

        b1_eval = np.argsort(-visual_eval, axis=1)
        methods.append(evaluate_method("B1_visual_only", b1_eval, positive_eval))

        v1_eval, v1_alpha = v1_retrieval(visual_eval, language_eval)
        methods.append(evaluate_method("V1_sigmoid_threshold", v1_eval, positive_eval, v1_alpha))

        global_eval, global_alpha = v2_retrieval(visual_eval, language_eval, rho=GLOBAL_RHO)
        methods.append(evaluate_method("V2_global_rho_3", global_eval, positive_eval, global_alpha))

        oracle_eval, oracle_alpha = v2_retrieval(visual_eval, language_eval, rho=selected_rho)
        methods.append(evaluate_method("V2_nordland_calibrated_rho", oracle_eval, positive_eval, oracle_alpha))

        for m in methods:
            m2 = {
                "seed": seed,
                "n_calibration": len(cal_idx),
                "n_evaluation": len(eval_idx),
                "selected_rho": selected_rho,
                **m,
            }
            split_metric_rows.append(m2)

        # Convenient one-row summary for selected rho and deltas.
        metric_df_seed = pd.DataFrame([
            {
                "method": m["method"],
                "R@1": m["R@1"],
                "R@5": m["R@5"],
                "R@10": m["R@10"],
            }
            for m in methods
        ]).set_index("method")

        split_summary_rows.append({
            "seed": seed,
            "n_calibration": len(cal_idx),
            "n_evaluation": len(eval_idx),
            "selected_rho": selected_rho,
            "B1_R@1": metric_df_seed.loc["B1_visual_only", "R@1"],
            "V1_R@1": metric_df_seed.loc["V1_sigmoid_threshold", "R@1"],
            "V2_global_R@1": metric_df_seed.loc["V2_global_rho_3", "R@1"],
            "V2_oracle_R@1": metric_df_seed.loc["V2_nordland_calibrated_rho", "R@1"],
            "oracle_minus_global_R@1": metric_df_seed.loc["V2_nordland_calibrated_rho", "R@1"] - metric_df_seed.loc["V2_global_rho_3", "R@1"],
            "oracle_minus_V1_R@1": metric_df_seed.loc["V2_nordland_calibrated_rho", "R@1"] - metric_df_seed.loc["V1_sigmoid_threshold", "R@1"],
            "oracle_minus_B1_R@1": metric_df_seed.loc["V2_nordland_calibrated_rho", "R@1"] - metric_df_seed.loc["B1_visual_only", "R@1"],
            "V2_global_R@5": metric_df_seed.loc["V2_global_rho_3", "R@5"],
            "V2_oracle_R@5": metric_df_seed.loc["V2_nordland_calibrated_rho", "R@5"],
            "oracle_minus_global_R@5": metric_df_seed.loc["V2_nordland_calibrated_rho", "R@5"] - metric_df_seed.loc["V2_global_rho_3", "R@5"],
        })

    split_summary_df = pd.DataFrame(split_summary_rows)
    split_metrics_df = pd.DataFrame(split_metric_rows)
    grid_all_df = pd.concat(grid_rows, ignore_index=True)

    full_path = OUT_DIR / "nordland_oracle_rho_full_data_diagnostic.csv"
    split_summary_path = OUT_DIR / "nordland_oracle_rho_split_summary.csv"
    split_metrics_path = OUT_DIR / "nordland_oracle_rho_split_metrics.csv"
    grid_path = OUT_DIR / "nordland_oracle_rho_calibration_grid.csv"

    full_df.to_csv(full_path, index=False)
    split_summary_df.to_csv(split_summary_path, index=False)
    split_metrics_df.to_csv(split_metrics_path, index=False)
    grid_all_df.to_csv(grid_path, index=False)

    print("\nSPLIT-BASED NORDLAND ORACLE RHO SUMMARY")
    print(split_summary_df.to_string(index=False))

    print("\nSelected rho counts:")
    print(split_summary_df["selected_rho"].value_counts().sort_index().to_string())

    print("\nOracle minus global R@1 summary:")
    print(split_summary_df["oracle_minus_global_R@1"].describe().to_string())

    print("\nOracle minus V1 R@1 summary:")
    print(split_summary_df["oracle_minus_V1_R@1"].describe().to_string())

    print("\nOracle minus B1 R@1 summary:")
    print(split_summary_df["oracle_minus_B1_R@1"].describe().to_string())

    print("\nSplit metrics by method:")
    method_summary = (
        split_metrics_df
        .groupby("method")
        .agg(
            R1_mean=("R@1", "mean"),
            R1_std=("R@1", "std"),
            R5_mean=("R@5", "mean"),
            R5_std=("R@5", "std"),
            R10_mean=("R@10", "mean"),
            R10_std=("R@10", "std"),
        )
        .reset_index()
    )
    print(method_summary.to_string(index=False))

    print("\nSaved:")
    print(full_path)
    print(split_summary_path)
    print(split_metrics_path)
    print(grid_path)


if __name__ == "__main__":
    main()
