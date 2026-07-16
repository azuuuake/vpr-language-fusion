
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path("/content/vpr-language-fusion")
sys.path.insert(0, str(REPO_ROOT / "src"))

from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    uncertainty_to_variance,
    precision_weighted_fusion,
)


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"
FIG_DIR = REPO_ROOT / "figures" / "redesign_v2"

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

COVERAGES = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])


def correct_at_1(retrieved_indices, positive):
    correct = np.zeros(retrieved_indices.shape[0], dtype=bool)

    for i in range(retrieved_indices.shape[0]):
        correct[i] = positive[i, retrieved_indices[i, 0]]

    return correct


def acc(correct):
    return 100.0 * correct.mean()


def build_v2_outputs(visual_sim, language_sim, uncertainty_k, candidate_pool, rho):
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :uncertainty_k]

    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    sigma_v2 = uncertainty_to_variance(su_norm)
    sigma_l2 = uncertainty_to_variance(lu_norm)

    joint_variance = sigma_v2 + sigma_l2

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
    retrieved_v2 = np.take_along_axis(pool_idx, order, axis=1)

    return {
        "retrieved_v2": retrieved_v2,
        "SU": su,
        "LU": lu,
        "joint_variance_unweighted": joint_variance,
        "alpha_visual_weight": alpha,
        "language_weight_1_minus_alpha": 1.0 - alpha,
        "mean_alpha": float(alpha.mean()),
        "mean_SU": float(su.mean()),
        "mean_LU": float(lu.mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    selection_path = OUTPUT_DIR / "calibrated_v2_selection_locked_rule.json"
    split_path = OUTPUT_DIR / "msls_calibration_split_indices.npz"

    with open(selection_path, "r") as f:
        selection = json.load(f)

    rho = float(selection["selected_rho"])
    candidate_pool = int(selection["selected_candidate_pool"])
    uncertainty_k = int(selection["uncertainty_k"])

    split_data = np.load(split_path)
    msls_eval_indices = split_data["evaluation_indices"]

    print("Frozen calibrated setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)

    all_rows = []
    aurc_rows = []

    for key, cfg in DATASETS.items():
        pretty = cfg["pretty"]
        split = cfg["split"]

        print("\n" + "=" * 80)
        print(pretty)
        print("=" * 80)

        visual_sim = np.load(EMBED_DIR / cfg["visual"])
        language_sim = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if split == "heldout":
            visual_sim = visual_sim[msls_eval_indices]
            language_sim = language_sim[msls_eval_indices]
            positive = positive[msls_eval_indices]

        out = build_v2_outputs(
            visual_sim=visual_sim,
            language_sim=language_sim,
            uncertainty_k=uncertainty_k,
            candidate_pool=candidate_pool,
            rho=rho,
        )

        v2_correct = correct_at_1(out["retrieved_v2"], positive)

        axes = {
            "SU": out["SU"],
            "LU": out["LU"],
            "joint_variance_unweighted": out["joint_variance_unweighted"],
            "language_weight_1_minus_alpha": out["language_weight_1_minus_alpha"],
        }

        for axis_name, score in axes.items():
            order = np.argsort(score)  # lower score is treated as safer
            risks = []

            for coverage in COVERAGES:
                n_keep = int(np.ceil(coverage * len(order)))
                keep = order[:n_keep]

                r1 = acc(v2_correct[keep])
                risk = 100.0 - r1
                risks.append(risk)

                all_rows.append({
                    "Dataset": pretty,
                    "Split": split,
                    "Axis": axis_name,
                    "Coverage": float(coverage),
                    "n_kept": int(n_keep),
                    "mean_axis_score_kept": float(score[keep].mean()),
                    "V2_R@1": r1,
                    "V2_risk": risk,
                    "rho": rho,
                    "candidate_pool": candidate_pool,
                    "uncertainty_k": uncertainty_k,
                    "mean_alpha_full": out["mean_alpha"],
                    "mean_SU_full": out["mean_SU"],
                    "mean_LU_full": out["mean_LU"],
                })

            aurc = float(np.trapezoid(risks, COVERAGES))

            aurc_rows.append({
                "Dataset": pretty,
                "Split": split,
                "Axis": axis_name,
                "V2_AURC": aurc,
                "rho": rho,
                "candidate_pool": candidate_pool,
                "uncertainty_k": uncertainty_k,
            })

        # Plot
        plot_df = pd.DataFrame([r for r in all_rows if r["Dataset"] == pretty])

        plt.figure(figsize=(7, 5))

        for axis_name in axes.keys():
            sub = plot_df[plot_df["Axis"] == axis_name]
            plt.plot(
                sub["Coverage"],
                sub["V2_risk"],
                marker="o",
                label=axis_name,
            )

        plt.xlabel("Coverage")
        plt.ylabel("Risk = 100 - R@1")
        plt.title(f"V2 Risk-Coverage by Uncertainty Axis: {pretty}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = FIG_DIR / f"alpha_risk_coverage_{key}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print("Saved figure:", fig_path)

    risk_df = pd.DataFrame(all_rows)
    aurc_df = pd.DataFrame(aurc_rows)

    risk_path = OUTPUT_DIR / "alpha_risk_coverage_curves.csv"
    aurc_path = OUTPUT_DIR / "alpha_risk_coverage_aurc_summary.csv"

    risk_df.to_csv(risk_path, index=False)
    aurc_df.to_csv(aurc_path, index=False)

    print("\nSaved:")
    print(risk_path)
    print(aurc_path)

    print("\nAURC SUMMARY")
    print(aurc_df.to_string(index=False))

    print("\nCHECKPOINTS: LANGUAGE WEIGHT AXIS")
    show = risk_df[
        (risk_df["Axis"] == "language_weight_1_minus_alpha")
        & (risk_df["Coverage"].isin([0.10, 0.20, 0.30, 0.50, 1.00]))
    ][
        [
            "Dataset",
            "Coverage",
            "n_kept",
            "mean_axis_score_kept",
            "V2_R@1",
            "V2_risk",
        ]
    ]

    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
