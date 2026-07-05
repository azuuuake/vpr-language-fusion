
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


def build_methods_and_uncertainties(visual_sim, language_sim, uncertainty_k, candidate_pool, rho):
    # B1 visual-only
    retrieved_b1 = np.argsort(-visual_sim, axis=1)

    # B2 fixed full-database fusion
    fixed_sim = 0.5 * visual_sim + 0.5 * language_sim
    retrieved_b2 = np.argsort(-fixed_sim, axis=1)

    # Top-k for SU/LU and V1
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :uncertainty_k]
    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

    # V1 old method
    visual_topk_v1 = old_sort_scores_desc(visual_topk)
    su_norm_v1 = old_SU_normalised(visual_topk_v1)

    fused_v1, alpha_v1 = old_confidence_gated_fusion(
        visual_sims=visual_topk_v1,
        lang_sims=language_topk,
        su_norm=su_norm_v1,
        tau=1.0,
        lang_threshold=0.05,
    )

    order_v1 = np.argsort(-fused_v1, axis=1)
    retrieved_v1 = np.take_along_axis(topk_idx, order_v1, axis=1)

    # V2 calibrated method
    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    sigma_v2 = uncertainty_to_variance(su_norm)
    sigma_l2 = uncertainty_to_variance(lu_norm)
    joint_uncertainty = sigma_v2 + sigma_l2

    pool_idx = np.argsort(-visual_sim, axis=1)[:, :candidate_pool]
    visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

    fused_v2, alpha_v2 = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    order_v2 = np.argsort(-fused_v2, axis=1)
    retrieved_v2 = np.take_along_axis(pool_idx, order_v2, axis=1)

    retrievals = {
        "B1 visual-only": retrieved_b1,
        "B2 fixed fusion": retrieved_b2,
        "V1 sigmoid+threshold": retrieved_v1,
        "V2 calibrated precision": retrieved_v2,
    }

    uncertainties = {
        "LU": lu,
        "SU": su,
        "joint_variance": joint_uncertainty,
    }

    diagnostics = {
        "mean_LU": float(lu.mean()),
        "mean_SU": float(su.mean()),
        "mean_alpha": float(alpha_v2.mean()),
    }

    return retrievals, uncertainties, diagnostics


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

    print("Frozen V2 setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)

    rows = []
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

        retrievals, uncertainties, diagnostics = build_methods_and_uncertainties(
            visual_sim=visual_sim,
            language_sim=language_sim,
            uncertainty_k=uncertainty_k,
            candidate_pool=candidate_pool,
            rho=rho,
        )

        correct = {
            method: correct_at_1(retrieved, positive)
            for method, retrieved in retrievals.items()
        }

        for axis_name, uncertainty_score in uncertainties.items():
            order = np.argsort(uncertainty_score)  # low uncertainty first
            axis_risks = []

            for coverage in COVERAGES:
                n_keep = int(np.ceil(coverage * len(order)))
                keep = order[:n_keep]

                row = {
                    "Dataset": pretty,
                    "Split": split,
                    "Uncertainty_axis": axis_name,
                    "Coverage": float(coverage),
                    "n_kept": int(n_keep),
                    "mean_uncertainty_kept": float(uncertainty_score[keep].mean()),
                    "rho": rho,
                    "candidate_pool": candidate_pool,
                    "uncertainty_k": uncertainty_k,
                    "mean_LU_full": diagnostics["mean_LU"],
                    "mean_SU_full": diagnostics["mean_SU"],
                    "mean_alpha_full": diagnostics["mean_alpha"],
                }

                for method in correct:
                    method_acc = acc(correct[method][keep])
                    row[f"{method}_R@1"] = method_acc
                    row[f"{method}_risk"] = 100.0 - method_acc

                rows.append(row)

                axis_risks.append(row["V2 calibrated precision_risk"])

            aurc = float(np.trapz(axis_risks, COVERAGES))

            aurc_rows.append({
                "Dataset": pretty,
                "Split": split,
                "Uncertainty_axis": axis_name,
                "V2_AURC": aurc,
                "rho": rho,
                "candidate_pool": candidate_pool,
                "uncertainty_k": uncertainty_k,
            })

        # Plot V2 risk-coverage for three uncertainty axes
        plot_df = pd.DataFrame([r for r in rows if r["Dataset"] == pretty])

        plt.figure(figsize=(7, 5))

        for axis_name in ["LU", "SU", "joint_variance"]:
            sub = plot_df[plot_df["Uncertainty_axis"] == axis_name]
            plt.plot(
                sub["Coverage"],
                sub["V2 calibrated precision_risk"],
                marker="o",
                label=axis_name,
            )

        plt.xlabel("Coverage")
        plt.ylabel("Risk = 100 - R@1")
        plt.title(f"Risk-Coverage Curve: {pretty}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = FIG_DIR / f"risk_coverage_{key}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print("Saved figure:", fig_path)

    risk_df = pd.DataFrame(rows)
    aurc_df = pd.DataFrame(aurc_rows)

    risk_path = OUTPUT_DIR / "risk_coverage_curves.csv"
    aurc_path = OUTPUT_DIR / "risk_coverage_aurc_summary.csv"

    risk_df.to_csv(risk_path, index=False)
    aurc_df.to_csv(aurc_path, index=False)

    print("\nSaved:")
    print(risk_path)
    print(aurc_path)

    print("\nAURC SUMMARY")
    print(aurc_df.to_string(index=False))

    print("\nLU-AXIS CHECKPOINTS")
    lu_check = risk_df[
        (risk_df["Uncertainty_axis"] == "LU")
        & (risk_df["Coverage"].isin([0.10, 0.20, 0.30, 0.50, 1.00]))
    ][
        [
            "Dataset",
            "Coverage",
            "n_kept",
            "mean_uncertainty_kept",
            "B1 visual-only_R@1",
            "V1 sigmoid+threshold_R@1",
            "V2 calibrated precision_R@1",
            "V2 calibrated precision_risk",
        ]
    ]

    print(lu_check.to_string(index=False))


if __name__ == "__main__":
    main()
