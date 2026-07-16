import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/content/vpr-language-fusion")
OUT_DIR = REPO_ROOT / "results" / "redesign_v2"
EMBED_DIR = Path("/content/drive/MyDrive/vpr_research/embeddings")

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

    n_cal = int(round(calibration_fraction * n_queries))
    cal_idx = np.sort(indices[:n_cal])
    eval_idx = np.sort(indices[n_cal:])

    np.savez(
        split_path,
        calibration_indices=cal_idx,
        evaluation_indices=eval_idx,
        seed=seed,
        calibration_fraction=calibration_fraction,
        n_queries=n_queries,
        n_calibration=len(cal_idx),
        n_evaluation=len(eval_idx),
    )

    return split_path


def correct_at_1(retrieved, positive):
    correct = np.zeros(retrieved.shape[0], dtype=bool)
    for i in range(retrieved.shape[0]):
        correct[i] = positive[i, retrieved[i, 0]]
    return correct


def phi_coefficient(a, b, c, d):
    # Table:
    # a = visual correct, language correct
    # b = visual correct, language wrong
    # c = visual wrong, language correct
    # d = visual wrong, language wrong
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return np.nan
    return (a * d - b * c) / np.sqrt(denom)


def odds_ratio_haldane(a, b, c, d):
    # Add 0.5 to avoid divide-by-zero.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def contingency_summary(dataset_name, language_mode, visual_correct, language_correct):
    n = len(visual_correct)

    a = int(np.sum(visual_correct & language_correct))
    b = int(np.sum(visual_correct & ~language_correct))
    c = int(np.sum(~visual_correct & language_correct))
    d = int(np.sum(~visual_correct & ~language_correct))

    visual_acc = 100.0 * visual_correct.mean()
    language_acc = 100.0 * language_correct.mean()

    observed_double_wrong = 100.0 * d / n

    expected_double_wrong = 100.0 * (
        (1.0 - visual_correct.mean()) *
        (1.0 - language_correct.mean())
    )

    lift_double_wrong = (
        observed_double_wrong / expected_double_wrong
        if expected_double_wrong > 0
        else np.nan
    )

    phi = phi_coefficient(a, b, c, d)
    odds = odds_ratio_haldane(a, b, c, d)

    return {
        "Dataset": dataset_name,
        "Language_mode": language_mode,
        "n_queries": n,
        "both_correct": a,
        "visual_correct_language_wrong": b,
        "visual_wrong_language_correct": c,
        "both_wrong": d,
        "visual_R@1": visual_acc,
        "language_R@1": language_acc,
        "observed_double_wrong_pct": observed_double_wrong,
        "expected_double_wrong_if_independent_pct": expected_double_wrong,
        "double_wrong_lift_vs_independence": lift_double_wrong,
        "phi_correlation": phi,
        "odds_ratio_haldane": odds,
    }


def main():
    split_path = ensure_msls_split()
    split = np.load(split_path)
    eval_indices = split["evaluation_indices"]

    rows = []

    for key, cfg in DATASETS.items():
        visual = np.load(EMBED_DIR / cfg["visual"])
        language = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual = visual[eval_indices]
            language = language[eval_indices]
            positive = positive[eval_indices]

        visual_ret = np.argsort(-visual, axis=1)
        visual_correct = correct_at_1(visual_ret, positive)

        # Mode 1: pure language-only over the full database.
        lang_full_ret = np.argsort(-language, axis=1)
        lang_full_correct = correct_at_1(lang_full_ret, positive)

        rows.append(
            contingency_summary(
                cfg["pretty"],
                "language_full_database_top1",
                visual_correct,
                lang_full_correct,
            )
        )

        # Mode 2: language reranking only within visual top-10,
        # matching the actual fusion candidate-pool setting.
        candidate_pool = 10
        pool_idx = np.argsort(-visual, axis=1)[:, :candidate_pool]
        language_pool = np.take_along_axis(language, pool_idx, axis=1)
        order = np.argsort(-language_pool, axis=1)
        lang_pool_ret = np.take_along_axis(pool_idx, order, axis=1)
        lang_pool_correct = correct_at_1(lang_pool_ret, positive)

        rows.append(
            contingency_summary(
                cfg["pretty"],
                "language_within_visual_top10_top1",
                visual_correct,
                lang_pool_correct,
            )
        )

    df = pd.DataFrame(rows)

    out_path = OUT_DIR / "visual_language_error_correlation.csv"
    df.to_csv(out_path, index=False)

    print("\nVISUAL-LANGUAGE ERROR CORRELATION")
    print(df.to_string(index=False))

    print("\nInterpretation guide:")
    print("phi near 0  = weak/no query-level error association")
    print("phi > 0     = modalities tend to be correct/wrong on the same queries")
    print("double_wrong_lift > 1 = both-wrong cases occur more often than independence predicts")

    print("\nSaved:")
    print(out_path)


if __name__ == "__main__":
    main()
