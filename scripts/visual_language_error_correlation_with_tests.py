import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact, chi2_contingency


REPO_ROOT = Path("/content/vpr-language-fusion")
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


def ensure_msls_split():
    split_path = OUT_DIR / "msls_calibration_split_indices.npz"
    if split_path.exists():
        return split_path

    visual = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    n_queries = visual.shape[0]

    rng = np.random.default_rng(42)
    idx = rng.permutation(n_queries)
    n_cal = int(round(0.30 * n_queries))

    np.savez(
        split_path,
        calibration_indices=np.sort(idx[:n_cal]),
        evaluation_indices=np.sort(idx[n_cal:]),
        seed=42,
        calibration_fraction=0.30,
    )
    return split_path


def correct_at_1(retrieved, positive):
    out = np.zeros(retrieved.shape[0], dtype=bool)
    for i in range(retrieved.shape[0]):
        out[i] = positive[i, retrieved[i, 0]]
    return out


def phi_coefficient(a, b, c, d):
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return np.nan
    return (a * d - b * c) / np.sqrt(denom)


def summarize(dataset, mode, visual_correct, language_correct):
    # Rows: visual correct/wrong
    # Cols: language correct/wrong
    a = int(np.sum(visual_correct & language_correct))
    b = int(np.sum(visual_correct & ~language_correct))
    c = int(np.sum(~visual_correct & language_correct))
    d = int(np.sum(~visual_correct & ~language_correct))

    table = np.array([[a, b], [c, d]])

    fisher_or, fisher_p = fisher_exact(table)
    chi2, chi2_p, _, _ = chi2_contingency(table, correction=False)

    n = len(visual_correct)

    visual_wrong_rate = 1.0 - visual_correct.mean()
    language_wrong_rate = 1.0 - language_correct.mean()

    observed_double_wrong = d / n
    expected_double_wrong = visual_wrong_rate * language_wrong_rate

    return {
        "Dataset": dataset,
        "Language_mode": mode,
        "n_queries": n,
        "both_correct": a,
        "visual_correct_language_wrong": b,
        "visual_wrong_language_correct": c,
        "both_wrong": d,
        "visual_R@1": 100 * visual_correct.mean(),
        "language_R@1": 100 * language_correct.mean(),
        "phi": phi_coefficient(a, b, c, d),
        "odds_ratio_fisher": fisher_or,
        "fisher_exact_p": fisher_p,
        "chi_square": chi2,
        "chi_square_p": chi2_p,
        "observed_double_wrong_pct": 100 * observed_double_wrong,
        "expected_double_wrong_if_independent_pct": 100 * expected_double_wrong,
        "double_wrong_lift_vs_independence": observed_double_wrong / expected_double_wrong if expected_double_wrong > 0 else np.nan,
    }


def main():
    split = np.load(ensure_msls_split())
    eval_indices = split["evaluation_indices"]

    rows = []

    for _, cfg in DATASETS.items():
        visual = np.load(EMBED_DIR / cfg["visual"])
        language = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual = visual[eval_indices]
            language = language[eval_indices]
            positive = positive[eval_indices]

        visual_ret = np.argsort(-visual, axis=1)
        visual_correct = correct_at_1(visual_ret, positive)

        # Full database language top-1
        lang_full_ret = np.argsort(-language, axis=1)
        lang_full_correct = correct_at_1(lang_full_ret, positive)

        rows.append(
            summarize(
                cfg["pretty"],
                "language_full_database_top1",
                visual_correct,
                lang_full_correct,
            )
        )

        # Language top-1 inside visual top-10 candidate pool
        pool_idx = np.argsort(-visual, axis=1)[:, :10]
        language_pool = np.take_along_axis(language, pool_idx, axis=1)
        order = np.argsort(-language_pool, axis=1)
        lang_pool_ret = np.take_along_axis(pool_idx, order, axis=1)
        lang_pool_correct = correct_at_1(lang_pool_ret, positive)

        rows.append(
            summarize(
                cfg["pretty"],
                "language_within_visual_top10_top1",
                visual_correct,
                lang_pool_correct,
            )
        )

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "visual_language_error_correlation_with_tests.csv"
    df.to_csv(out_csv, index=False)

    print("\nVISUAL-LANGUAGE ERROR CORRELATION WITH TESTS")
    print(df.to_string(index=False))

    print("\nShort paper table:")
    cols = [
        "Dataset",
        "Language_mode",
        "both_correct",
        "visual_correct_language_wrong",
        "visual_wrong_language_correct",
        "both_wrong",
        "phi",
        "odds_ratio_fisher",
        "fisher_exact_p",
        "double_wrong_lift_vs_independence",
    ]
    print(df[cols].to_string(index=False))

    print("\nSaved:")
    print(out_csv)


if __name__ == "__main__":
    main()
