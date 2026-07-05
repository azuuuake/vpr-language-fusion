
import numpy as np
import pandas as pd
from pathlib import Path


REPO_ROOT = Path("/content/vpr-language-fusion")
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

QUERY_PATH = OUTPUT_DIR / "lu_query_level_validation.csv"

BOOTSTRAPS = 5000
SEED = 42


def to_bool_array(series):
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().map({"true": True, "false": False}).to_numpy(dtype=bool)


def bootstrap_paired_diff(a, b, n_boot=BOOTSTRAPS, seed=SEED):
    """
    Paired bootstrap.
    Difference = A - B in percentage points.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = len(a)
    point = 100.0 * (a.mean() - b.mean())

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = 100.0 * (a[idx].mean() - b[idx].mean())

    low, high = np.percentile(diffs, [2.5, 97.5])
    return point, low, high


def bootstrap_independent_diff(a, b, n_boot=BOOTSTRAPS, seed=SEED):
    """
    Independent bootstrap for comparing two different groups.
    Difference = A - B in percentage points.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    point = 100.0 * (a.mean() - b.mean())

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs[i] = 100.0 * (a[ia].mean() - b[ib].mean())

    low, high = np.percentile(diffs, [2.5, 97.5])
    return point, low, high


def main():
    if not QUERY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {QUERY_PATH}. Run lu_decile_validation.py first.")

    df = pd.read_csv(QUERY_PATH)

    # Make sure booleans are booleans after reading CSV
    df["visual_correct@1"] = to_bool_array(df["visual_correct@1"])
    df["language_correct@1"] = to_bool_array(df["language_correct@1"])
    df["v2_correct@1"] = to_bool_array(df["v2_correct@1"])

    # ------------------------------------------------------------
    # 1. Full 10-decile curve
    # ------------------------------------------------------------
    decile_rows = []

    for dataset in df["Dataset"].unique():
        ddf = df[df["Dataset"] == dataset].copy()

        for decile in range(1, 11):
            sub = ddf[ddf["LU_decile"] == decile]

            visual = sub["visual_correct@1"].to_numpy()
            lang = sub["language_correct@1"].to_numpy()
            v2 = sub["v2_correct@1"].to_numpy()

            decile_rows.append({
                "Dataset": dataset,
                "LU_decile": decile,
                "n_queries": len(sub),
                "mean_LU": sub["LU"].mean(),
                "visual_R@1": 100.0 * visual.mean(),
                "language_only_R@1": 100.0 * lang.mean(),
                "v2_R@1": 100.0 * v2.mean(),
                "language_gain_vs_visual_pp": 100.0 * (lang.mean() - visual.mean()),
                "v2_gain_vs_visual_pp": 100.0 * (v2.mean() - visual.mean()),
            })

    decile_df = pd.DataFrame(decile_rows)
    decile_curve_path = OUTPUT_DIR / "lu_full_10_decile_curve.csv"
    decile_df.to_csv(decile_curve_path, index=False)

    # ------------------------------------------------------------
    # 2. Decile-1 paired bootstrap comparisons
    # ------------------------------------------------------------
    decile1_rows = []

    for dataset in df["Dataset"].unique():
        sub = df[(df["Dataset"] == dataset) & (df["LU_decile"] == 1)].copy()

        visual = sub["visual_correct@1"].to_numpy()
        lang = sub["language_correct@1"].to_numpy()
        v2 = sub["v2_correct@1"].to_numpy()

        lang_diff, lang_low, lang_high = bootstrap_paired_diff(lang, visual)
        v2_diff, v2_low, v2_high = bootstrap_paired_diff(v2, visual)

        decile1_rows.append({
            "Dataset": dataset,
            "Bucket": "LU decile 1",
            "n_queries": len(sub),
            "mean_LU": sub["LU"].mean(),
            "visual_R@1": 100.0 * visual.mean(),
            "language_only_R@1": 100.0 * lang.mean(),
            "v2_R@1": 100.0 * v2.mean(),
            "language_minus_visual_pp": lang_diff,
            "language_minus_visual_CI95_low": lang_low,
            "language_minus_visual_CI95_high": lang_high,
            "language_minus_visual_significant": (lang_low > 0) or (lang_high < 0),
            "v2_minus_visual_pp": v2_diff,
            "v2_minus_visual_CI95_low": v2_low,
            "v2_minus_visual_CI95_high": v2_high,
            "v2_minus_visual_significant": (v2_low > 0) or (v2_high < 0),
        })

    decile1_df = pd.DataFrame(decile1_rows)
    decile1_path = OUTPUT_DIR / "lu_decile1_bootstrap_ci.csv"
    decile1_df.to_csv(decile1_path, index=False)

    # ------------------------------------------------------------
    # 3. Bottom-3 vs top-3 pooled bucket analysis
    # ------------------------------------------------------------
    pooled_rows = []
    contrast_rows = []

    for dataset in df["Dataset"].unique():
        ddf = df[df["Dataset"] == dataset].copy()

        buckets = {
            "bottom_3_low_LU": [1, 2, 3],
            "top_3_high_LU": [8, 9, 10],
        }

        bucket_stats = {}

        for bucket_name, deciles in buckets.items():
            sub = ddf[ddf["LU_decile"].isin(deciles)].copy()

            visual = sub["visual_correct@1"].to_numpy()
            lang = sub["language_correct@1"].to_numpy()
            v2 = sub["v2_correct@1"].to_numpy()

            lang_diff, lang_low, lang_high = bootstrap_paired_diff(lang, visual)
            v2_diff, v2_low, v2_high = bootstrap_paired_diff(v2, visual)

            pooled_rows.append({
                "Dataset": dataset,
                "Bucket": bucket_name,
                "LU_deciles": ",".join(map(str, deciles)),
                "n_queries": len(sub),
                "mean_LU": sub["LU"].mean(),
                "visual_R@1": 100.0 * visual.mean(),
                "language_only_R@1": 100.0 * lang.mean(),
                "v2_R@1": 100.0 * v2.mean(),
                "language_minus_visual_pp": lang_diff,
                "language_minus_visual_CI95_low": lang_low,
                "language_minus_visual_CI95_high": lang_high,
                "language_minus_visual_significant": (lang_low > 0) or (lang_high < 0),
                "v2_minus_visual_pp": v2_diff,
                "v2_minus_visual_CI95_low": v2_low,
                "v2_minus_visual_CI95_high": v2_high,
                "v2_minus_visual_significant": (v2_low > 0) or (v2_high < 0),
            })

            bucket_stats[bucket_name] = {
                "visual": visual,
                "language": lang,
                "v2": v2,
            }

        # Independent contrast: low-LU language-only R@1 minus high-LU language-only R@1
        low_lang = bucket_stats["bottom_3_low_LU"]["language"]
        high_lang = bucket_stats["top_3_high_LU"]["language"]

        lang_contrast, lang_contrast_low, lang_contrast_high = bootstrap_independent_diff(
            low_lang,
            high_lang,
        )

        low_v2 = bucket_stats["bottom_3_low_LU"]["v2"]
        high_v2 = bucket_stats["top_3_high_LU"]["v2"]

        v2_contrast, v2_contrast_low, v2_contrast_high = bootstrap_independent_diff(
            low_v2,
            high_v2,
        )

        contrast_rows.append({
            "Dataset": dataset,
            "Contrast": "bottom_3_low_LU minus top_3_high_LU",
            "language_only_R@1_diff_pp": lang_contrast,
            "language_only_CI95_low": lang_contrast_low,
            "language_only_CI95_high": lang_contrast_high,
            "language_only_significant": (lang_contrast_low > 0) or (lang_contrast_high < 0),
            "v2_R@1_diff_pp": v2_contrast,
            "v2_CI95_low": v2_contrast_low,
            "v2_CI95_high": v2_contrast_high,
            "v2_significant": (v2_contrast_low > 0) or (v2_contrast_high < 0),
        })

    pooled_df = pd.DataFrame(pooled_rows)
    contrast_df = pd.DataFrame(contrast_rows)

    pooled_path = OUTPUT_DIR / "lu_bottom_top_pooled_bucket_ci.csv"
    contrast_path = OUTPUT_DIR / "lu_bottom_top_contrast_ci.csv"

    pooled_df.to_csv(pooled_path, index=False)
    contrast_df.to_csv(contrast_path, index=False)

    print("\nSaved:")
    print(decile_curve_path)
    print(decile1_path)
    print(pooled_path)
    print(contrast_path)

    print("\nFULL 10-DECILE CURVE")
    print(decile_df.to_string(index=False))

    print("\nDECILE-1 BOOTSTRAP CI")
    print(decile1_df.to_string(index=False))

    print("\nBOTTOM-3 VS TOP-3 POOLED BUCKET CI")
    print(pooled_df.to_string(index=False))

    print("\nBOTTOM-3 MINUS TOP-3 CONTRAST CI")
    print(contrast_df.to_string(index=False))


if __name__ == "__main__":
    main()
