"""
run_gating_ablation.py

Verifies the component-ablation results used in the manuscript.

This script does not recompute the ablation from raw similarity matrices.
Instead, it validates the saved verified output:

    results/gating_ablation_results.csv

A full matrix-based rerun requires the saved visual, language, and positive
matrices under /content/drive/MyDrive/vpr_research/embeddings/.

For the initial stage, the verified CSV is the result evidence.
"""

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = REPO_ROOT / "results" / "gating_ablation_results.csv"


EXPECTED_R1 = {
    ("AmsterTime", "B1 visual-only"): 43.460601,
    ("AmsterTime", "B2 fixed fusion"): 43.623071,
    ("AmsterTime", "SU-only top10"): 43.054427,
    ("AmsterTime", "Pre-filter-only top10"): 43.704305,
    ("AmsterTime", "Margin-gated top10"): 43.054427,
    ("AmsterTime", "Full method top10"): 43.216897,

    ("MSLS-val", "B1 visual-only"): 84.864865,
    ("MSLS-val", "B2 fixed fusion"): 84.459459,
    ("MSLS-val", "SU-only top10"): 84.054054,
    ("MSLS-val", "Pre-filter-only top10"): 84.864865,
    ("MSLS-val", "Margin-gated top10"): 85.000000,
    ("MSLS-val", "Full method top10"): 85.135135,

    ("Nordland-clean", "B1 visual-only"): 52.750000,
    ("Nordland-clean", "B2 fixed fusion"): 45.750000,
    ("Nordland-clean", "SU-only top10"): 49.500000,
    ("Nordland-clean", "Pre-filter-only top10"): 51.750000,
    ("Nordland-clean", "Margin-gated top10"): 51.500000,
    ("Nordland-clean", "Full method top10"): 53.000000,
}


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_FILE}")

    df = pd.read_csv(RESULTS_FILE)

    required_columns = [
        "Dataset",
        "Method",
        "R@1",
        "R@5",
        "R@10",
        "count@1",
        "Language active",
        "Mean alpha",
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    print("=" * 80)
    print("Loaded component-ablation results")
    print("=" * 80)
    print(RESULTS_FILE)
    print()

    all_ok = True

    for (dataset, method), expected_r1 in EXPECTED_R1.items():
        row = df[(df["Dataset"] == dataset) & (df["Method"] == method)]

        if row.empty:
            print(f"MISSING: {dataset} | {method}")
            all_ok = False
            continue

        actual_r1 = float(row.iloc[0]["R@1"])
        ok = abs(actual_r1 - expected_r1) < 0.01

        status = "OK" if ok else "MISMATCH"
        print(
            f"{status:8s} {dataset:15s} | {method:25s} "
            f"R@1 actual={actual_r1:.6f}, expected={expected_r1:.6f}"
        )

        if not ok:
            all_ok = False

    print()
    print("=" * 80)
    print("Paper Table II rounded view")
    print("=" * 80)

    table = df.copy()
    table["R"] = table.apply(
        lambda r: f"{r['R@1']:.1f}/{r['R@5']:.1f}/{r['R@10']:.1f}",
        axis=1,
    )

    paper_table = table.pivot(index="Method", columns="Dataset", values="R")

    order = [
        "B1 visual-only",
        "B2 fixed fusion",
        "SU-only top10",
        "Pre-filter-only top10",
        "Margin-gated top10",
        "Full method top10",
    ]

    paper_table = paper_table.loc[order]
    print(paper_table)

    print()
    if all_ok:
        print("All component-ablation R@1 values match the manuscript evidence.")
    else:
        raise SystemExit("Some values do not match. Check the CSV.")


if __name__ == "__main__":
    main()
