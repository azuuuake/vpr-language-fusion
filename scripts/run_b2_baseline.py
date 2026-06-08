"""
run_b2_baseline.py

Reproduces the fixed-fusion B2 baseline.

B2 uses a fixed alpha=0.5 fusion between:
    - EigenPlaces visual similarity
    - BGE-large language similarity

Usage:
    python scripts/run_b2_baseline.py --dataset amstertime
    python scripts/run_b2_baseline.py --dataset msls_val
    python scripts/run_b2_baseline.py --dataset nordland_clean

Expected R@1 / R@5 / R@10:
    amstertime:      43.6 / 65.8 / 73.4
    msls_val:        84.5 / 91.5 / 93.2
    nordland_clean:  45.8 / 75.0 / 83.8
"""

import argparse
from pathlib import Path

import numpy as np


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"

EXPECTED = {
    "amstertime": "43.6 / 65.8 / 73.4",
    "msls_val": "84.5 / 91.5 / 93.2",
    "nordland_clean": "45.8 / 75.0 / 83.8",
}

FILES = {
    "amstertime": {
        "visual": "amstertime_visual_sim_matrix.npy",
        "language": "amstertime_lang_sim_matrix.npy",
        "positive": "amstertime_positive_matrix.npy",
    },
    "msls_val": {
        "visual": "msls_val_visual_sim_matrix.npy",
        "language": "msls_val_lang_sim_matrix.npy",
        "positive": "msls_val_positive_matrix.npy",
    },
    "nordland_clean": {
        "visual": "nordland_clean_visual_sim_matrix.npy",
        "language": "nordland_clean_lang_sim_matrix.npy",
        "positive": "nordland_clean_positive_matrix.npy",
    },
}


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def compute_recall(sim_matrix: np.ndarray, positive_matrix: np.ndarray, k: int) -> float:
    top_k = np.argsort(-sim_matrix, axis=1)[:, :k]

    correct = 0
    for i in range(sim_matrix.shape[0]):
        if positive_matrix[i, top_k[i]].any():
            correct += 1

    return 100.0 * correct / sim_matrix.shape[0]


def run(dataset: str, alpha: float) -> None:
    if dataset not in FILES:
        raise ValueError(f"Unknown dataset: {dataset}")

    files = FILES[dataset]

    visual = load_matrix(EMBED_DIR / files["visual"])
    language = load_matrix(EMBED_DIR / files["language"])
    positive = load_matrix(EMBED_DIR / files["positive"]).astype(bool)

    if visual.shape != language.shape:
        raise ValueError("Visual and language similarity matrices have different shapes.")

    if visual.shape != positive.shape:
        raise ValueError("Visual and positive matrices have different shapes.")

    fused = alpha * visual + (1.0 - alpha) * language

    r1 = compute_recall(fused, positive, 1)
    r5 = compute_recall(fused, positive, 5)
    r10 = compute_recall(fused, positive, 10)

    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Fixed fusion alpha: {alpha}")
    print("=" * 70)
    print(f"R@1:  {r1:.1f}")
    print(f"R@5:  {r5:.1f}")
    print(f"R@10: {r10:.1f}")
    print("-" * 70)
    print(f"Expected R@1 / R@5 / R@10: {EXPECTED[dataset]}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amstertime", "msls_val", "nordland_clean"],
    )
    parser.add_argument("--alpha", type=float, default=0.5)

    args = parser.parse_args()
    run(dataset=args.dataset, alpha=args.alpha)


if __name__ == "__main__":
    main()
