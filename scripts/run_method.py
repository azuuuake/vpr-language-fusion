"""
run_method.py

Reproduces the Confidence-Gated-Top10 method results.

Usage:
    python scripts/run_method.py --dataset amstertime --tau 1.0
    python scripts/run_method.py --dataset msls_val --tau 1.0
    python scripts/run_method.py --dataset nordland_clean --tau 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from su_metric import sort_scores_desc, SU_normalised, confidence_gated_fusion


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
RESULTS_DIR = DRIVE_BASE / "results"

EXPECTED_R1 = {
    "amstertime": 43.2,
    "msls_val": 85.1,
    "nordland_clean": 53.0,
}

EXPECTED_FILES = {
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


def compute_recall_at_k(retrieved_indices: np.ndarray, positive_matrix: np.ndarray, k: int) -> float:
    correct = 0
    for i in range(retrieved_indices.shape[0]):
        if positive_matrix[i, retrieved_indices[i, :k]].any():
            correct += 1
    return 100.0 * correct / retrieved_indices.shape[0]


def run(dataset: str, tau: float, lang_threshold: float = 0.05, top_k: int = 10) -> None:
    if dataset not in EXPECTED_FILES:
        raise ValueError(f"Unknown dataset: {dataset}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = EXPECTED_FILES[dataset]

    visual_path = EMBED_DIR / files["visual"]
    language_path = EMBED_DIR / files["language"]
    positive_path = EMBED_DIR / files["positive"]

    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Tau: {tau}")
    print(f"Language threshold: {lang_threshold}")
    print(f"Top-K reranking: {top_k}")
    print("=" * 70)

    print("\nLoading matrices...")
    visual_sim = load_matrix(visual_path)
    language_sim = load_matrix(language_path)
    positive = load_matrix(positive_path).astype(bool)

    print("Visual sim shape:  ", visual_sim.shape)
    print("Language sim shape:", language_sim.shape)
    print("Positive shape:    ", positive.shape)

    if visual_sim.shape != language_sim.shape:
        raise ValueError("Visual and language matrices have different shapes.")

    if visual_sim.shape != positive.shape:
        raise ValueError("Visual and positive matrices have different shapes.")

    n_queries = visual_sim.shape[0]

    print("\nSelecting top-K visual candidates...")
    top_k_idx = np.argsort(-visual_sim, axis=1)[:, :top_k]
    visual_topk = np.take_along_axis(visual_sim, top_k_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, top_k_idx, axis=1)

    visual_topk = sort_scores_desc(visual_topk)

    print("Computing SU_normalised...")
    su_norm = SU_normalised(visual_topk)

    print("Applying confidence-gated fusion...")
    fused_topk, alpha = confidence_gated_fusion(
        visual_sims=visual_topk,
        lang_sims=language_topk,
        su_norm=su_norm,
        tau=tau,
        lang_threshold=lang_threshold,
    )

    print("Reranking top-K candidates...")
    rerank_order = np.argsort(-fused_topk, axis=1)
    reranked_indices = np.take_along_axis(top_k_idx, rerank_order, axis=1)

    r1 = compute_recall_at_k(reranked_indices, positive, 1)
    r5 = compute_recall_at_k(reranked_indices, positive, 5)
    r10 = compute_recall_at_k(reranked_indices, positive, 10)

    language_active = int((alpha < 1.0).sum())
    language_suppressed = int((alpha == 1.0).sum())

    print("\nRESULTS")
    print("-" * 70)
    print(f"R@1:  {r1:.1f}")
    print(f"R@5:  {r5:.1f}")
    print(f"R@10: {r10:.1f}")
    print(f"Language active:     {language_active}/{n_queries}")
    print(f"Language suppressed: {language_suppressed}/{n_queries}")
    print(f"Alpha mean: {alpha.mean():.3f}")
    print(f"Alpha std:  {alpha.std():.3f}")

    expected = EXPECTED_R1.get(dataset)
    if expected is not None:
        if abs(r1 - expected) < 0.3:
            print(f"Expected R@1: {expected:.1f} -> REPRODUCED")
        else:
            print(f"Expected R@1: {expected:.1f} -> MISMATCH, investigate")

    output_path = RESULTS_DIR / f"{dataset}_gated_alpha_tau{tau}_lang{lang_threshold}.npy"
    np.save(output_path, alpha)

    print("\nSaved alpha values to:")
    print(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amstertime", "msls_val", "nordland_clean"],
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lang_threshold", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    run(
        dataset=args.dataset,
        tau=args.tau,
        lang_threshold=args.lang_threshold,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
