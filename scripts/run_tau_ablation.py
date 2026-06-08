"""
run_tau_ablation.py

Runs tau ablation for Confidence-Gated-Top10 fusion.

Default use:
    python scripts/run_tau_ablation.py --dataset msls_val

Expected MSLS-val results:
    tau=0.5 -> R@1=85.1, R@5=91.1, R@10=92.8
    tau=1.0 -> R@1=85.1, R@5=91.1, R@10=92.8
    tau=2.0 -> R@1=85.0, R@5=91.1, R@10=92.8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from su_metric import sort_scores_desc, SU_normalised, confidence_gated_fusion


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
RESULTS_DIR = DRIVE_BASE / "results"

FILE_MAP = {
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


def run_single_tau(dataset: str, tau: float, lang_threshold: float = 0.05, top_k: int = 10) -> dict:
    files = FILE_MAP[dataset]

    visual_sim = load_matrix(EMBED_DIR / files["visual"])
    language_sim = load_matrix(EMBED_DIR / files["language"])
    positive = load_matrix(EMBED_DIR / files["positive"]).astype(bool)

    top_k_idx = np.argsort(-visual_sim, axis=1)[:, :top_k]
    visual_topk = np.take_along_axis(visual_sim, top_k_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, top_k_idx, axis=1)

    visual_topk = sort_scores_desc(visual_topk)
    su_norm = SU_normalised(visual_topk)

    fused_topk, alpha = confidence_gated_fusion(
        visual_sims=visual_topk,
        lang_sims=language_topk,
        su_norm=su_norm,
        tau=tau,
        lang_threshold=lang_threshold,
    )

    rerank_order = np.argsort(-fused_topk, axis=1)
    reranked_indices = np.take_along_axis(top_k_idx, rerank_order, axis=1)

    r1 = compute_recall_at_k(reranked_indices, positive, 1)
    r5 = compute_recall_at_k(reranked_indices, positive, 5)
    r10 = compute_recall_at_k(reranked_indices, positive, 10)

    language_active = int((alpha < 1.0).sum())
    language_suppressed = int((alpha == 1.0).sum())

    return {
        "dataset": dataset,
        "tau": tau,
        "top_k": top_k,
        "lang_threshold": lang_threshold,
        "mean_alpha": round(float(alpha.mean()), 3),
        "std_alpha": round(float(alpha.std()), 3),
        "language_active": language_active,
        "language_suppressed": language_suppressed,
        "R@1": round(r1, 3),
        "R@5": round(r5, 3),
        "R@10": round(r10, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="msls_val",
        choices=["amstertime", "msls_val", "nordland_clean"],
    )
    parser.add_argument("--taus", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--lang_threshold", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for tau in args.taus:
        print(f"Running {args.dataset} tau={tau}")
        row = run_single_tau(
            dataset=args.dataset,
            tau=tau,
            lang_threshold=args.lang_threshold,
            top_k=args.top_k,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\nTau ablation results:")
    print(df)

    output_path = RESULTS_DIR / f"{args.dataset}_tau_ablation.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
