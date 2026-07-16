
from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path("/content/drive/MyDrive/vpr_research/embeddings")
OUT = Path("/content/vpr-language-fusion/results/redesign_v2")
OUT.mkdir(parents=True, exist_ok=True)

FOLDERS = {
    "nordland_b1_descriptors": BASE / "nordland_b1_descriptors",
    "nordland_b1_descriptors_bad_alignment_check": BASE / "nordland_b1_descriptors_bad_alignment_check",
    "nordland_clean_b1_descriptors": BASE / "nordland_clean_b1_descriptors",
}


def l2_normalise(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def recall_at_k_from_diagonal_positive(q_desc, db_desc, ks=(1, 5, 10), chunk=256):
    q_desc = l2_normalise(q_desc.astype(np.float32))
    db_desc = l2_normalise(db_desc.astype(np.float32))

    n_q = q_desc.shape[0]
    n_db = db_desc.shape[0]

    if n_q != n_db:
        print("Warning: n_q != n_db. Diagonal-positive assumption may not apply.")

    max_k = max(ks)
    correct_counts = {k: 0 for k in ks}
    top1_indices = np.empty(n_q, dtype=np.int64)
    diag_scores = np.empty(n_q, dtype=np.float32)
    top1_scores = np.empty(n_q, dtype=np.float32)

    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        sims = q_desc[start:end] @ db_desc.T

        topk = np.argpartition(-sims, kth=max_k-1, axis=1)[:, :max_k]
        topk_scores = np.take_along_axis(sims, topk, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        topk_sorted = np.take_along_axis(topk, order, axis=1)

        for local_i, global_i in enumerate(range(start, end)):
            top1_indices[global_i] = topk_sorted[local_i, 0]
            top1_scores[global_i] = sims[local_i, top1_indices[global_i]]
            diag_scores[global_i] = sims[local_i, global_i] if global_i < n_db else np.nan

            for k in ks:
                if global_i in topk_sorted[local_i, :k]:
                    correct_counts[k] += 1

    recalls = {f"R@{k}": 100.0 * correct_counts[k] / n_q for k in ks}
    return recalls, top1_indices, diag_scores, top1_scores


def main():
    rows = []

    for name, folder in FOLDERS.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        q_path = folder / "queries_descriptors.npy"
        db_path = folder / "database_descriptors.npy"

        if not q_path.exists() or not db_path.exists():
            print("Missing descriptors.")
            continue

        q = np.load(q_path)
        db = np.load(db_path)

        print("Query descriptors:", q.shape, q.dtype)
        print("Database descriptors:", db.shape, db.dtype)

        recalls, top1_idx, diag_scores, top1_scores = recall_at_k_from_diagonal_positive(q, db)

        row = {
            "folder": name,
            "n_queries": q.shape[0],
            "n_database": db.shape[0],
            "dim": q.shape[1],
            **recalls,
            "mean_diag_score": float(np.nanmean(diag_scores)),
            "mean_top1_score": float(np.nanmean(top1_scores)),
            "top1_diagonal_match_count": int((top1_idx == np.arange(len(top1_idx))).sum()) if q.shape[0] == db.shape[0] else None,
        }

        rows.append(row)

        print(pd.DataFrame([row]).to_string(index=False))

        # Save first 50 top1 matches for manual inspection
        inspect = pd.DataFrame({
            "query_index": np.arange(min(50, len(top1_idx))),
            "top1_database_index": top1_idx[:50],
            "is_diagonal_match": top1_idx[:50] == np.arange(min(50, len(top1_idx))),
            "diag_score": diag_scores[:50],
            "top1_score": top1_scores[:50],
        })

        inspect_path = OUT / f"{name}_alignment_first50.csv"
        inspect.to_csv(inspect_path, index=False)
        print("Saved:", inspect_path)

    summary = pd.DataFrame(rows)
    out_path = OUT / "nordland_descriptor_alignment_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\nFINAL ALIGNMENT SUMMARY")
    print(summary.to_string(index=False))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
