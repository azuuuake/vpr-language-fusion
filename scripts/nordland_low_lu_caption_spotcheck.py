
import numpy as np
import pandas as pd
from pathlib import Path


REPO_ROOT = Path("/content/vpr-language-fusion")
DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

QUERY_LEVEL_PATH = OUTPUT_DIR / "lu_query_level_validation.csv"

DESC_PATH = DRIVE_BASE / "nordland_clean_descriptions.csv"

VISUAL_PATH = EMBED_DIR / "nordland_clean_visual_sim_matrix.npy"
LANG_PATH = EMBED_DIR / "nordland_clean_lang_sim_matrix.npy"
POSITIVE_PATH = EMBED_DIR / "nordland_clean_positive_matrix.npy"

TOP_K = 10
N_EXAMPLES = 15


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False})


def get_desc(df, idx):
    if idx < 0 or idx >= len(df):
        return "[index out of range]"
    row = df.iloc[idx]
    return f"{row['image_path']} | {row['description']}"


def main():
    qdf = pd.read_csv(QUERY_LEVEL_PATH)

    qdf["visual_correct@1"] = as_bool(qdf["visual_correct@1"])
    qdf["language_correct@1"] = as_bool(qdf["language_correct@1"])
    qdf["v2_correct@1"] = as_bool(qdf["v2_correct@1"])

    nord = qdf[
        (qdf["Dataset"] == "Nordland-clean")
        & (qdf["LU_decile"] == 1)
    ].copy()

    # Cases where vision was correct but language-only was wrong.
    failures = nord[
        (nord["visual_correct@1"] == True)
        & (nord["language_correct@1"] == False)
    ].copy()

    failures = failures.sort_values("LU").head(N_EXAMPLES)

    visual_sim = np.load(VISUAL_PATH)
    lang_sim = np.load(LANG_PATH)
    positive = np.load(POSITIVE_PATH).astype(bool)

    desc_df = pd.read_csv(DESC_PATH)

    # The file has 800 rows: usually 400 database + 400 query images.
    n_db = visual_sim.shape[1]
    n_query = visual_sim.shape[0]

    db_desc = desc_df.iloc[:n_db].reset_index(drop=True)
    query_desc = desc_df.iloc[n_db:n_db+n_query].reset_index(drop=True)

    print("Description file:", DESC_PATH)
    print("desc shape:", desc_df.shape)
    print("visual matrix shape:", visual_sim.shape)
    print("db_desc shape:", db_desc.shape)
    print("query_desc shape:", query_desc.shape)

    print("\nNordland LU decile 1 total queries:", len(nord))
    print("Visual-correct but language-wrong cases:", len(failures))
    print("Showing examples:", len(failures))

    rows = []

    for _, r in failures.iterrows():
        qid = int(r["query_id"])

        topk_idx = np.argsort(-visual_sim[qid])[:TOP_K]

        visual_top1 = int(topk_idx[0])

        lang_scores_topk = lang_sim[qid, topk_idx]
        lang_order = np.argsort(-lang_scores_topk)
        language_top1 = int(topk_idx[lang_order[0]])

        positive_idx = np.where(positive[qid])[0]
        first_positive = int(positive_idx[0]) if len(positive_idx) > 0 else -1

        row = {
            "query_id": qid,
            "LU": float(r["LU"]),
            "SU": float(r["SU"]),
            "alpha": float(r["alpha"]),
            "visual_top1_idx": visual_top1,
            "language_top1_idx": language_top1,
            "first_positive_idx": first_positive,
            "query_caption": get_desc(query_desc, qid),
            "visual_top1_caption": get_desc(db_desc, visual_top1),
            "language_top1_caption": get_desc(db_desc, language_top1),
            "positive_caption": get_desc(db_desc, first_positive),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "nordland_low_lu_caption_spotcheck.csv"
    out.to_csv(out_path, index=False)

    print("\nSaved:", out_path)

    print("\nSPOT-CHECK EXAMPLES")
    print("=" * 100)

    for i, row in out.iterrows():
        print("\n" + "-" * 100)
        print(f"Example {i+1}")
        print(f"query_id: {row['query_id']}")
        print(f"LU: {row['LU']:.6f}, SU: {row['SU']:.6f}, alpha: {row['alpha']:.6f}")
        print(f"visual_top1_idx: {row['visual_top1_idx']}")
        print(f"language_top1_idx: {row['language_top1_idx']}")
        print(f"first_positive_idx: {row['first_positive_idx']}")

        print("\nQUERY CAPTION:")
        print(row["query_caption"])

        print("\nVISUAL TOP-1 CAPTION:")
        print(row["visual_top1_caption"])

        print("\nLANGUAGE TOP-1 CAPTION:")
        print(row["language_top1_caption"])

        print("\nPOSITIVE CAPTION:")
        print(row["positive_caption"])


if __name__ == "__main__":
    main()
