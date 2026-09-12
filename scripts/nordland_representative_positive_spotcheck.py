import numpy as np
import pandas as pd
from pathlib import Path


REPO_ROOT = Path("/content/vpr-language-fusion")
EMBED_DIR = Path("/content/drive/MyDrive/vpr_research/embeddings")
DATA_DIR = Path("/content/drive/MyDrive/vpr_research")
OUT_DIR = REPO_ROOT / "results" / "redesign_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spotcheck_candidates = [
    OUT_DIR / "nordland_low_lu_caption_spotcheck.csv",
    OUT_DIR / "nordland_prompted_caption_low_lu_failure_comparison.csv",
]

spot_path = None
for p in spotcheck_candidates:
    if p.exists():
        spot_path = p
        break

if spot_path is None:
    raise FileNotFoundError("Could not find Nordland spot-check CSV.")

positive = np.load(EMBED_DIR / "nordland_clean_positive_matrix.npy").astype(bool)
visual = np.load(EMBED_DIR / "nordland_clean_visual_sim_matrix.npy")
language = np.load(EMBED_DIR / "nordland_clean_lang_sim_matrix.npy")

captions_path = DATA_DIR / "nordland_clean_descriptions.csv"
captions = pd.read_csv(captions_path)

db_caps = captions.iloc[:400].reset_index(drop=True)
query_caps = captions.iloc[400:800].reset_index(drop=True)

spot = pd.read_csv(spot_path)

print("Using spot-check file:", spot_path)
print("Spot-check columns:")
print(spot.columns.tolist())

# Try to find query id column robustly
qid_col = None
for candidate in ["query_id", "qid", "query_index", "query_idx"]:
    if candidate in spot.columns:
        qid_col = candidate
        break

if qid_col is None:
    raise ValueError("Could not identify query_id column in spot-check CSV.")

rows = []

for _, row in spot.iterrows():
    qid = int(row[qid_col])

    pos_ids = np.where(positive[qid])[0]

    # first positive = old edge-of-window style
    first_pos = int(pos_ids[0])

    # centre of positive tolerance window
    centre_pos = int(pos_ids[len(pos_ids) // 2])

    # representative positive = highest visual similarity inside tolerance window
    best_visual_pos = int(pos_ids[np.argmax(visual[qid, pos_ids])])

    # language top-1 full database
    lang_top1 = int(np.argmax(language[qid]))

    # language top-1 inside visual top-10 pool
    visual_top10 = np.argsort(-visual[qid])[:10]
    lang_top1_in_visual_top10 = int(visual_top10[np.argmax(language[qid, visual_top10])])

    rows.append({
        "case_id": len(rows) + 1,
        "query_id": qid,
        "n_positive_db_ids": len(pos_ids),

        "first_positive_db_id": first_pos,
        "centre_positive_db_id": centre_pos,
        "best_visual_positive_db_id": best_visual_pos,

        "language_top1_full_db_id": lang_top1,
        "language_top1_within_visual_top10_db_id": lang_top1_in_visual_top10,

        "query_caption": query_caps.loc[qid, "description"],
        "first_positive_caption": db_caps.loc[first_pos, "description"],
        "centre_positive_caption": db_caps.loc[centre_pos, "description"],
        "best_visual_positive_caption": db_caps.loc[best_visual_pos, "description"],
        "language_top1_full_db_caption": db_caps.loc[lang_top1, "description"],
        "language_top1_within_visual_top10_caption": db_caps.loc[lang_top1_in_visual_top10, "description"],

        "visual_score_first_positive": float(visual[qid, first_pos]),
        "visual_score_centre_positive": float(visual[qid, centre_pos]),
        "visual_score_best_positive": float(visual[qid, best_visual_pos]),
        "visual_score_language_top1_full_db": float(visual[qid, lang_top1]),
        "visual_score_language_top1_within_visual_top10": float(visual[qid, lang_top1_in_visual_top10]),

        "language_score_best_visual_positive": float(language[qid, best_visual_pos]),
        "language_score_language_top1_full_db": float(language[qid, lang_top1]),
        "language_score_language_top1_within_visual_top10": float(language[qid, lang_top1_in_visual_top10]),

        "author_corrected_failure_type": "",
        "author_corrected_notes": "",
    })

out = pd.DataFrame(rows)

out_csv = OUT_DIR / "nordland_representative_positive_caption_spotcheck.csv"
out.to_csv(out_csv, index=False)

note = """# Nordland Representative-Positive Caption Spot-check

This file corrects the earlier manual caption spot-check.

Nordland-clean uses a positive tolerance window rather than one unique positive database image per query.
Therefore, comparing a query caption with only the first positive database frame can be misleading because
the first positive may be an edge-of-window frame.

For each spot-check case, this file records:
- first positive in the tolerance window,
- centre positive in the tolerance window,
- best-visual positive inside the tolerance window,
- language top-1 over the full database,
- language top-1 inside the visual top-10 candidate pool.

The corrected qualitative interpretation should compare the language-selected wrong match against the
representative positive, especially the best-visual positive within the tolerance window.

This affects only the manual qualitative caption analysis. Quantitative R@K retrieval metrics are unaffected
because they are computed against the full positive tolerance window.
"""

note_path = OUT_DIR / "nordland_representative_positive_caption_spotcheck_note.md"
note_path.write_text(note)

print("\nREPRESENTATIVE-POSITIVE SPOT-CHECK TABLE")
print(out.to_string(index=False))

print("\nSaved:")
print(out_csv)
print(note_path)
