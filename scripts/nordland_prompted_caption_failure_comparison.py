
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/content/vpr-language-fusion")
sys.path.insert(0, str(REPO_ROOT / "src"))

from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
)


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

ORIGINAL_CAPTIONS = DRIVE_BASE / "nordland_clean_descriptions.csv"
PROMPTED_CAPTIONS = DRIVE_BASE / "nordland_clean_prompted_descriptions.csv"

VISUAL_PATH = EMBED_DIR / "nordland_clean_visual_sim_matrix.npy"
LANGUAGE_PATH = EMBED_DIR / "nordland_clean_lang_sim_matrix.npy"
POSITIVE_PATH = EMBED_DIR / "nordland_clean_positive_matrix.npy"

UNCERTAINTY_K = 10


STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "and", "or", "with", "at", "by",
    "for", "from", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "there", "it", "as", "into", "over", "under", "next", "near",
    "view", "image", "photo", "picture", "scene", "showing", "shows"
}


def clean_tokens(text):
    text = str(text).lower()
    words = re.findall(r"[a-zA-Z]+", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def token_count(text):
    return len(clean_tokens(text))


def unique_token_count(text):
    return len(set(clean_tokens(text)))


def jaccard(a, b):
    A = set(clean_tokens(a))
    B = set(clean_tokens(b))
    if len(A | B) == 0:
        return 0.0
    return len(A & B) / len(A | B)


def caption_col(df):
    if "description" in df.columns:
        return "description"
    if "caption" in df.columns:
        return "caption"
    raise ValueError(f"Could not find caption column. Columns found: {list(df.columns)}")


def positive_indices_for_query(positive_row, max_items=5):
    idx = np.where(positive_row)[0]
    return idx[:max_items].tolist()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visual_sim = np.load(VISUAL_PATH)
    language_sim = np.load(LANGUAGE_PATH)
    positive = np.load(POSITIVE_PATH).astype(bool)

    n_queries, n_db = visual_sim.shape

    original_df = pd.read_csv(ORIGINAL_CAPTIONS)
    prompted_df = pd.read_csv(PROMPTED_CAPTIONS)

    orig_col = caption_col(original_df)
    prompt_col = caption_col(prompted_df)

    print("Original captions:", ORIGINAL_CAPTIONS)
    print("Prompted captions:", PROMPTED_CAPTIONS)
    print("Original shape:", original_df.shape)
    print("Prompted shape:", prompted_df.shape)
    print("Caption columns:", orig_col, prompt_col)
    print("n_queries:", n_queries, "n_db:", n_db)

    if len(original_df) < n_db + n_queries:
        raise ValueError("Original caption file does not contain DB + query rows.")
    if len(prompted_df) < n_db + n_queries:
        raise ValueError("Prompted caption file does not contain DB + query rows.")

    # Compute LU over visual top-k candidates
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :UNCERTAINTY_K]
    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    low_lu_threshold = np.quantile(lu, 0.10)
    low_lu_mask = lu <= low_lu_threshold

    visual_top1 = np.argmax(visual_sim, axis=1)
    language_top1 = np.argmax(language_sim, axis=1)

    visual_top1_correct = positive[np.arange(n_queries), visual_top1]
    language_top1_correct = positive[np.arange(n_queries), language_top1]

    # Core failure case:
    # language appears confident / low-LU, visual top1 was correct, but language-only top1 is wrong.
    failure_mask = low_lu_mask & visual_top1_correct & (~language_top1_correct)
    failure_query_ids = np.where(failure_mask)[0]

    print("\nLow-LU threshold:", low_lu_threshold)
    print("Low-LU queries:", int(low_lu_mask.sum()))
    print("Low-LU + visual correct + language wrong failures:", len(failure_query_ids))

    rows = []

    for qid in failure_query_ids:
        v_idx = int(visual_top1[qid])
        l_idx = int(language_top1[qid])
        pos_list = positive_indices_for_query(positive[qid], max_items=5)
        pos_idx = int(pos_list[0]) if len(pos_list) > 0 else -1

        # In the caption CSVs:
        # rows 0:n_db are database images
        # rows n_db:n_db+n_queries are query images
        q_row = n_db + qid

        orig_query = str(original_df.loc[q_row, orig_col])
        orig_positive = str(original_df.loc[pos_idx, orig_col]) if pos_idx >= 0 else ""
        orig_visual_top1 = str(original_df.loc[v_idx, orig_col])
        orig_language_top1 = str(original_df.loc[l_idx, orig_col])

        prompt_query = str(prompted_df.loc[q_row, prompt_col])
        prompt_positive = str(prompted_df.loc[pos_idx, prompt_col]) if pos_idx >= 0 else ""
        prompt_visual_top1 = str(prompted_df.loc[v_idx, prompt_col])
        prompt_language_top1 = str(prompted_df.loc[l_idx, prompt_col])

        rows.append({
            "query_id": int(qid),
            "SU": float(su[qid]),
            "LU": float(lu[qid]),
            "positive_db_indices_first5": str(pos_list),
            "visual_top1_idx": v_idx,
            "language_top1_idx": l_idx,
            "language_similarity_to_wrong_top1": float(language_sim[qid, l_idx]),
            "language_similarity_to_positive": float(language_sim[qid, pos_idx]) if pos_idx >= 0 else np.nan,

            "original_query_caption": orig_query,
            "original_positive_caption": orig_positive,
            "original_visual_top1_caption": orig_visual_top1,
            "original_language_wrong_top1_caption": orig_language_top1,

            "prompted_query_caption": prompt_query,
            "prompted_positive_caption": prompt_positive,
            "prompted_visual_top1_caption": prompt_visual_top1,
            "prompted_language_wrong_top1_caption": prompt_language_top1,

            "orig_query_token_count": token_count(orig_query),
            "prompt_query_token_count": token_count(prompt_query),
            "orig_positive_token_count": token_count(orig_positive),
            "prompt_positive_token_count": token_count(prompt_positive),
            "orig_wrong_language_token_count": token_count(orig_language_top1),
            "prompt_wrong_language_token_count": token_count(prompt_language_top1),

            "orig_query_unique_tokens": unique_token_count(orig_query),
            "prompt_query_unique_tokens": unique_token_count(prompt_query),
            "orig_positive_unique_tokens": unique_token_count(orig_positive),
            "prompt_positive_unique_tokens": unique_token_count(prompt_positive),

            "orig_query_vs_wrong_jaccard": jaccard(orig_query, orig_language_top1),
            "prompt_query_vs_wrong_jaccard": jaccard(prompt_query, prompt_language_top1),
            "orig_query_vs_positive_jaccard": jaccard(orig_query, orig_positive),
            "prompt_query_vs_positive_jaccard": jaccard(prompt_query, prompt_positive),

            "query_token_gain_prompt_minus_original": token_count(prompt_query) - token_count(orig_query),
            "positive_token_gain_prompt_minus_original": token_count(prompt_positive) - token_count(orig_positive),
        })

    out_df = pd.DataFrame(rows)

    out_csv = OUTPUT_DIR / "nordland_prompted_caption_low_lu_failure_comparison.csv"
    out_md = OUTPUT_DIR / "nordland_prompted_caption_low_lu_failure_summary.md"

    out_df.to_csv(out_csv, index=False)

    print("\nSaved CSV:", out_csv)

    if len(out_df) == 0:
        print("No matching failure cases found.")
        return

    summary = {
        "n_failures": len(out_df),
        "mean_orig_query_tokens": out_df["orig_query_token_count"].mean(),
        "mean_prompt_query_tokens": out_df["prompt_query_token_count"].mean(),
        "mean_query_token_gain": out_df["query_token_gain_prompt_minus_original"].mean(),
        "mean_orig_positive_tokens": out_df["orig_positive_token_count"].mean(),
        "mean_prompt_positive_tokens": out_df["prompt_positive_token_count"].mean(),
        "mean_positive_token_gain": out_df["positive_token_gain_prompt_minus_original"].mean(),
        "mean_orig_query_wrong_jaccard": out_df["orig_query_vs_wrong_jaccard"].mean(),
        "mean_prompt_query_wrong_jaccard": out_df["prompt_query_vs_wrong_jaccard"].mean(),
        "mean_orig_query_positive_jaccard": out_df["orig_query_vs_positive_jaccard"].mean(),
        "mean_prompt_query_positive_jaccard": out_df["prompt_query_vs_positive_jaccard"].mean(),
    }

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")

    md_lines = []
    md_lines.append("# Nordland prompted-caption comparison for low-LU failures\n")
    md_lines.append("This diagnostic compares original captions and prompted captions for low-LU Nordland cases where visual top-1 is correct but language-only top-1 is wrong.\n")
    md_lines.append("## Aggregate summary\n")
    for k, v in summary.items():
        md_lines.append(f"- **{k}**: {v}\n")

    md_lines.append("\n## Example cases\n")

    for _, row in out_df.head(12).iterrows():
        md_lines.append(f"\n### Query {int(row['query_id'])}\n")
        md_lines.append(f"- LU: {row['LU']:.6f}\n")
        md_lines.append(f"- Positive DB indices: {row['positive_db_indices_first5']}\n")
        md_lines.append("\n**Original captions**\n")
        md_lines.append(f"- Query: {row['original_query_caption']}\n")
        md_lines.append(f"- True positive: {row['original_positive_caption']}\n")
        md_lines.append(f"- Wrong language top-1: {row['original_language_wrong_top1_caption']}\n")
        md_lines.append("\n**Prompted captions**\n")
        md_lines.append(f"- Query: {row['prompted_query_caption']}\n")
        md_lines.append(f"- True positive: {row['prompted_positive_caption']}\n")
        md_lines.append(f"- Wrong language top-1: {row['prompted_language_wrong_top1_caption']}\n")

    out_md.write_text("\n".join(md_lines))

    print("Saved Markdown:", out_md)

    print("\nFIRST 8 CASES")
    show_cols = [
        "query_id",
        "LU",
        "original_query_caption",
        "original_positive_caption",
        "original_language_wrong_top1_caption",
        "prompted_query_caption",
        "prompted_positive_caption",
        "prompted_language_wrong_top1_caption",
        "query_token_gain_prompt_minus_original",
        "positive_token_gain_prompt_minus_original",
    ]

    pd.set_option("display.max_colwidth", 120)
    print(out_df[show_cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
