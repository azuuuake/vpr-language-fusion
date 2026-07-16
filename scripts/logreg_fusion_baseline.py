
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path("/content/vpr-language-fusion")
sys.path.insert(0, str(REPO_ROOT / "src"))

from uncertainty_v2 import (
    visual_uncertainty,
    language_uncertainty,
    normalise_uncertainty,
    uncertainty_to_variance,
    precision_weighted_fusion,
)


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")
EMBED_DIR = DRIVE_BASE / "embeddings"
OUTPUT_DIR = REPO_ROOT / "results" / "redesign_v2"

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

FEATURE_NAMES = [
    "visual_score",
    "language_score",
    "visual_rank_norm",
    "language_rank_norm",
    "visual_margin_from_top",
    "language_margin_from_top",
    "SU_raw",
    "LU_raw",
    "SU_norm",
    "LU_norm",
    "sigma_v2",
    "sigma_l2",
    "alpha_visual_weight",
    "language_weight_1_minus_alpha",
    "visual_x_language",
    "language_minus_visual",
]


def ensure_msls_split():
    split_path = OUTPUT_DIR / "msls_calibration_split_indices.npz"

    if split_path.exists():
        return split_path

    print("Split file missing. Recreating fixed MSLS split with seed=42.")

    msls_visual = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    n_queries = msls_visual.shape[0]

    seed = 42
    calibration_fraction = 0.30

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_queries)

    n_calibration = int(round(calibration_fraction * n_queries))
    calibration_indices = np.sort(indices[:n_calibration])
    evaluation_indices = np.sort(indices[n_calibration:])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        split_path,
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        seed=seed,
        calibration_fraction=calibration_fraction,
        n_queries=n_queries,
        n_calibration=len(calibration_indices),
        n_evaluation=len(evaluation_indices),
    )

    return split_path


def load_locked_setting():
    path = OUTPUT_DIR / "calibrated_v2_selection_locked_rule.json"

    if not path.exists():
        print("Warning: selection JSON missing. Using locked fallback values.")
        return {
            "selected_rho": 3.0,
            "selected_candidate_pool": 10,
            "uncertainty_k": 10,
        }

    with open(path, "r") as f:
        return json.load(f)


def retrieved_from_scores(score_matrix):
    return np.argsort(-score_matrix, axis=1)


def correct_at_k(retrieved, positive, k):
    n = retrieved.shape[0]
    out = np.zeros(n, dtype=bool)

    for i in range(n):
        out[i] = positive[i, retrieved[i, :k]].any()

    return out


def metric_row(dataset, method, retrieved, positive):
    return {
        "Dataset": dataset,
        "Method": method,
        "R@1": 100.0 * correct_at_k(retrieved, positive, 1).mean(),
        "R@5": 100.0 * correct_at_k(retrieved, positive, 5).mean(),
        "R@10": 100.0 * correct_at_k(retrieved, positive, 10).mean(),
        "count@1": int(correct_at_k(retrieved, positive, 1).sum()),
        "n_queries": int(retrieved.shape[0]),
    }


def query_uncertainties(visual_sim, language_sim, uncertainty_k):
    topk_idx = np.argsort(-visual_sim, axis=1)[:, :uncertainty_k]

    visual_topk = np.take_along_axis(visual_sim, topk_idx, axis=1)
    language_topk = np.take_along_axis(language_sim, topk_idx, axis=1)

    su = visual_uncertainty(visual_topk)
    lu = language_uncertainty(language_topk)

    su_norm = normalise_uncertainty(su)
    lu_norm = normalise_uncertainty(lu)

    sigma_v2 = uncertainty_to_variance(su_norm)
    sigma_l2 = uncertainty_to_variance(lu_norm)

    return su, lu, su_norm, lu_norm, sigma_v2, sigma_l2


def v2_retrieved(visual_sim, language_sim, uncertainty_k, candidate_pool, rho):
    su, lu, su_norm, lu_norm, sigma_v2, sigma_l2 = query_uncertainties(
        visual_sim=visual_sim,
        language_sim=language_sim,
        uncertainty_k=uncertainty_k,
    )

    pool_idx = np.argsort(-visual_sim, axis=1)[:, :candidate_pool]

    visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

    fused_pool, alpha = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    order = np.argsort(-fused_pool, axis=1)
    retrieved = np.take_along_axis(pool_idx, order, axis=1)

    return retrieved, alpha


def build_candidate_features(visual_sim, language_sim, positive, uncertainty_k, candidate_pool, rho):
    n_queries = visual_sim.shape[0]

    su, lu, su_norm, lu_norm, sigma_v2, sigma_l2 = query_uncertainties(
        visual_sim=visual_sim,
        language_sim=language_sim,
        uncertainty_k=uncertainty_k,
    )

    pool_idx = np.argsort(-visual_sim, axis=1)[:, :candidate_pool]

    visual_pool = np.take_along_axis(visual_sim, pool_idx, axis=1)
    language_pool = np.take_along_axis(language_sim, pool_idx, axis=1)

    _, alpha = precision_weighted_fusion(
        visual_sims=visual_pool,
        language_sims=language_pool,
        su_norm=su_norm,
        lu_norm=lu_norm,
        rho=rho,
    )

    visual_rank = np.tile(np.arange(candidate_pool), (n_queries, 1))
    visual_rank_norm = visual_rank / max(candidate_pool - 1, 1)

    language_rank = np.argsort(np.argsort(-language_pool, axis=1), axis=1)
    language_rank_norm = language_rank / max(candidate_pool - 1, 1)

    visual_margin = visual_pool[:, [0]] - visual_pool
    language_margin = language_pool.max(axis=1, keepdims=True) - language_pool

    X = np.column_stack([
        visual_pool.ravel(),
        language_pool.ravel(),
        visual_rank_norm.ravel(),
        language_rank_norm.ravel(),
        visual_margin.ravel(),
        language_margin.ravel(),
        np.repeat(su, candidate_pool),
        np.repeat(lu, candidate_pool),
        np.repeat(su_norm, candidate_pool),
        np.repeat(lu_norm, candidate_pool),
        np.repeat(sigma_v2, candidate_pool),
        np.repeat(sigma_l2, candidate_pool),
        np.repeat(alpha, candidate_pool),
        np.repeat(1.0 - alpha, candidate_pool),
        (visual_pool * language_pool).ravel(),
        (language_pool - visual_pool).ravel(),
    ])

    y = positive[np.arange(n_queries)[:, None], pool_idx].ravel().astype(int)

    return X, y, pool_idx


def logreg_retrieved(model, visual_sim, language_sim, positive, uncertainty_k, candidate_pool, rho):
    X, y, pool_idx = build_candidate_features(
        visual_sim=visual_sim,
        language_sim=language_sim,
        positive=positive,
        uncertainty_k=uncertainty_k,
        candidate_pool=candidate_pool,
        rho=rho,
    )

    prob = model.predict_proba(X)[:, 1].reshape(visual_sim.shape[0], candidate_pool)
    order = np.argsort(-prob, axis=1)
    retrieved = np.take_along_axis(pool_idx, order, axis=1)

    return retrieved, prob


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = load_locked_setting()
    rho = float(selection["selected_rho"])
    candidate_pool = int(selection["selected_candidate_pool"])
    uncertainty_k = int(selection["uncertainty_k"])

    split_path = ensure_msls_split()
    split = np.load(split_path)

    calibration_indices = split["calibration_indices"]
    evaluation_indices = split["evaluation_indices"]

    print("Frozen setting:")
    print("rho:", rho)
    print("candidate_pool:", candidate_pool)
    print("uncertainty_k:", uncertainty_k)
    print("MSLS calibration:", len(calibration_indices))
    print("MSLS held-out:", len(evaluation_indices))

    # Train on MSLS calibration only
    msls_visual = np.load(EMBED_DIR / "msls_val_visual_sim_matrix.npy")
    msls_language = np.load(EMBED_DIR / "msls_val_lang_sim_matrix.npy")
    msls_positive = np.load(EMBED_DIR / "msls_val_positive_matrix.npy").astype(bool)

    train_visual = msls_visual[calibration_indices]
    train_language = msls_language[calibration_indices]
    train_positive = msls_positive[calibration_indices]

    X_train, y_train, _ = build_candidate_features(
        visual_sim=train_visual,
        language_sim=train_language,
        positive=train_positive,
        uncertainty_k=uncertainty_k,
        candidate_pool=candidate_pool,
        rho=rho,
    )

    print("\nTRAINING DATA")
    print("Candidate samples:", X_train.shape[0])
    print("Features:", X_train.shape[1])
    print("Positive candidates:", int(y_train.sum()))
    print("Negative candidates:", int((1 - y_train).sum()))
    print("Positive rate:", float(y_train.mean()))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
    )

    model.fit(X_train, y_train)

    coef = model.named_steps["logisticregression"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)

    coef_path = OUTPUT_DIR / "logreg_fusion_baseline_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    print("\nTOP LOGISTIC-REGRESSION COEFFICIENTS")
    print(coef_df.head(16).to_string(index=False))

    rows = []

    for key, cfg in DATASETS.items():
        pretty = cfg["pretty"]

        visual_sim = np.load(EMBED_DIR / cfg["visual"])
        language_sim = np.load(EMBED_DIR / cfg["language"])
        positive = np.load(EMBED_DIR / cfg["positive"]).astype(bool)

        if cfg["split"] == "heldout":
            visual_sim = visual_sim[evaluation_indices]
            language_sim = language_sim[evaluation_indices]
            positive = positive[evaluation_indices]

        print("\n" + "=" * 80)
        print(pretty)
        print("=" * 80)

        visual_retrieved = retrieved_from_scores(visual_sim)
        language_retrieved = retrieved_from_scores(language_sim)
        fixed_retrieved = retrieved_from_scores(0.5 * visual_sim + 0.5 * language_sim)

        v2_ret, alpha = v2_retrieved(
            visual_sim=visual_sim,
            language_sim=language_sim,
            uncertainty_k=uncertainty_k,
            candidate_pool=candidate_pool,
            rho=rho,
        )

        logreg_ret, logreg_prob = logreg_retrieved(
            model=model,
            visual_sim=visual_sim,
            language_sim=language_sim,
            positive=positive,
            uncertainty_k=uncertainty_k,
            candidate_pool=candidate_pool,
            rho=rho,
        )

        rows.append(metric_row(pretty, "B1_visual_only_full_db", visual_retrieved, positive))
        rows.append(metric_row(pretty, "Language_only_full_db", language_retrieved, positive))
        rows.append(metric_row(pretty, "B2_fixed_0.5_full_db", fixed_retrieved, positive))
        rows.append(metric_row(pretty, "V2_precision_weighted_pool10", v2_ret, positive))
        rows.append(metric_row(pretty, "LogReg_learned_reranker_pool10", logreg_ret, positive))

        tmp = pd.DataFrame([r for r in rows if r["Dataset"] == pretty])
        print(tmp.to_string(index=False))

    metrics_df = pd.DataFrame(rows)

    metrics_path = OUTPUT_DIR / "logreg_fusion_baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    train_summary = {
        "rho": rho,
        "candidate_pool": candidate_pool,
        "uncertainty_k": uncertainty_k,
        "train_dataset": "MSLS-val calibration split only",
        "n_calibration_queries": int(len(calibration_indices)),
        "n_heldout_queries": int(len(evaluation_indices)),
        "candidate_samples": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "positive_candidates": int(y_train.sum()),
        "negative_candidates": int((1 - y_train).sum()),
        "positive_rate": float(y_train.mean()),
        "model": "StandardScaler + LogisticRegression(class_weight='balanced', C=1.0, random_state=42)",
    }

    summary_path = OUTPUT_DIR / "logreg_fusion_baseline_train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(train_summary, f, indent=2)

    print("\nSaved:")
    print(metrics_path)
    print(coef_path)
    print(summary_path)

    print("\nFINAL METRICS TABLE")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
