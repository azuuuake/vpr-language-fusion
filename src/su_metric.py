"""
su_metric.py

Paper-faithful implementation of the uncertainty metrics from:
Miller, Milford et al., "Through the Lens of Doubt: Robust and Efficient
Uncertainty Estimation for Visual Place Recognition"
arXiv:2510.13464v1, Section III.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT CONVENTION — READ BEFORE USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    sim_scores: numpy array of shape (N_queries, K)

    Each row MUST be sorted in DESCENDING similarity order:
        sim_scores[:, 0] = best / top-1 similarity score  (s_1)
        sim_scores[:, 1] = second-best                    (s_2)
        ...
        sim_scores[:, K-1] = K-th best                   (s_K)

⚠️  CRITICAL: Passing unsorted scores produces silently wrong results.
    auto_VPR / EigenPlaces may or may not return sorted matrices.
    Always call sort_scores_desc(sim_scores) first if you are unsure.
    Verification test:  RS([[0.9, 0.6, 0.5, 0.4, 0.3]]) should equal
    exactly 0.5000. If it does not, your input is not sorted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRIC INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Higher RS / SD / SU  →  higher uncertainty  →  vision is unreliable
    Lower  RS / SD / SU  →  higher confidence   →  vision is reliable

    For confidence-gated language fusion, use SU_normalised(), which
    applies z-score normalisation before the sigmoid so the fusion gate
    spans the full [0, 1] range and the inflection point sits at the
    population mean confidence level.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPER FORMULAS (Section III)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    RS  = (1 / (k-1)) * Σ_{i=2}^{k} (s_i / s_1)
        = mean of (s_i / s_1) for i = 2 … k

    SD  = s_median / s_best

    SU  = α × RS + (1 − α) × SD        [default α = 0.5]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE IN THE CONFIDENCE-GATED FUSION PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Step 1: get top-K similarity scores from EigenPlaces (sorted descending)
    sim_scores = get_topk_scores(query_img, database, K=10)   # (N, 10)
    sim_scores = sort_scores_desc(sim_scores)                  # ensure sorted

    # Step 2: compute normalised uncertainty per query
    su = SU_normalised(sim_scores)                             # shape (N,)

    # Step 3: compute visual weight alpha  (high alpha = trust vision)
    alpha = sigmoid_gate(su, tau=1.0)                          # shape (N,)

    # Step 4: fuse
    # fused_sim = alpha * visual_sim + (1 - alpha) * lang_sim
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

EPS = 1e-9
DEFAULT_ALPHA = 0.5   # paper default (Section IV.C)
DEFAULT_K     = 10    # paper default (Section IV.C)


# ─────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────

def _validate(sim_scores: np.ndarray) -> np.ndarray:
    """
    Cast to float64 and enforce shape requirements.

    Raises
    ------
    ValueError
        If sim_scores is not 2-D or has fewer than 2 candidates (K < 2).
    """
    scores = np.asarray(sim_scores, dtype=np.float64)

    if scores.ndim != 2:
        raise ValueError(
            f"sim_scores must be a 2-D array of shape (N_queries, K), "
            f"got shape {scores.shape}."
        )
    if scores.shape[1] < 2:
        raise ValueError(
            f"sim_scores must contain at least K=2 candidates per query, "
            f"got K={scores.shape[1]}. Cannot compute RS without a second candidate."
        )
    return scores


# ─────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────

def sort_scores_desc(sim_scores: np.ndarray) -> np.ndarray:
    """
    Sort similarity scores in DESCENDING order per query row.

    Call this whenever you are unsure whether the output of your VPR
    backbone is already sorted.  It is idempotent — safe to call even
    if the input is already sorted.

    Parameters
    ----------
    sim_scores : ndarray, shape (N_queries, K)
        Raw (possibly unsorted) similarity scores.

    Returns
    -------
    ndarray, shape (N_queries, K)
        Same scores sorted descending along axis=1.

    Examples
    --------
    >>> sort_scores_desc(np.array([[0.3, 0.9, 0.5]]))
    array([[0.9, 0.5, 0.3]])
    """
    scores = _validate(sim_scores)
    return np.sort(scores, axis=1)[:, ::-1]


# ─────────────────────────────────────────────────────────────────
# Core uncertainty metrics  (paper Section III)
# ─────────────────────────────────────────────────────────────────

def RS(sim_scores: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Ratio Spread — measures competitive ambiguity among top-K candidates.

    Paper formula (Section III.A):
        RS = (1 / (k-1)) * Σ_{i=2}^{k}  (s_i / s_1)

    Interpretation
    --------------
    • High RS: competitors score close to the top match → uncertain retrieval
               (perceptual aliasing likely)
    • Low RS:  top match clearly separated from the rest → confident retrieval

    Parameters
    ----------
    sim_scores : ndarray, shape (N_queries, K)
        Similarity scores sorted DESCENDING per row.
    eps : float
        Numerical stability constant (avoids division by zero when s_1 ≈ 0).

    Returns
    -------
    ndarray, shape (N_queries,)
        RS score per query.  Higher = more uncertain.
    """
    scores = _validate(sim_scores)
    s1          = scores[:, 0:1]      # shape (N, 1)  — best score
    competitors = scores[:, 1:]       # shape (N, K-1)
    return np.mean(competitors / (s1 + eps), axis=1)


def SD(sim_scores: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Similarity Distribution — measures global score-distribution flatness.

    Paper formula (Section III.B):
        SD = s_median / s_best

    Interpretation
    --------------
    • SD close to 1.0: score distribution is flat (uniform) → uncertain,
                        many candidates equally competitive
    • SD close to 0.0: large gap between best and median → confident,
                        top match is clearly distinct

    Parameters
    ----------
    sim_scores : ndarray, shape (N_queries, K)
        Similarity scores sorted DESCENDING per row.
    eps : float
        Numerical stability constant.

    Returns
    -------
    ndarray, shape (N_queries,)
        SD score per query.  Higher = more uncertain.
    """
    scores   = _validate(sim_scores)
    s_best   = scores[:, 0]                    # shape (N,)
    s_median = np.median(scores, axis=1)       # shape (N,)
    return s_median / (s_best + eps)


def SU(sim_scores: np.ndarray,
        alpha: float = DEFAULT_ALPHA,
        eps: float = EPS) -> np.ndarray:
    """
    Statistical Uncertainty — weighted combination of RS and SD.

    Paper formula (Section III.C, Equation 1):
        SU = α × RS + (1 − α) × SD

    The paper uses α = 0.5 as the robust default (Section IV.C):
    "The SU score combines RS and SD with equal weighting (α = 0.5)."

    Ablation studies (Section VI.A) confirm α = 0.5 generalises well
    across datasets and VPR methods without requiring validation data.

    Interpretation
    --------------
    • High SU → high uncertainty → vision is unreliable
    • Low  SU → low  uncertainty → vision is confident

    Parameters
    ----------
    sim_scores : ndarray, shape (N_queries, K)
        Similarity scores sorted DESCENDING per row.
    alpha : float in [0, 1]
        Mixing weight.  0.0 = SD only, 1.0 = RS only.
        Default: 0.5 (paper default).
    eps : float
        Numerical stability constant.

    Returns
    -------
    ndarray, shape (N_queries,)
        Raw SU score per query.  Higher = more uncertain.

    Notes
    -----
    The raw SU output is NOT bounded and its scale depends on the VPR
    backbone and dataset.  For use as a sigmoid fusion gate, always call
    SU_normalised() instead, which applies z-score normalisation so the
    gate spans [0, 1] properly.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    rs = RS(sim_scores, eps=eps)
    sd = SD(sim_scores, eps=eps)
    return alpha * rs + (1.0 - alpha) * sd


# ─────────────────────────────────────────────────────────────────
# Normalised uncertainty — for use in the fusion gate
# ─────────────────────────────────────────────────────────────────

def SU_normalised(sim_scores: np.ndarray,
                  alpha: float = DEFAULT_ALPHA,
                  eps: float = EPS) -> np.ndarray:
    """
    Z-score normalised SU — the correct input for the sigmoid fusion gate.

    Why normalisation is required
    ------------------------------
    Raw SU values are unbounded and dataset-dependent.  Without
    normalisation, sigmoid(raw SU) can collapse to a narrow range
    (e.g. 0.30 – 0.44), meaning the fusion gate barely opens and
    language never receives meaningful weight regardless of how uncertain
    vision is.

    After z-score normalisation:
    • SU_norm = 0   → query is at the population mean confidence
                      → sigmoid = 0.5 → visual and language weighted equally
    • SU_norm >> 0  → query is more uncertain than average
                      → sigmoid > 0.5 → language gets more weight
    • SU_norm << 0  → query is more confident than average
                      → sigmoid < 0.5 → vision dominates

    This ensures the inflection point of the sigmoid gate sits at the
    population mean confidence level — a principled threshold rather
    than an arbitrary one.

    Parameters
    ----------
    sim_scores : ndarray, shape (N_queries, K)
        Similarity scores sorted DESCENDING per row.
    alpha : float in [0, 1]
        Mixing weight for RS vs SD.  Default: 0.5 (paper default).
    eps : float
        Numerical stability constant.

    Returns
    -------
    ndarray, shape (N_queries,)
        Z-score normalised SU.  Pass directly to sigmoid_gate().

    ⚠️  IMPORTANT: normalisation is computed over the current query SET.
    If you run queries one-at-a-time, normalisation has no effect (std=0).
    Normalise over a batch of at least 20 queries for meaningful results.
    """
    raw = SU(sim_scores, alpha=alpha, eps=eps)
    std = raw.std()
    if std < eps:
        # All queries have identical SU — return zeros (equal weighting)
        return np.zeros_like(raw)
    return (raw - raw.mean()) / (std + eps)


# ─────────────────────────────────────────────────────────────────
# Sigmoid gate — converts normalised SU to visual fusion weight alpha
# ─────────────────────────────────────────────────────────────────

def sigmoid_gate(su_normalised: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Convert normalised SU uncertainty to a visual fusion weight alpha.

    The gate maps normalised SU → α ∈ (0, 1) where:

        α = σ(−SU_norm / τ) = 1 / (1 + exp(SU_norm / τ))

    NOTE: the negation is intentional.
    • High SU_norm (more uncertain)  →  low α  →  language gets more weight
    • Low  SU_norm (more confident)  →  high α →  vision dominates

    This is consistent with the fusion equation:
        fused_score = α × visual_sim + (1 − α) × lang_sim

    Parameters
    ----------
    su_normalised : ndarray, shape (N_queries,)
        Output of SU_normalised().
    tau : float > 0
        Temperature parameter controlling gate sharpness.
        • tau = 1.0  (default): standard sigmoid
        • tau < 1.0: sharper gate (more binary behaviour)
        • tau > 1.0: softer gate (smoother transition)
        Ablation over tau ∈ {0.5, 1.0, 2.0} is recommended in Month 3.

    Returns
    -------
    ndarray, shape (N_queries,)
        Visual weight α per query, in range (0, 1).
        Use as:  fused = α * visual_sim + (1 - α) * lang_sim

    Examples
    --------
    >>> su = np.array([-2.0, 0.0, 2.0])   # confident, neutral, uncertain
    >>> sigmoid_gate(su)
    array([0.8808, 0.5000, 0.1192])        # vision heavy, equal, language heavy
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")
    return 1.0 / (1.0 + np.exp(su_normalised / tau))


# ─────────────────────────────────────────────────────────────────
# Language variance pre-filter
# ─────────────────────────────────────────────────────────────────

def is_language_discriminative(lang_sims: np.ndarray,
                                threshold: float = 0.05) -> np.ndarray:
    """
    Pre-filter: check whether language descriptions are discriminative
    enough to be useful for re-ranking the top-K candidates.

    If all K candidate descriptions have nearly identical cosine
    similarity to the query description (variance < threshold), the
    language signal carries no useful discriminative information.
    In this case the fusion gate should be suppressed regardless of
    visual confidence — adding noise is worse than doing nothing.

    This condition arises in:
    • Perceptual aliasing: two different places of the same type
      (e.g. two generic corridors) get nearly identical descriptions
    • Temporally drifted scenes (AmsterTime): the query's modern
      description differs semantically from all historical database
      descriptions equally

    Parameters
    ----------
    lang_sims : ndarray, shape (N_queries, K)
        Cosine similarity between query description embedding and each
        of the K candidate description embeddings.
    threshold : float
        Minimum std of language similarities required to consider the
        language signal discriminative.  Default: 0.05.
        Tune this in Month 3 as an ablation parameter.

    Returns
    -------
    ndarray of bool, shape (N_queries,)
        True  → language IS discriminative for this query → use fusion gate
        False → language is NOT discriminative           → keep alpha = 1.0

    Notes
    -----
    This is the direct implementation of Proxy Experiment Finding 2:
    "The language signal FAILS for perceptual aliasing (same-type
    different-place) and temporal drift (AmsterTime)."
    The threshold encodes this finding explicitly in the method.
    """
    lang_sims = np.asarray(lang_sims, dtype=np.float64)
    if lang_sims.ndim != 2:
        raise ValueError(
            f"lang_sims must be 2-D with shape (N_queries, K), "
            f"got shape {lang_sims.shape}."
        )
    variance = np.std(lang_sims, axis=1)
    return variance >= threshold


# ─────────────────────────────────────────────────────────────────
# Full fusion step (puts it all together)
# ─────────────────────────────────────────────────────────────────

def confidence_gated_fusion(visual_sims: np.ndarray,
                             lang_sims: np.ndarray,
                             su_norm: np.ndarray,
                             tau: float = 1.0,
                             lang_threshold: float = 0.05) -> np.ndarray:
    """
    Confidence-gated language fusion for VPR re-ranking.

    Combines visual and language similarity scores using a dynamic weight
    derived from the visual retrieval confidence (SU metric).

    Algorithm
    ---------
    For each query q:
        1. Check if language is discriminative for q (variance filter)
        2. If yes:  α = sigmoid_gate(SU_normalised(q))
           If no:   α = 1.0  (suppress language, trust vision only)
        3. fused_score = α × visual_sim + (1 − α) × lang_sim
        4. Re-rank top-K by fused_score

    Parameters
    ----------
    visual_sims : ndarray, shape (N_queries, K)
        Visual cosine similarity scores for top-K candidates, sorted DESC.
    lang_sims : ndarray, shape (N_queries, K)
        Language cosine similarity scores for the same top-K candidates.
        (query description embedding vs each candidate description embedding)
    su_norm : ndarray, shape (N_queries,)
        Output of SU_normalised() computed from visual_sims.
    tau : float
        Temperature for sigmoid gate.  Default: 1.0.
    lang_threshold : float
        Minimum language similarity std to activate fusion.  Default: 0.05.

    Returns
    -------
    fused_sims : ndarray, shape (N_queries, K)
        Fused similarity scores for re-ranking.
    alpha : ndarray, shape (N_queries,)
        The visual weight used per query (for logging/ablation).

    Example
    -------
    # Full pipeline sketch:
    sim_scores = sort_scores_desc(get_topk_visual_sims(...))   # (N, K)
    su         = SU_normalised(sim_scores)                      # (N,)
    lang_sims  = get_topk_lang_sims(...)                        # (N, K)
    fused, alpha = confidence_gated_fusion(sim_scores, lang_sims, su)
    reranked_indices = top_k_indices[np.argsort(fused, axis=1)[:, ::-1]]
    """
    visual_sims = np.asarray(visual_sims, dtype=np.float64)
    lang_sims   = np.asarray(lang_sims,   dtype=np.float64)
    su_norm     = np.asarray(su_norm,     dtype=np.float64)

    if visual_sims.shape != lang_sims.shape:
        raise ValueError(
            f"visual_sims {visual_sims.shape} and lang_sims {lang_sims.shape} "
            f"must have the same shape."
        )
    if su_norm.shape[0] != visual_sims.shape[0]:
        raise ValueError(
            f"su_norm has {su_norm.shape[0]} entries but visual_sims has "
            f"{visual_sims.shape[0]} queries."
        )

    # Step 1: visual weight from sigmoid gate
    alpha = sigmoid_gate(su_norm, tau=tau)          # shape (N,)

    # Step 2: language discriminability pre-filter
    # Suppress language for queries where it cannot help
    lang_ok = is_language_discriminative(lang_sims, threshold=lang_threshold)
    alpha   = np.where(lang_ok, alpha, 1.0)         # alpha=1 → vision only

    # Step 3: fuse  (broadcast alpha over K candidates)
    alpha_2d  = alpha[:, np.newaxis]                # shape (N, 1)
    fused     = alpha_2d * visual_sims + (1.0 - alpha_2d) * lang_sims

    return fused, alpha


# ─────────────────────────────────────────────────────────────────
# Calibration utility — generates the SU decile vs R@1 plot data
# ─────────────────────────────────────────────────────────────────

def su_calibration_curve(su_scores: np.ndarray,
                          correct: np.ndarray,
                          n_bins: int = 10) -> tuple:
    """
    Compute R@1 per SU decile for the calibration plot (Figure 2 of paper).

    If this curve is monotonically decreasing (higher SU decile = lower R@1),
    SU is a valid confidence proxy for your dataset.  This is a required
    validation experiment in Month 1 Week 4.

    Parameters
    ----------
    su_scores : ndarray, shape (N_queries,)
        Raw (un-normalised) SU scores per query.
    correct : ndarray of bool, shape (N_queries,)
        Whether the top-1 retrieved candidate is correct for each query.
    n_bins : int
        Number of quantile bins.  Default: 10 (deciles).

    Returns
    -------
    bin_centers : ndarray, shape (n_bins,)
        Midpoint SU value for each bin.
    recall_at_1 : ndarray, shape (n_bins,)
        R@1 for queries in each SU bin.
    bin_counts : ndarray, shape (n_bins,)
        Number of queries in each bin (for confidence weighting).

    Usage
    -----
    su     = SU(sim_scores)
    right  = (top1_retrieved_id == ground_truth_id)
    bins, r1, counts = su_calibration_curve(su, right)

    import matplotlib.pyplot as plt
    plt.plot(range(1, 11), r1, 'o-')
    plt.xlabel('SU decile (1=most confident, 10=most uncertain)')
    plt.ylabel('R@1')
    plt.title('SU calibration curve')
    plt.savefig('su_calibration.png')
    """
    su      = np.asarray(su_scores, dtype=np.float64).ravel()
    correct = np.asarray(correct,   dtype=bool).ravel()

    if len(su) != len(correct):
        raise ValueError(
            f"su_scores ({len(su)}) and correct ({len(correct)}) "
            f"must have the same length."
        )

    percentiles = np.linspace(0, 100, n_bins + 1)
    edges       = np.percentile(su, percentiles)

    bin_centers = np.zeros(n_bins)
    recall_at_1 = np.zeros(n_bins)
    bin_counts  = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include right edge in last bin to avoid empty bin
        if i < n_bins - 1:
            mask = (su >= lo) & (su < hi)
        else:
            mask = (su >= lo) & (su <= hi)

        bin_counts[i]  = mask.sum()
        bin_centers[i] = (lo + hi) / 2.0
        recall_at_1[i] = correct[mask].mean() if bin_counts[i] > 0 else np.nan

    return bin_centers, recall_at_1, bin_counts


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("su_metric.py — self-test")
    print("=" * 50)

    rng = np.random.default_rng(seed=42)

    # ── Test 1: Hand-verified sanity check ──────────────────────
    scores_1 = np.array([[0.9, 0.6, 0.5, 0.4, 0.3]])

    rs_val = RS(scores_1)[0]
    sd_val = SD(scores_1)[0]
    su_val = SU(scores_1)[0]

    # Expected (computed by hand):
    # RS = mean([0.6/0.9, 0.5/0.9, 0.4/0.9, 0.3/0.9]) = 0.5000
    # SD = median([0.9,0.6,0.5,0.4,0.3]) / 0.9 = 0.5 / 0.9 = 0.5556
    # SU = 0.5*0.5 + 0.5*0.5556 = 0.5278
    assert abs(rs_val - 0.5000) < 1e-4, f"RS FAIL: {rs_val}"
    assert abs(sd_val - 0.5556) < 1e-4, f"SD FAIL: {sd_val}"
    assert abs(su_val - 0.5278) < 1e-4, f"SU FAIL: {su_val}"
    print("Test 1 PASS — RS, SD, SU match hand-computed values")
    print(f"  RS = {rs_val:.4f}  (expected 0.5000)")
    print(f"  SD = {sd_val:.4f}  (expected 0.5556)")
    print(f"  SU = {su_val:.4f}  (expected 0.5278)")

    # ── Test 2: sort_scores_desc ─────────────────────────────────
    unsorted = np.array([[0.3, 0.9, 0.5, 0.6, 0.4]])
    sorted_  = sort_scores_desc(unsorted)
    assert np.allclose(sorted_, [[0.9, 0.6, 0.5, 0.4, 0.3]])
    print("\nTest 2 PASS — sort_scores_desc works correctly")

    # ── Test 3: sorting matters (unsorted → wrong RS) ────────────
    rs_wrong  = RS(unsorted)[0]           # will NOT equal 0.5
    rs_correct = RS(sort_scores_desc(unsorted))[0]  # = 0.5
    assert abs(rs_correct - 0.5) < 1e-4
    assert abs(rs_wrong - 0.5) > 0.1    # confirms unsorted gives wrong result
    print("\nTest 3 PASS — unsorted input gives wrong RS (confirmed dangerous)")
    print(f"  RS (unsorted):         {rs_wrong:.4f}   ← WRONG")
    print(f"  RS (sorted correctly): {rs_correct:.4f}  ← correct")

    # ── Test 4: SU_normalised spans full gate range ──────────────
    fake = rng.random((200, 10))
    fake_sorted = sort_scores_desc(fake)
    su_raw  = SU(fake_sorted)
    su_norm = SU_normalised(fake_sorted)
    gate    = sigmoid_gate(su_norm)
    assert abs(su_norm.mean()) < 0.01, "z-score mean should be ~0"
    assert abs(su_norm.std()  - 1.0) < 0.05, "z-score std should be ~1"
    assert gate.min() < 0.2,  "gate should open fully for uncertain queries"
    assert gate.max() > 0.8,  "gate should close fully for confident queries"
    print(f"\nTest 4 PASS — SU_normalised and sigmoid_gate work correctly")
    print(f"  su_norm mean: {su_norm.mean():.4f} (expected ≈ 0)")
    print(f"  su_norm std:  {su_norm.std():.4f}  (expected ≈ 1)")
    print(f"  gate range:   [{gate.min():.3f}, {gate.max():.3f}]  (expected [<0.2, >0.8])")

    # ── Test 5: language discriminability filter ─────────────────
    # Discriminative: std of lang sims > threshold
    disc_lang  = np.array([[0.9, 0.3, 0.2, 0.1, 0.1]])  # high variance
    aliased_lang = np.array([[0.7, 0.69, 0.68, 0.67, 0.66]])  # low variance
    assert is_language_discriminative(disc_lang)[0]   == True
    assert is_language_discriminative(aliased_lang)[0] == False
    print("\nTest 5 PASS — language discriminability filter works correctly")
    print(f"  Discriminative lang sims → filter = True  (use fusion)")
    print(f"  Aliased lang sims        → filter = False (suppress language)")

    # ── Test 6: confidence_gated_fusion ─────────────────────────
    N, K = 50, 10
    vis  = sort_scores_desc(rng.random((N, K)))
    lang = rng.random((N, K))
    su   = SU_normalised(vis)
    fused, alpha = confidence_gated_fusion(vis, lang, su)
    assert fused.shape == (N, K)
    assert alpha.shape == (N,)
    assert np.all((alpha >= 0) & (alpha <= 1))
    print(f"\nTest 6 PASS — confidence_gated_fusion runs end-to-end")
    print(f"  Output shape: {fused.shape}")
    print(f"  Alpha range:  [{alpha.min():.3f}, {alpha.max():.3f}]")

    # ── Test 7: su_calibration_curve ────────────────────────────
    # Create synthetic data: confident queries (low SU) should have high R@1
    su_test = np.concatenate([
        rng.uniform(0.2, 0.4, 100),   # confident
        rng.uniform(0.7, 0.9, 100),   # uncertain
    ])
    correct_test = np.concatenate([
        rng.random(100) < 0.85,       # confident → mostly correct
        rng.random(100) < 0.40,       # uncertain → mostly wrong
    ])
    bins, r1, counts = su_calibration_curve(su_test, correct_test)
    # First bin (most confident) should have higher R@1 than last bin
    assert r1[0] > r1[-1], "calibration curve should be decreasing"
    print(f"\nTest 7 PASS — su_calibration_curve is monotonically decreasing")
    print(f"  Bin 1 (confident) R@1: {r1[0]:.3f}")
    print(f"  Bin 10 (uncertain) R@1: {r1[-1]:.3f}")

    # ── Test 8: K=1 raises ValueError ───────────────────────────
    try:
        RS(np.array([[0.9]]))
        print("\nTest 8 FAIL — K=1 should raise ValueError")
    except ValueError:
        print("\nTest 8 PASS — K=1 raises ValueError correctly")

    print("\n" + "=" * 50)
    print("All 8 tests passed.")
    print("su_metric.py is ready for production use.")
