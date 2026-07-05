"""
uncertainty_v2.py

V2 uncertainty and precision-weighted fusion module.

This file implements the redesigned method:

1. Visual uncertainty: SU
   - Same RS + SD structure as the original paper.

2. Language uncertainty: LU
   - New language-side uncertainty.
   - Same RS + SD structure, but computed from the top-K language similarity distribution.

3. Precision-weighted fusion:
   - Replaces the old sigmoid + hard language threshold mechanism.
   - Alpha is derived from visual and language uncertainty variance proxies.

Notation:
    alpha = visual weight

So:
    alpha close to 1 -> trust visual similarity more
    alpha close to 0 -> trust language similarity more
"""

import numpy as np


def sort_scores_desc(scores: np.ndarray) -> np.ndarray:
    """
    Sort each query row in descending order.

    Args:
        scores: Array of shape (n_queries, k) or (n_queries, n_database)

    Returns:
        Sorted scores with the same shape.
    """
    scores = np.asarray(scores)
    return np.sort(scores, axis=1)[:, ::-1]


def ratio_spread(sorted_scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Ratio Spread (RS).

    Measures how close the remaining candidates are to the top candidate.

    Higher RS means the top candidate is not clearly separated.
    """
    sorted_scores = np.asarray(sorted_scores)

    if sorted_scores.ndim != 2:
        raise ValueError("sorted_scores must be a 2D array.")

    top = sorted_scores[:, [0]]
    rest = sorted_scores[:, 1:]

    rs = np.mean(rest / (top + eps), axis=1)
    return rs


def similarity_distribution(sorted_scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Similarity Distribution (SD).

    Measures how flat the score distribution is using the median/top ratio.

    Higher SD means the score distribution is flatter.
    """
    sorted_scores = np.asarray(sorted_scores)

    if sorted_scores.ndim != 2:
        raise ValueError("sorted_scores must be a 2D array.")

    top = sorted_scores[:, 0]
    median = np.median(sorted_scores, axis=1)

    sd = median / (top + eps)
    return sd


def statistical_uncertainty(
    sorted_scores: np.ndarray,
    lam: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Statistical Uncertainty (SU)-style score.

    This function is used for both:
        - visual uncertainty SU
        - language uncertainty LU

    Args:
        sorted_scores: Descending sorted scores, shape (n_queries, k)
        lam: Weight between RS and SD
        eps: Numerical stability value

    Returns:
        uncertainty score of shape (n_queries,)
    """
    rs = ratio_spread(sorted_scores, eps=eps)
    sd = similarity_distribution(sorted_scores, eps=eps)

    return lam * rs + (1.0 - lam) * sd


def visual_uncertainty(
    visual_topk_scores: np.ndarray,
    lam: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Compute visual uncertainty SU from top-K visual scores.

    The input may already be sorted by visual similarity, but this function
    sorts again to avoid accidental misuse.
    """
    visual_sorted = sort_scores_desc(visual_topk_scores)
    return statistical_uncertainty(visual_sorted, lam=lam, eps=eps)


def language_uncertainty(
    language_topk_scores: np.ndarray,
    lam: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Compute language uncertainty LU from top-K language scores.

    Important:
        The language scores are sorted by language value before LU is computed.
        This makes LU a property of the language score distribution, not the
        existing visual rank order.

    Higher LU:
        language scores are flat / non-discriminative

    Lower LU:
        one language candidate stands out more clearly
    """
    language_sorted = sort_scores_desc(language_topk_scores)
    return statistical_uncertainty(language_sorted, lam=lam, eps=eps)


def normalise_uncertainty(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Z-score normalisation for uncertainty scores.

    This is unsupervised. It uses only the uncertainty values across queries.
    """
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + eps)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softplus.

    softplus(x) = log(1 + exp(x))

    Used to convert normalised uncertainty into a positive variance proxy.
    """
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def uncertainty_to_variance(
    uncertainty_norm: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert normalised uncertainty into a positive variance proxy.

    Higher normalised uncertainty -> larger variance.
    """
    return softplus(uncertainty_norm) + eps


def precision_weighted_alpha(
    su_norm: np.ndarray,
    lu_norm: np.ndarray,
    rho: float = 6.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute precision-weighted visual alpha.

    alpha is the visual weight:

        alpha = sigma_l^2 / (sigma_v^2 + sigma_l^2)

    where:
        sigma_v^2 = visual variance proxy from SU_norm
        sigma_l^2 = language variance proxy from LU_norm

    Interpretation:
        high language uncertainty -> high sigma_l^2 -> alpha closer to 1
        high visual uncertainty + low language uncertainty -> alpha closer to 0
    """
    su_norm = np.asarray(su_norm)
    lu_norm = np.asarray(lu_norm)

    if su_norm.shape != lu_norm.shape:
        raise ValueError("su_norm and lu_norm must have the same shape.")

    sigma_v2 = uncertainty_to_variance(su_norm, eps=eps)
    sigma_l2 = uncertainty_to_variance(lu_norm, eps=eps)

    alpha = (rho * sigma_l2) / (sigma_v2 + rho * sigma_l2 + eps)

    return alpha


def precision_weighted_fusion(
    visual_sims: np.ndarray,
    language_sims: np.ndarray,
    su_norm: np.ndarray,
    lu_norm: np.ndarray,
    rho: float = 6.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply precision-weighted visual-language fusion.

    Args:
        visual_sims: Top-K visual similarities, shape (n_queries, k)
        language_sims: Top-K language similarities for same candidates, shape (n_queries, k)
        su_norm: Normalised visual uncertainty, shape (n_queries,)
        lu_norm: Normalised language uncertainty, shape (n_queries,)

    Returns:
        fused_sims: Fused top-K scores, shape (n_queries, k)
        alpha: Visual weight per query, shape (n_queries,)
    """
    visual_sims = np.asarray(visual_sims)
    language_sims = np.asarray(language_sims)

    if visual_sims.shape != language_sims.shape:
        raise ValueError("visual_sims and language_sims must have the same shape.")

    if visual_sims.shape[0] != su_norm.shape[0]:
        raise ValueError("Number of queries in visual_sims and su_norm does not match.")

    if visual_sims.shape[0] != lu_norm.shape[0]:
        raise ValueError("Number of queries in visual_sims and lu_norm does not match.")

    alpha = precision_weighted_alpha(su_norm, lu_norm, rho=rho, eps=eps)

    fused_sims = alpha[:, None] * visual_sims + (1.0 - alpha[:, None]) * language_sims

    return fused_sims, alpha


def compute_su_lu_alpha(
    visual_topk_scores: np.ndarray,
    language_topk_scores: np.ndarray,
    lam: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function.

    Computes:
        SU
        LU
        SU_norm
        LU_norm
        alpha

    from top-K visual and language scores.
    """
    su = visual_uncertainty(visual_topk_scores, lam=lam, eps=eps)
    lu = language_uncertainty(language_topk_scores, lam=lam, eps=eps)

    su_norm = normalise_uncertainty(su, eps=eps)
    lu_norm = normalise_uncertainty(lu, eps=eps)

    alpha = precision_weighted_alpha(su_norm, lu_norm, rho=6.0)

    return su, lu, su_norm, lu_norm, alpha
