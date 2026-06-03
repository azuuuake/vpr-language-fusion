import numpy as np

EPS = 1e-9


def sort_scores_desc(sim_scores):
    """Sort similarity scores in descending order per query."""
    sim_scores = np.asarray(sim_scores, dtype=float)
    return np.sort(sim_scores, axis=1)[:, ::-1]


def RS(sim_scores):
    """
    Ratio Spread uncertainty metric.

    RS = mean(s_i / s_1), for i = 2..k.
    Higher RS means higher uncertainty.
    """
    sim_scores = np.asarray(sim_scores, dtype=float)
    top = sim_scores[:, [0]]
    competitors = sim_scores[:, 1:]
    return np.mean(competitors / (top + EPS), axis=1)


def SD(sim_scores):
    """
    Similarity Distribution uncertainty metric.

    SD = median(top-k scores) / best score.
    Higher SD means higher uncertainty.
    """
    sim_scores = np.asarray(sim_scores, dtype=float)
    best = sim_scores[:, 0]
    median = np.median(sim_scores, axis=1)
    return median / (best + EPS)


def SU(sim_scores, alpha=0.5):
    """
    Statistical Uncertainty.

    SU = alpha * RS + (1 - alpha) * SD.
    Higher SU means higher visual retrieval uncertainty.
    """
    return alpha * RS(sim_scores) + (1 - alpha) * SD(sim_scores)
