import numpy as np
from scipy import stats

def rout_outliers(y, Q=0.10):
    """
    Identify outliers using the ROUT method (Motulsky & Brown, 2006),
    as implemented in GraphPad Prism.

    Parameters
    ----------
    y : array-like
        1D array of values (a single group, no x predictor).
    Q : float
        Maximum False Discovery Rate (0 < Q <= 1).
        GraphPad's "remove likely outliers" = 0.10 (most aggressive).
        Other common choices: 0.01 (strict), 0.05 (moderate).

    Returns
    -------
    outlier_mask : np.ndarray of bool
        True where a value is flagged as an outlier.
    clean_values : np.ndarray
        Values with outliers removed.
    outlier_values : np.ndarray
        Values flagged as outliers.

    Reference
    ---------
    Motulsky HJ & Brown RE (2006). Detecting outliers when fitting data
    with nonlinear regression. BMC Bioinformatics, 7:123.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # --- Step 1: Robust estimate of center and spread ---
    # GraphPad uses the median as the robust center for a 1-sample group,
    # and MADN (median absolute deviation, normalized) as the robust spread.
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    # Normalize MAD to be a consistent estimator of sigma for normal data
    madn = mad / 0.6745

    if madn == 0:
        # All values identical (or nearly so); no outliers possible
        return np.zeros(n, dtype=bool), y.copy(), np.array([])

    # --- Step 2: Compute standardized residuals ---
    residuals = np.abs(y - median) / madn

    # --- Step 3: Convert residuals to two-tailed p-values ---
    # Each residual is treated as a z-score under the normal distribution
    p_values = 2 * (1 - stats.norm.cdf(residuals))

    # --- Step 4: Apply Benjamini-Hochberg FDR correction at level Q ---
    outlier_mask = _benjamini_hochberg(p_values, Q)

    clean_values = y[~outlier_mask]
    outlier_values = y[outlier_mask]

    return outlier_mask, clean_values, outlier_values


def _benjamini_hochberg(p_values, Q):
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate.
    Returns a boolean mask of rejected hypotheses (i.e., outliers).
    """
    n = len(p_values)
    # Rank p-values (smallest = rank 1)
    order = np.argsort(p_values)
    ranked_p = p_values[order]

    # BH threshold: p(i) <= (i/n) * Q
    bh_threshold = (np.arange(1, n + 1) / n) * Q

    # Find the largest rank where p <= threshold (all ranks up to it are rejected)
    below = ranked_p <= bh_threshold
    if not below.any():
        return np.zeros(n, dtype=bool)

    max_rank = np.max(np.where(below))
    rejected = np.zeros(n, dtype=bool)
    rejected[order[:max_rank + 1]] = True
    return rejected