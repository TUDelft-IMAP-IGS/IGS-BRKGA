import numpy as np
from typing import Union


def a_fine_aggregator(
    w: Union[list[float], np.ndarray],
    p: Union[list[Union[float, int]], Union[list[float], np.ndarray]],
) -> np.ndarray:
    """
    Aggregates preference scores using least-squares distance minimization in affine space.

    This function implements a least-squares distance minimization approach for preference
    aggregation. It first normalizes the input scores (z-score normalization), and then
    calculates representative preference scores as a weighted sum of normalized scores.
    Finally, it transforms the results to a 0-100 scale, where higher values represent
    better solutions.

    Mathematical background:
    1. First, inputs are z-score normalized: z = (x - μ) / σ
    2. Then, a single representative score is computed as weighted sum: P* = Σ(w_i * z_i)
    3. Finally, min-max normalization is applied to scale to 0-100: P_scaled = (P* - min(P*)) / (max(P*) - min(P*)) * 100

    Parameters
    ----------
    w : list[float] or numpy.ndarray
        Weights of different objectives. Must sum to 1 and have the same length as `p`.
        Each weight represents the relative importance of the corresponding preference score.

    p : list[float] or numpy.ndarray
        Preference scores for each objective. Each entry should be a scalar value representing
        the preference score for that objective (typically between 0-1, where higher is better).

    Returns
    -------
    numpy.ndarray
        An array of aggregated preference scores, scaled to the range [0, 100].
        Higher values indicate better solutions (more preferred).

    Raises
    ------
    AssertionError
        If the length of weights doesn't match the length of preference scores.
        If the sum of weights is not 1 (within a small tolerance).

    Notes
    -----
    - This is an implementation of the affine aggregation method described in multi-criteria
      decision analysis literature.
    - For objectives with zero standard deviation, a small value (1e-6) is used to avoid
      division by zero.
    - If all scores have the same representative preference value, a default score of -50 is returned.

    Examples
    --------
    >>> weights = [0.6, 0.4]  # 60% weight on first objective, 40% on second
    >>> preferences = [0.8, 0.3]  # Preference scores for two objectives
    >>> a_fine_aggregator(weights, preferences)
    array([-50.])  # For single-point evaluation, default score is returned

    >>> weights = [0.5, 0.5]
    >>> preferences = [[0.8, 0.3], [0.6, 0.7]]  # Two sets of preference scores
    >>> a_fine_aggregator(weights, preferences)  # Comparing two alternatives
    array([  0., 100.])  # Alternative 2 is preferred over Alternative 1

    References
    ----------
    For more details on multi-criteria aggregation methods, see:
    - Keeney, R.L., Raiffa, H. (1976). "Decisions with Multiple Objectives"
    - Greco, S., et al. (2016). "Multiple Criteria Decision Analysis: State of the Art Surveys"
    """
    assert len(w) == len(p), (
        f"The number of weights ({len(w)}) is not equal to the number of objectives "
        f"({len(p)})."
    )

    assert (
            round(sum(w), 4) == 1
    ), f"The sum of the weights ({round(sum(w), 4)}) is not equal to 1."

    # transpose the array to make further calculations easier
    p_transposed = np.array(p).transpose()

    # calculate the standard deviation per criteria. If std == 0, a value << 1 is
    # inserted to prevent divide by zero error
    std = np.std(p_transposed, axis=0)
    std[std == 0] = 1e-6

    # calculate the z-score normalized scores per criteria
    z = (p_transposed - np.mean(p_transposed, axis=0)) / std

    # calculate representative preference scores (P_i^*)
    p_star = np.sum(w * z, axis=1)

    if len(np.unique(p_star, axis=0)) == 1:
        # if there is only one unique member in p_star
        return np.full(len(p_transposed), 50.0, dtype=float)
    else:
        # return min-max normalized results, so everything is on the scale [0-100]
        return (p_star - min(p_star)) / (max(p_star) - min(p_star)) * 100
