from __future__ import annotations

from optimization._aggregator import a_fine_aggregator

from typing import Callable

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Preference-function constructors
# ═══════════════════════════════════════════════════════════════════════════════

def make_pref_cv(
    y_step: float = 1e-6,
    tail_k: float = 2.0,
    p: float = 1.5,
) -> Callable[[float], float]:
    """
    Constraint-violation (CV) preference function with a three-zone step-decay shape.

    Zone 1 — Exact zero (feasible):
        ``f(0) = 100``

    Zone 2 — The step (any positive value up to *y_step*):
        ``f(cv) = 10``  — a sharp drop the moment a solution becomes infeasible.

    Zone 3 — Gentle tail (cv > *y_step*):
        ``f(cv) = 10 / (1 + tail_k · (cv - y_step) ^ p)``

        The tail is anchored at 10 at the step boundary and decays toward 0,
        but never reaches it.  Smaller *tail_k* or *p* gives a gentler descent.

    Parameters
    ----------
    y_step:
        Step threshold.  Any strictly positive cv ≤ y_step maps to exactly 10.
        Defaults to ``1e-6``.
    tail_k:
        Scale factor controlling the steepness of the tail decay.
        Smaller values produce a gentler tail.  Default ``2.0``.
    p:
        Power exponent for the tail decay shape.  Default ``1.5``.
    """
    def fn(cv: float) -> float:
        if cv == 0.0:
            return 100.0
        if cv <= y_step:
            return 10.0
        shifted = cv - y_step
        return float(10.0 / (1.0 + tail_k * (shifted ** p)))

    return fn


def make_pref_linear(
    lo: float,
    hi: float,
    *,
    reverse: bool = False,
) -> Callable[[float], float]:
    """
    Linear preference function mapping ``[lo, hi]`` → ``[100, 0]``.

    Lower raw value → higher preference (minimisation direction).
    Values outside the range are clamped to ``[0, 100]``.

    Parameters
    ----------
    lo, hi:
        Raw objective bounds.
    reverse:
        When ``True`` the mapping is flipped: higher raw value → higher
        preference (maximisation direction).  Equivalent to swapping lo and hi.
    """
    if reverse:
        lo, hi = hi, lo

    def fn(x: float) -> float:
        if hi == lo:
            return 50.0
        return float(np.clip((hi - x) / (hi - lo) * 100.0, 0.0, 100.0))

    return fn


def make_pref_nonlinear(
    lo: float,
    hi: float,
    shape: str,
    *,
    reverse: bool = False,
) -> Callable[[float], float]:
    """
    Non-linear preference function mapping ``[lo, hi]`` → ``[100, 0]``.

    All variants use the normalised ratio ``t = (hi - x) / (hi - lo) ∈ [0, 1]``,
    where ``t = 1`` corresponds to the best observed value and ``t = 0`` to the
    worst.

    Parameters
    ----------
    lo, hi:
        Raw objective bounds.
    shape:
        ``"convex"``  — ``f(t) = t²  * 100`` : extra credit near optimum.
        ``"concave"`` — ``f(t) = √t  * 100`` : diminishing returns.
        ``"sigmoid"`` — logistic centred at ``t = 0.5``, steepness ``k = 8``.
    reverse:
        When ``True`` the lo/hi mapping is flipped before constructing the
        normalisation (maximisation direction).
    """
    _SUPPORTED = ("convex", "concave", "sigmoid")
    if shape not in _SUPPORTED:
        raise ValueError(f"Unknown preference shape {shape!r}. Choose from {_SUPPORTED}.")

    if reverse:
        lo, hi = hi, lo

    def fn(x: float) -> float:
        if hi == lo:
            return 50.0
        t = float(np.clip((hi - x) / (hi - lo), 0.0, 1.0))
        if shape == "convex":
            return t ** 2 * 100.0
        elif shape == "concave":
            return float(np.sqrt(t)) * 100.0
        else:  # sigmoid
            return float(1.0 / (1.0 + np.exp(-8.0 * (t - 0.5))) * 100.0)

    return fn


def make_pref_fn(
    lo: float,
    hi: float,
    shape: str = "linear",
    *,
    reverse: bool = False,
) -> Callable[[float], float]:
    """
    Dispatch to the correct preference-function constructor.

    Parameters
    ----------
    lo, hi:
        Raw objective bounds.
    shape:
        ``"linear"`` (default), ``"convex"``, ``"concave"``, or ``"sigmoid"``.
    reverse:
        Flip lo/hi before constructing the function (maximisation direction).
    """
    if shape == "linear":
        return make_pref_linear(lo, hi, reverse=reverse)
    return make_pref_nonlinear(lo, hi, shape, reverse=reverse)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAP config builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_imap_config(
    n_obj: int,
    obj_bounds: list[tuple[float, float]],
    obj_names: list[str],
    stakeholder_configs: list[dict] | None = None,
    *,
    default_shape: str = "linear",
) -> tuple[
    dict[int, dict[str, Callable[[float], float]]],
    dict[int, dict[str, float]],
    list[float],
]:
    """
    Build the three structures required by IMAPGA / BRKGAVRPTWAlgorithm.

    Parameters
    ----------
    n_obj:
        Number of objectives.
    obj_bounds:
        ``[(lo, hi), ...]`` — one pair per objective in *obj_names* order.
    obj_names:
        Objective name strings (used as dict keys).
    stakeholder_configs:
        List of stakeholder dicts, each with optional keys:

        ``"obj_weights"``   — ``list[float]`` of length *n_obj* (normalised
                              internally; defaults to equal weights).
        ``"weight"``        — importance of this stakeholder (default ``1.0``).
        ``"pref_shape"``    — ``"linear"`` | ``"convex"`` | ``"concave"``
                              | ``"sigmoid"`` (default *default_shape*).
        ``"reverse"``       — ``bool`` (default ``False``).  Set ``True`` to
                              flip lo/hi so higher raw value → higher preference.

        When ``None`` or empty, a single equal-weight stakeholder is used.
    default_shape:
        Preference-function shape to fall back on when a stakeholder dict does
        not specify ``"pref_shape"``.

    Returns
    -------
    preference_functions : ``dict[int, dict[str, Callable]]``
    objective_weights    : ``dict[int, dict[str, float]]``
    stakeholder_weights  : ``list[float]``
    """
    if not stakeholder_configs:
        stakeholder_configs = [
            {
                "obj_weights": [1.0 / n_obj] * n_obj,
                "weight": 1.0,
                "pref_shape": default_shape,
                "reverse": False,
            }
        ]

    preference_functions: dict[int, dict[str, Callable[[float], float]]] = {}
    objective_weights:    dict[int, dict[str, float]] = {}
    stakeholder_weights:  list[float] = []

    for sid, cfg in enumerate(stakeholder_configs, start=1):
        raw_w   = list(cfg.get("obj_weights", [1.0 / n_obj] * n_obj))
        s_w     = float(cfg.get("weight", 1.0))
        shape   = cfg.get("pref_shape", default_shape)
        reverse = bool(cfg.get("reverse", False))

        if len(raw_w) != n_obj:
            print(
                f"  [imap_helpers] Stakeholder {sid}: obj_weights length "
                f"{len(raw_w)} != n_obj={n_obj}; using equal weights."
            )
            raw_w = [1.0 / n_obj] * n_obj

        w_sum = sum(raw_w)
        if abs(w_sum - 1.0) > 1e-6:
            raw_w = [w / w_sum for w in raw_w]

        preference_functions[sid] = {
            name: make_pref_fn(lo, hi, shape, reverse=reverse)
            for name, (lo, hi) in zip(obj_names, obj_bounds)
        }
        objective_weights[sid] = dict(zip(obj_names, raw_w))
        stakeholder_weights.append(s_w)

    return preference_functions, objective_weights, stakeholder_weights


# ═══════════════════════════════════════════════════════════════════════════════
# IMAP score computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_imap_scores(
    F_matrix: np.ndarray,
    obj_names: list[str],
    preference_functions: dict[int, dict[str, Callable[[float], float]]],
    objective_weights: dict[int, dict[str, float]],
    stakeholder_weights: list[float],
) -> np.ndarray:
    """
    Compute IMAP aggregated scores for a matrix of objective vectors.

    Mirrors ``IMAPGA.get_aggregated_scores`` exactly so that post-hoc
    comparisons use the same arithmetic as the algorithm did during evolution.

    Parameters
    ----------
    F_matrix:
        Shape ``(n_solutions, n_obj)``.
    obj_names:
        Objective name strings in column order of *F_matrix*.
    preference_functions:
        ``{stakeholder_id: {obj_name: callable}}``
    objective_weights:
        ``{stakeholder_id: {obj_name: float}}``
    stakeholder_weights:
        One weight per stakeholder (normalised internally).

    Returns
    -------
    np.ndarray of shape ``(n_solutions,)`` with scores in ``[0, 100]``.
    """

    total_sw    = sum(stakeholder_weights)
    s_norm      = [sw / total_sw for sw in stakeholder_weights]
    w: list[float]       = []
    p: list[list[float]] = []

    for sid, sw in zip(sorted(preference_functions), s_norm):
        for j, obj_name in enumerate(obj_names):
            fn  = preference_functions[sid][obj_name]
            o_w = objective_weights[sid][obj_name]
            p.append([fn(float(v)) for v in F_matrix[:, j]])
            w.append(o_w * sw)

    return a_fine_aggregator(w, p)  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════════
# IMAP-preferred representative from a Pareto front
# ═══════════════════════════════════════════════════════════════════════════════

def imap_best_from_front(
    pareto_F: np.ndarray,
    obj_names: list[str],
    preference_functions: dict[int, dict[str, Callable[[float], float]]],
    objective_weights: dict[int, dict[str, float]],
    stakeholder_weights: list[float],
) -> np.ndarray:
    """
    Return the IMAP-preferred representative from a Pareto front.

    Parameters
    ----------
    pareto_F:
        Shape ``(k, n_obj)`` array of non-dominated objective vectors.
    obj_names, preference_functions, objective_weights, stakeholder_weights:
        Same objects passed to :func:`compute_imap_scores`.

    Returns
    -------
    np.ndarray of shape ``(n_obj,)`` — the row of *pareto_F* with the highest
    IMAP score.
    """
    if len(pareto_F) == 1:
        return pareto_F[0]
    scores = compute_imap_scores(
        pareto_F, obj_names, preference_functions, objective_weights, stakeholder_weights
    )
    return pareto_F[int(np.argmax(scores))]


# ═══════════════════════════════════════════════════════════════════════════════
# IMAP post-hoc comparison
# ═══════════════════════════════════════════════════════════════════════════════

def run_imap_comparison(
    algo_labels: list[str],
    best_F_per_algo: list[np.ndarray],
    obj_names: list[str],
    preference_functions: dict[int, dict[str, Callable[[float], float]]],
    objective_weights: dict[int, dict[str, float]],
    stakeholder_weights: list[float],
    *,
    context_label: str = "",
    logger=None,
) -> dict[str, float]:
    """
    Rank one representative solution per algorithm using IMAP affine
    aggregation on the combined pool.

    Parameters
    ----------
    algo_labels:
        Algorithm name strings, one per representative.
    best_F_per_algo:
        Corresponding objective vectors, each of shape ``(n_obj,)``.
    obj_names:
        Objective name strings in column order.
    preference_functions, objective_weights, stakeholder_weights:
        IMAP config (output of :func:`build_imap_config`).
    context_label:
        Short string added to log messages (e.g. problem or instance name).
    logger:
        Optional logger object with an ``info`` / ``warning`` method.
        Falls back to ``print`` when ``None``.

    Returns
    -------
    dict mapping algorithm label → IMAP score in ``[0, 100]``.
    NaN is returned for any algorithm whose representative was ``None``
    (excluded from the pool before scoring).
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def _warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

    valid_pairs = [
        (lbl, F) for lbl, F in zip(algo_labels, best_F_per_algo) if F is not None
    ]
    nan_labels  = [lbl for lbl, F in zip(algo_labels, best_F_per_algo) if F is None]

    if len(valid_pairs) < 2:
        _warn(
            f"{context_label + ': ' if context_label else ''}"
            "fewer than 2 representatives — IMAP comparison skipped."
        )
        return {lbl: float("nan") for lbl in algo_labels}

    labels  = [lbl for lbl, _ in valid_pairs]
    F_pool  = np.array([F for _, F in valid_pairs])   # (n_algos, n_obj)

    scores    = compute_imap_scores(
        F_pool, obj_names, preference_functions, objective_weights, stakeholder_weights
    )
    score_map: dict[str, float] = {lbl: float(s) for lbl, s in zip(labels, scores)}
    for lbl in nan_labels:
        score_map[lbl] = float("nan")

    winners = [lbl for lbl, s in zip(labels, scores) if s >= 100.0 - 1e-9]
    if not winners:
        winners = [labels[int(np.argmax(scores))]]
    prefix = f"  {context_label}: " if context_label else "  "
    if len(winners) == 1:
        _log(f"{prefix}IMAP winner: {winners[0]}")
    else:
        _log(f"{prefix}IMAP winners (tied at 100): {', '.join(winners)}")

    # Pre-compute preference scores per stakeholder per objective for every algo
    # Shape: pref_table[sid][obj_name] = [score_algo0, score_algo1, ...]
    stakeholder_ids = sorted(preference_functions)
    pref_table: dict[int, dict[str, list[float]]] = {
        sid: {
            obj: [preference_functions[sid][obj](float(F_pool[i, j]))
                  for i in range(len(labels))]
            for j, obj in enumerate(obj_names)
        }
        for sid in stakeholder_ids
    }

    for algo_idx, (lbl, F) in enumerate(zip(labels, F_pool)):
        obj_str  = "  ".join(f"{n}={v:.2f}" for n, v in zip(obj_names, F))
        pref_parts: list[str] = []
        for sid in stakeholder_ids:
            for obj in obj_names:
                pref_parts.append(f"S{sid}.{obj}={pref_table[sid][obj][algo_idx]:.2f}")
        pref_str = "  ".join(pref_parts)
        _log(f"    {lbl:<16s}: {obj_str}  |  {pref_str}  |  IMAP={score_map[lbl]:.2f}")

    return score_map
