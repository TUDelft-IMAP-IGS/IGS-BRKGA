"""
imap_plotting.py
================
Shared plotting utilities for IMAP benchmark and convergence experiments.

Used by
-------
- test_imap_ga_convergence.py
- benchmark_dascmop.py
- benchmark_vrptw.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level save helper
# ═══════════════════════════════════════════════════════════════════════════════

def savefig(
    fig: plt.Figure, # type: ignore
    path: str | Path,
    *,
    dpi: int = 150,
    logger=None,
) -> None:
    """
    Save *fig* to *path* and close it.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    path:
        Destination file path (any matplotlib-supported extension).
    dpi:
        Output resolution (default 150).
    logger:
        Optional logger object with an ``info`` method.  Falls back to
        ``print`` when ``None``.
    """
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    msg = f"  Saved: {path}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def bar_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
    *,
    group_col: str = "instance",
    hue_col: str = "algorithm",
    palette: dict | None = None,
    hue_order: list[str] | None = None,
    higher_is_better: bool | None = None,
    hline: float | None = None,
    fig_size: tuple[float, float] = (16, 5),
    dpi: int = 150,
    logger=None,
) -> plt.Figure: # type: ignore
    """
    Grouped bar chart of *metric* per group, one bar per algorithm.

    Parameters
    ----------
    df:
        DataFrame containing at least *metric*, *group_col*, and *hue_col*.
    metric:
        Column to plot on the y-axis.
    ylabel:
        Y-axis label.
    title:
        Figure title (a direction hint is appended when *higher_is_better* is
        not ``None``).
    save_path:
        If provided, the figure is saved there via :func:`savefig`.
    group_col:
        Column used for x-axis groups (default ``"instance"``).
    hue_col:
        Column used for bar colours (default ``"algorithm"``).
    palette:
        ``{algorithm_label: colour}`` mapping.
    hue_order:
        Explicit bar ordering within each group.
    higher_is_better:
        When not ``None``, appends ``"↑ higher is better"`` or
        ``"↓ lower is better"`` to the title.
    hline:
        If provided, a horizontal dashed reference line is drawn at this value.
    fig_size:
        ``(width, height)`` in inches.
    dpi:
        Resolution for saving.
    logger:
        Optional logger used by :func:`savefig`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return plt.figure()

    groups = sorted(subset[group_col].unique())
    fig, ax = plt.subplots(figsize=fig_size)

    bar_kwargs: dict = dict(
        data=subset,
        x=group_col, y=metric, hue=hue_col,
        ax=ax,
        edgecolor="white", linewidth=0.8,
        order=groups,
    )
    if palette is not None:
        bar_kwargs["palette"] = palette
    if hue_order is not None:
        bar_kwargs["hue_order"] = hue_order

    sns.barplot(**bar_kwargs)

    if hline is not None:
        ax.axhline(hline, color="green", linestyle="--", linewidth=0.8, alpha=0.7)

    full_title = title
    if higher_is_better is True:
        full_title += "  (↑ higher is better)"
    elif higher_is_better is False:
        full_title += "  (↓ lower is better)"

    ax.set_xlabel(group_col.capitalize())
    ax.set_ylabel(ylabel)
    ax.set_title(full_title, fontsize=13, fontweight="bold")
    ax.legend(title=hue_col.capitalize(), fontsize=8, loc="best")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    if save_path is not None:
        savefig(fig, save_path, dpi=dpi, logger=logger)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# IMAP winner heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def heatmap_imap_winner(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    row_col: str = "instance",
    col_col: str = "algorithm",
    winner_col: str = "imap_winner",
    palette: dict | None = None,
    title: str = "IMAP Post-hoc Comparison Winner  (1 = winner)",
    fig_size: tuple[float, float] | None = None,
    dpi: int = 150,
    logger=None,
) -> plt.Figure: # type: ignore
    """
    Heatmap showing which algorithm won the IMAP post-hoc comparison in each
    row group (problem / instance).

    Parameters
    ----------
    df:
        DataFrame with *row_col*, *col_col*, and *winner_col* columns.
    save_path:
        If provided, the figure is saved there.
    row_col:
        Column used for heatmap rows (default ``"instance"``).
    col_col:
        Column used for heatmap columns (default ``"algorithm"``).
    winner_col:
        Boolean column indicating the winner (default ``"imap_winner"``).
    palette:
        If provided, only algorithms present as keys are included in columns
        (preserving the order).
    title:
        Figure title.
    fig_size:
        ``(width, height)`` in inches.  Auto-sized when ``None``.
    dpi:
        Resolution for saving.
    logger:
        Optional logger.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = sorted(df[row_col].unique())
    if palette is not None:
        cols = [a for a in palette if a in df[col_col].unique()]
    else:
        cols = sorted(df[col_col].unique())

    matrix = pd.DataFrame(0, index=rows, columns=cols)
    for _, row in df[df[winner_col]].iterrows():
        r, c = row[row_col], row[col_col]
        if r in matrix.index and c in matrix.columns:
            matrix.loc[r, c] = 1

    if fig_size is None:
        fig_size = (max(6, len(cols) * 1.4), max(3, len(rows) * 0.6 + 1))

    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlGn",
        cbar=False, linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel(col_col.capitalize())
    ax.set_ylabel(row_col.capitalize())
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        savefig(fig, save_path, dpi=dpi, logger=logger)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Preference-function plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_preference_functions(
    pref_fns: dict[int, dict[str, Callable[[float], float]]],
    obj_bounds: list[tuple[float, float]] | dict[str, tuple[float, float]],
    obj_names: list[str] | None = None,
    *,
    save_path: str | Path | None = None,
    n_points: int = 300,
    fig_size_per_obj: tuple[float, float] = (4.0, 4.0),
    dpi: int = 150,
    logger=None,
) -> plt.Figure: # type: ignore
    """
    Plot each stakeholder's preference function for every objective.

    One subplot per objective (arranged in a row), one line per stakeholder.

    Parameters
    ----------
    pref_fns:
        ``{stakeholder_id: {obj_name: callable}}`` — output of
        :func:`~imap_helpers.build_imap_config`.
    obj_bounds:
        Raw value range for each objective used to span the x-axis.
        Either a list of ``(lo, hi)`` tuples in *obj_names* order, or a dict
        ``{obj_name: (lo, hi)}``.
    obj_names:
        Objective names.  Inferred from the inner dicts of *pref_fns* when
        ``None``.
    save_path:
        If provided, the figure is saved to this path.
    n_points:
        Number of sample points along the x-axis per curve.
    fig_size_per_obj:
        ``(width, height)`` per subplot in inches.
    dpi:
        Resolution for saving.
    logger:
        Optional logger.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if obj_names is None:
        first_sid = next(iter(pref_fns))
        obj_names = list(pref_fns[first_sid].keys())

    if isinstance(obj_bounds, dict):
        bounds_list = [obj_bounds[n] for n in obj_names]
    else:
        bounds_list = list(obj_bounds)

    n_obj = len(obj_names)
    stakeholder_ids = list(pref_fns.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    w, h = fig_size_per_obj
    fig, axes = plt.subplots(1, n_obj, figsize=(w * n_obj, h), sharey=True)
    if n_obj == 1:
        axes = [axes]

    for ax, name, (lo, hi) in zip(axes, obj_names, bounds_list):
        xs = np.linspace(lo, hi, n_points)
        for idx, sid in enumerate(stakeholder_ids):
            fn = pref_fns[sid][name]
            ys = [fn(x) for x in xs]
            ax.plot(xs, ys, color=colors[idx % len(colors)], label=f"Stakeholder {sid}")
        ax.set_title(name)
        ax.set_xlabel("Objective value")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Preference score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        savefig(fig, save_path, dpi=dpi, logger=logger)

    return fig
