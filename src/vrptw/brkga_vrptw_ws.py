"""
BRKGA-WS for multi-objective VRPTW.

Weighted-sum scalarization variant of BRKGAVRPTWAlgorithm.
Replaces the two-pass IMAP survival with a single-pass dynamic weighted sum:
objectives are normalized per generation using the population min/max,
then combined with the provided weights (lower score = better).

Classes
-------
VRPTWOutputWS : Output
    Generation-by-generation console display.
BRKGAVRPTWAlgorithmWS : Algorithm
    BRKGA generational loop with weighted-sum survival.
"""
from __future__ import annotations

import warnings

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

from optimization.igs_brkga import IGSBRKGACrossover
from .igs_brkga_vrptw import VRPTWProblem  # noqa: F401 — re-exported for callers


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class VRPTWOutputWS(Output):
    """Console display for :class:`BRKGAVRPTWAlgorithmWS`."""

    def __init__(self) -> None:
        super().__init__()
        self.distance = Column("distance", width=12)
        self.vehicles = Column("vehicles", width=10)
        self.balance  = Column("balance",  width=12)
        self.columns += [self.distance, self.vehicles, self.balance]

    def update(self, algorithm: "BRKGAVRPTWAlgorithmWS") -> None:
        super().update(algorithm)
        cb = algorithm.current_best
        if cb is None:
            return
        F = cb.get("F")
        self.distance.set(f"{float(F[0]):.2f}")  # type: ignore[index]
        self.vehicles.set(int(F[1]))              # type: ignore[index]
        self.balance.set(f"{float(F[2]):.2f}")   # type: ignore[index]


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class BRKGAVRPTWAlgorithmWS(Algorithm):
    """
    BRKGA with weighted-sum scalarization for multi-objective VRPTW.

    Generational loop
    -----------------
    1. ``_initialize_infill``  — random initial population.
    2. ``_initialize_advance`` — WS scoring; set current_best.
    3. ``_infill``             — single-pass WS sort → elite + mutants + crossover.
    4. ``_advance``            — merge elites + evaluated infills; rescore; update tracking.

    Parameters
    ----------
    weights:
        Three non-negative floats ``[w_distance, w_n_vehicles, w_route_balance]``.
        Normalized internally to sum to 1.  Defaults to equal weights.
    pop_size:
        Total population size.
    elite_fraction:
        Fraction of population kept as elite each generation.
    mutant_fraction:
        Fraction replaced by random mutants each generation.
    crossover_bias:
        Probability a crossover gene comes from the elite parent.
    """

    all_objectives: list[str] = ["distance", "n_vehicles", "route_balance"]

    def __init__(
        self,
        weights: list[float] | None = None,
        pop_size: int = 200,
        elite_fraction: float = 0.15,
        mutant_fraction: float = 0.10,
        crossover_bias: float = 0.70,
        **kwargs,
    ) -> None:
        if weights is None:
            weights = [1 / 3, 1 / 3, 1 / 3]
        w_arr = np.array(weights, dtype=float)
        self._weights_array = w_arr / w_arr.sum()

        kwargs.setdefault("output", VRPTWOutputWS())
        super().__init__(**kwargs)

        self.pop_size    = pop_size
        self.n_elites    = max(int(elite_fraction * pop_size), 1)
        self.n_mutants   = max(int(mutant_fraction * pop_size), 1)
        self.n_crossover = pop_size - self.n_elites - self.n_mutants

        self.crossover          = IGSBRKGACrossover(bias=crossover_bias)
        self.current_best:      Individual | None  = None
        self.current_best_gen:  int                = 0
        self.hall_of_fame:      list[Individual]   = []
        self._current_elites:   Population | None  = None
        self._gen:              int                = 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_ws_scores(self, F: np.ndarray) -> np.ndarray:
        """
        Normalized weighted sum per individual — lower score = better.

        Each objective is scaled to [0, 1] using the population min/max.
        If all individuals share the same value for an objective the
        denominator is set to 1 (that objective contributes 0 uniformly).
        """
        F = np.atleast_2d(F)
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        denom  = np.where(f_max - f_min > 0.0, f_max - f_min, 1.0)
        F_norm = (F - f_min) / denom
        return F_norm @ self._weights_array  # shape (n,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_current_best(self, pop: Population, scores: np.ndarray) -> None:
        """
        Track the best individual found so far (lower ws_score = better).

        I1 always produces feasible VRPTW routes, so no constraint handling
        is needed.
        """
        best_idx  = int(np.argmin(scores))
        candidate = pop[best_idx]

        if self.current_best is None:
            self.current_best     = candidate
            self.current_best_gen = self._gen
        else:
            cb_x = self.current_best.get("X")
            if not np.array_equal(cb_x, candidate.get("X")):  # type: ignore[arg-type]
                self.hall_of_fame.append(self.current_best)
                self.current_best     = candidate
                self.current_best_gen = self._gen

    # ------------------------------------------------------------------
    # pymoo lifecycle hooks
    # ------------------------------------------------------------------

    def _initialize_infill(self) -> Population:
        X = self.random_state.random((self.pop_size, self.problem.n_var))  # type: ignore[union-attr]
        return Population.new(X=X)

    def _initialize_advance(self, infills: Population | None = None, **kwargs) -> None:
        self.hall_of_fame     = []
        self.current_best     = None
        self.current_best_gen = 0
        self._gen             = 0

        F      = infills.get("F")  # type: ignore[union-attr]
        scores = self.get_ws_scores(F)  # type: ignore[arg-type]
        infills.set("ws_score", scores)  # type: ignore[union-attr]

        self._update_current_best(infills, scores)  # type: ignore[arg-type]
        self.pop = infills

    def _infill(self) -> Population:
        """
        Single-pass WS sort → unevaluated mutants + crossover offspring.

        Steps
        -----
        1. Score current population.
        2. Sort ascending (no feasibility split — VRPTW is always feasible).
        3. Take top n_elites as elite, rest as non-elite.
        4. Generate n_mutants random immigrants.
        5. Generate n_crossover parametrized uniform crossover offspring.
        6. Return mutants + crossover as unevaluated Population.
        """
        F      = self.pop.get("F")  # type: ignore[union-attr]
        scores = self.get_ws_scores(F)  # type: ignore[arg-type]

        sort_idx      = np.argsort(scores)          # ascending: lower ws = better
        elite_idx     = sort_idx[:self.n_elites]
        non_elite_idx = sort_idx[self.n_elites:]

        self._current_elites = self.pop[elite_idx]     # type: ignore[index]
        non_elites           = self.pop[non_elite_idx]  # type: ignore[index]

        # Mutants
        X_mutants = self.random_state.random((self.n_mutants, self.problem.n_var))  # type: ignore[union-attr]

        # Crossover (elite × non-elite)
        elite_X = self._current_elites.get("X")  # type: ignore[union-attr]
        if len(non_elites) > 0:  # type: ignore[arg-type]
            non_elite_X = non_elites.get("X")  # type: ignore[union-attr]
            idx_B = self.random_state.integers(0, len(non_elites), size=self.n_crossover)  # type: ignore[union-attr]
            X_B = non_elite_X[idx_B]
        else:
            idx_B_e = self.random_state.integers(  # type: ignore[union-attr]
                0, len(self._current_elites), size=self.n_crossover  # type: ignore[arg-type]
            )
            X_B = elite_X[idx_B_e]

        idx_A  = self.random_state.integers(  # type: ignore[union-attr]
            0, len(self._current_elites), size=self.n_crossover  # type: ignore[arg-type]
        )
        X_A = elite_X[idx_A]

        X_cross = self.crossover._do(
            self.problem,
            np.stack([X_A, X_B]),        # (2, n_crossover, n_var)
            random_state=self.random_state,
        )[0]                              # (n_crossover, n_var)

        return Population.new(X=np.vstack([X_mutants, X_cross]))

    def _advance(self, infills: Population | None = None, **kwargs) -> None:
        assert infills is not None,           "_advance called without infills."
        assert self._current_elites is not None

        self._gen += 1
        self.pop = Population.merge(self._current_elites, infills)

        F      = self.pop.get("F")  # type: ignore[union-attr]
        scores = self.get_ws_scores(F)  # type: ignore[arg-type]
        self.pop.set("ws_score", scores)  # type: ignore[union-attr]

        self._update_current_best(self.pop, scores)  # type: ignore[arg-type]

        assert len(self.pop) == self.pop_size, (  # type: ignore[arg-type]
            f"Population size invariant violated: {len(self.pop)} ≠ {self.pop_size}"  # type: ignore[arg-type]
        )

    def _set_optimum(self, **kwargs) -> None:
        if self.current_best is not None:
            self.opt = Population.create(self.current_best)
        elif self.pop is not None and len(self.pop) > 0:  # type: ignore[arg-type]
            scores = self.pop.get("ws_score")
            if scores is not None:
                self.opt = self.pop[[int(np.argmin(scores))]]  # type: ignore[index]
            else:
                warnings.warn(
                    "_set_optimum called before ws_score was set; "
                    "falling back to first individual.",
                    stacklevel=2,
                )
                self.opt = self.pop[[0]]  # type: ignore[index]
