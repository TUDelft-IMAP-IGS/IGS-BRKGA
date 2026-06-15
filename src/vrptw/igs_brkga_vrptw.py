"""
IGSBRKGA for multi-objective VRPTW.

Thin subclass of :class:`~optimization.igs_brkga.IGSBRKGA`
with VRPTW-specific defaults and display.

Classes
-------
VRPTWProblem : Problem
    pymoo Problem wrapping the VRPTWDecoder; fleet size constraint G <= 0.
VRPTWOutput : Output
    Generation-by-generation console display.
IGSBRKGAVRPTWAlgorithm : IGSBRKGA
    IGSBRKGA pre-configured for three VRPTW objectives.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from pymoo.core.problem import Problem
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

from optimization.igs_brkga import IGSBRKGA

from .data import VRPTWInstance
from .decoder import VRPTWDecoder


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class VRPTWProblem(Problem):
    """
    pymoo Problem wrapping the :class:`~optimization.igs_brkga.vrptw.decoder.VRPTWDecoder`.

    Decision variables: N+4 values in [0, 1] (Solomon chromosome).
    Objectives (F columns, all minimised):
        0  total_distance
        1  n_vehicles
        2  route_balance
    Inequality constraint (G <= 0 is feasible):
        0  n_vehicles_used - K  (fleet size; capacity and time windows are
           satisfied by construction — the decoder only inserts when feasible
           and opens a new route as fallback).

    Parameters
    ----------
    instance:
        Parsed :class:`~allodyn.vrptw.data.VRPTWInstance`.
    decoder:
        :class:`~allodyn.vrptw.decoder.VRPTWDecoder` instance.
    """

    #: Column order must match the order of ``BRKGAVRPTWAlgorithm.all_objectives``.
    F_COLS = ["distance", "n_vehicles", "route_balance"]

    def __init__(self, instance: VRPTWInstance, decoder: VRPTWDecoder, **kwargs) -> None:
        super().__init__(
            n_var=instance.n_customers + 4,
            n_obj=3,
            n_ieq_constr=1,
            xl=0.0,
            xu=1.0,
            **kwargs,
        )
        self.instance = instance
        self.decoder = decoder

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        pop_size = X.shape[0]
        F = np.zeros((pop_size, 3))
        G = np.zeros((pop_size, 1))
        for i in range(pop_size):
            sol = self.decoder.decode(X[i])
            F[i, 0] = sol.total_distance
            F[i, 1] = float(sol.n_vehicles)
            F[i, 2] = sol.route_balance
            G[i, 0] = float(sol.n_vehicles - self.instance.n_vehicles)
        out["F"] = F
        out["G"] = G


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class VRPTWOutput(Output):
    """Console display for :class:`IGSBRKGAVRPTWAlgorithm`."""

    def __init__(self) -> None:
        super().__init__()
        self.distance  = Column("distance",  width=12)
        self.vehicles  = Column("vehicles",  width=10)
        self.balance   = Column("balance",   width=12)
        self.hof_size  = Column("hof_size",  width=10)
        self.columns += [self.distance, self.vehicles, self.balance, self.hof_size]

    def update(self, algorithm: "IGSBRKGAVRPTWAlgorithm") -> None:
        super().update(algorithm)
        cb = algorithm.current_best
        if cb is None:
            return
        F = cb.get("F")
        self.distance.set(f"{float(F[0]):.2f}") # type: ignore
        self.vehicles.set(int(F[1])) # type: ignore
        self.balance.set(f"{float(F[2]):.2f}") # type: ignore
        self.hof_size.set(len(algorithm.hall_of_fame))


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class IGSBRKGAVRPTWAlgorithm(IGSBRKGA):
    """
    IGSBRKGA for multi-objective VRPTW.

    Inherits the full two-pass IMAP generational loop from
    :class:`~optimization.igs_brkga.IGSBRKGA`.
    VRPTW-specific defaults: ``elite_fraction=0.15``,
    ``mutant_fraction=0.10``, ``crossover_bias=0.70``.

    Parameters
    ----------
    preference_functions:
        Outer key: stakeholder ID (int, 1-indexed).
        Inner key: objective name (``"distance"``, ``"n_vehicles"``,
        ``"route_balance"``).
        Value: callable mapping one raw objective value → preference score
        in [0, 100].
    objective_weights:
        Per-stakeholder weight per objective.  Must sum to 1 per stakeholder.
    stakeholder_weights:
        One weight per stakeholder (normalised internally).
    pop_size:
        Total population size.
    elite_fraction:
        Fraction of population kept as elite each generation.
    mutant_fraction:
        Fraction replaced by random mutants each generation.
    crossover_bias:
        Elite-parent inheritance probability in parametrized uniform crossover.
    """

    #: Fixed objective names — also accessible as a class attribute for callers
    #: that need the list before constructing an instance.
    all_objectives: list[str] = ["distance", "n_vehicles", "route_balance"]

    def __init__(
        self,
        preference_functions: dict[int, dict[str, Callable]],
        objective_weights: dict[int, dict[str, float]],
        stakeholder_weights: list[float],
        pop_size: int = 200,
        elite_fraction: float = 0.15,
        mutant_fraction: float = 0.10,
        crossover_bias: float = 0.70,
        cv_pref_fn: Callable[[float], float] | None = None,
        cv_weight: float = 0.5,
        **kwargs,
    ) -> None:
        kwargs.setdefault("output", VRPTWOutput())
        super().__init__(
            preference_functions=preference_functions,
            objective_weights=objective_weights,
            stakeholder_weights=stakeholder_weights,
            pop_size=pop_size,
            elite_fraction=elite_fraction,
            mutant_fraction=mutant_fraction,
            crossover_bias=crossover_bias,
            cv_pref_fn=cv_pref_fn,
            cv_weight=cv_weight,
            **kwargs,
        )
