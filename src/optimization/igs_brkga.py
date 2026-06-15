"""
IGSBRKGA: pymoo-native Biased Random-Key Genetic Algorithm with IMAP selection.

Population partitioning follows strict BRKGA rules: elite, mutants, crossover.

Classes
-------
IGSBRKGACrossover : Crossover
    Parametrized uniform crossover biased towards the elite parent.
IGSBRKGAProblem : Problem
    Wraps Decoder + compute_fitness for pymoo evaluation.
IGSBRKGA : Algorithm
    IGSBRKGA generational loop with two-pass IMAP survival and
    current_best / hall_of_fame tracking.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.crossover import Crossover
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

from components.ga_types import Chromosome, FitnessVector
from optimization._aggregator import a_fine_aggregator
from utils.helper_functions import post_process_current_results


def _compute_fitness(
    solution,
    model,
    all_objectives: list[str],
    add_penalty: bool = True,
) -> FitnessVector:
    """Evaluate a decoded solution and compute its fitness vector, including constraint penalties."""
    sorted_assignments_per_vessel = solution.vessel_sequences
    start_times = solution.start_times.tolist()
    locations = solution.locations.tolist()

    results_totals = post_process_current_results(
        model=model,
        sorted_assignments_per_vessel=sorted_assignments_per_vessel,
        start_times=start_times,
        locations=locations,
        idle_discount=model.idle_cost_discount,
        speed_indices=solution.speed_indices,
        assigned_vessels=solution.assigned_vessels,
    )

    # --- TIER 1: Unassigned Role Penalty ---
    max_day_rate = max(v.day_rate_mob for v in model.vessels)
    max_horizon = max(act.possible_start[-1] + act.duration for act in model.activities)
    P_hard = max_day_rate * max_horizon * model.num_vessels

    num_unassigned = sum(
        1 for r in model.roles
        if solution.assigned_vessels[r.role_id] == -1
    )
    tier1_penalty = num_unassigned * P_hard

    # --- TIER 2: Time Window Violation Penalty ---
    P_soft = max_day_rate
    time_window_penalty = 0.0
    for idx_a in range(model.num_activities):
        act = model.activities[idx_a]
        st = start_times[idx_a]
        earliest = act.possible_start[0]
        latest = act.possible_start[-1]
        if st < earliest:
            time_window_penalty += (earliest - st) * P_soft
        elif st > latest:
            time_window_penalty += (st - latest) * P_soft

    # --- Precompute role lookup: (act_idx, v_idx) → role_id ---
    role_lookup: dict[tuple[int, int], int] = {}
    for r in model.roles:
        v_idx = solution.assigned_vessels[r.role_id]
        if v_idx >= 0:
            key = (r.parent_activity_idx, int(v_idx))
            if key not in role_lookup:
                role_lookup[key] = r.role_id

    # --- TIER 3: Travel Feasibility Penalty ---
    P_travel = max_day_rate
    travel_penalty = 0.0
    for v_idx in range(model.num_vessels):
        sequence = sorted_assignments_per_vessel[v_idx]
        if len(sequence) < 2:
            continue
        vessel = model.vessels[v_idx]
        dist_matrix = model.vessel_travel_dist_matrix_float[v_idx]

        for i in range(1, len(sequence)):
            prev_act_idx = sequence[i - 1]
            curr_act_idx = sequence[i]
            prev_activity = model.activities[prev_act_idx]
            curr_activity = model.activities[curr_act_idx]

            prev_end = start_times[prev_act_idx] + prev_activity.duration
            curr_start = start_times[curr_act_idx]
            available_time = curr_start - prev_end

            prev_end_loc = model._get_global_loc_id(
                prev_activity, locations[prev_act_idx], is_start_loc=False
            )
            curr_start_loc = model._get_global_loc_id(
                curr_activity, locations[curr_act_idx], is_start_loc=True
            )
            dist = float(dist_matrix[prev_end_loc, curr_start_loc])

            curr_role_id = role_lookup.get((curr_act_idx, v_idx))
            if (
                curr_role_id is not None
                and solution.speed_indices is not None
                and curr_role_id < len(solution.speed_indices)
                and solution.speed_indices[curr_role_id] >= 0
            ):
                s_idx = int(solution.speed_indices[curr_role_id])
                s_idx = max(0, min(s_idx, len(vessel.possible_speeds) - 1))
                speed_kn = float(vessel.possible_speeds[s_idx])
            else:
                speed_kn = float(vessel.possible_speeds[-1]) if vessel.possible_speeds else 1.0

            if dist <= 0 or speed_kn <= 0:
                travel_time = 0
            else:
                travel_time = int(math.ceil(dist / (speed_kn * 24.0)))

            if available_time < travel_time:
                travel_penalty += (travel_time - available_time) * P_travel

    # --- AGGREGATE ---
    cost = results_totals["cost"]
    total_penalty = tier1_penalty + time_window_penalty + travel_penalty
    if add_penalty:
        cost += total_penalty + solution.unassigned_penalty

    _raw_objectives = {
        "cost":     float(cost),
        "distance": float(results_totals["distance"]),
        "duration": float(results_totals["duration"]),
        "fuel":     float(results_totals["fuel"]),
    }
    return FitnessVector(
        distance=float(results_totals["distance"]),
        cost=float(cost),
        fuel=float(results_totals["fuel"]),
        duration=float(results_totals["duration"]),
        result=tuple(_raw_objectives[obj] for obj in all_objectives),
        constraint_penalty=float(total_penalty + solution.unassigned_penalty),
    )


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class IGSBRKGACrossover(Crossover):
    """
    Parametrized uniform crossover biased towards the elite parent.

    Each gene is inherited from the elite parent with probability ``bias``
    (default 0.7) and from the non-elite parent with probability ``1 - bias``.

    Parameters
    ----------
    bias:
        Probability that a gene comes from the elite parent.  Must satisfy
        ``bias > 0.5`` to maintain the elite bias property.
    """

    def __init__(self, bias: float = 0.7, **kwargs) -> None:
        super().__init__(n_parents=2, n_offsprings=1, **kwargs)
        self.bias = bias

    def _do(self, problem, X, random_state=None, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        X:
            Shape ``(2, n_matings, n_var)``.  ``X[0]`` = elite parents,
            ``X[1]`` = non-elite parents.
        random_state:
            Seeded generator for reproducible crossover.

        Returns
        -------
        np.ndarray
            Shape ``(1, n_matings, n_var)``.
        """
        rng = random_state if random_state is not None else np.random.default_rng()
        mask = rng.random(X[0].shape) < self.bias
        offspring = np.where(mask, X[0], X[1])
        return offspring[np.newaxis, ...]  # shape: (1, n_matings, n_var)


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

class IGSBRKGAProblem(Problem):
    """
    Pymoo Problem wrapping the Decoder and compute_fitness for vessel scheduling.

    Decision variable packing (all values in [0, 1)):
        [start_time_keys (n_act) | role_priority_keys (n_role) |
         location_keys (n_maint) | speed_keys (n_role) | start_window_keys (n_act)]

    chromosome_size = 2 * n_activities + 2 * n_roles + n_maintenance_slots

    Objectives (F columns, minimisation):
        0: cost
        1: distance
        2: duration
        3: fuel

    Constraints (G columns, pymoo: G <= 0 is feasible):
        0: constraint_penalty  (0 = feasible, > 0 = infeasible)

    Parameters
    ----------
    model:
        SchedulingModel instance.
    decoder:
        Decoder instance (any strategy).
    """

    def __init__(
        self,
        model,
        decoder,
        all_objectives: list[str] | None = None,
        **kwargs,
    ) -> None:
        n_activities = model.num_activities
        n_roles = model.num_roles
        n_maintenance = sum(1 for a in model.activities if a.is_maintenance_type())
        chromosome_size = 2 * n_activities + 2 * n_roles + n_maintenance

        self.all_objectives: list[str] = (
            all_objectives if all_objectives is not None
            else ["cost", "distance", "duration", "fuel"]
        )

        # Slice boundaries for chromosome unpacking
        self._s0 = 0
        self._s1 = n_activities                    # start_time_keys
        self._s2 = self._s1 + n_roles              # role_priority_keys
        self._s3 = self._s2 + n_maintenance        # location_keys
        self._s4 = self._s3 + n_roles              # speed_keys
        self._s5 = self._s4 + n_activities         # start_window_keys

        super().__init__(
            n_var=chromosome_size,
            n_obj=len(self.all_objectives),
            n_ieq_constr=1,
            xl=0.0,
            xu=1.0,
            **kwargs,
        )

        self.model = model
        self.decoder = decoder

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        Evaluate a batch of chromosomes.

        Parameters
        ----------
        X:
            Shape ``(pop_size, n_var)``.
        out:
            Dict populated with ``"F"`` (objectives) and ``"G"`` (constraints).
        """
        pop_size = X.shape[0]
        F = np.zeros((pop_size, len(self.all_objectives)))
        G = np.zeros((pop_size, 1))

        for i in range(pop_size):
            flat = X[i]
            chromosome = Chromosome(
                id=i,
                start_time_keys=flat[self._s0:self._s1],
                role_priority_keys=flat[self._s1:self._s2],
                location_keys=flat[self._s2:self._s3],
                speed_keys=flat[self._s3:self._s4],
                start_window_keys=flat[self._s4:self._s5],
            )
            decoded = self.decoder.decode(chromosome)
            fitness = _compute_fitness(decoded, self.model, self.all_objectives, add_penalty=True)

            F[i] = fitness.result
            G[i, 0] = fitness.constraint_penalty

        out["F"] = F
        out["G"] = G


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class IGSBRKGAOutput(Output):
    """
    Prints generation-by-generation progress for IGSBRKGA.

    Columns shown:
      n_gen      — current generation number (provided by pymoo automatically)
      n_eval     — total evaluations (provided by pymoo automatically)
      feasible   — whether current_best has CV == 0 (yes / no)
      <obj>      — one column per active objective, in all_objectives order
      hof_size   — number of solutions in hall_of_fame
    """

    def __init__(self, all_objectives: list[str]) -> None:
        super().__init__()
        self.feasible = Column("feasible", width=10)
        self._obj_columns: dict[str, Column] = {
            obj: Column(obj, width=14) for obj in all_objectives
        }
        self.hof_size = Column("hof_size", width=10)
        self.columns += (
            [self.feasible]
            + list(self._obj_columns.values())
            + [self.hof_size]
        )

    def update(self, algorithm) -> None:
        super().update(algorithm)

        cb = algorithm.current_best
        if cb is None:
            return  # columns remain None → displayed as "-"

        cv = cb.get("CV")
        if cv is None:
            self.feasible.set("unknown")
        else:
            self.feasible.set("yes" if float(np.max(cv)) <= 0.0 else "no")

        F = cb.get("F")
        for i, col in enumerate(self._obj_columns.values()):
            col.set(f"{float(F[i]):.2f}")
        self.hof_size.set(len(algorithm.hall_of_fame))


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class IGSBRKGA(Algorithm):
    """
    IGSBRKGA: pymoo-native BRKGA with IMAP affine aggregation selection.

    Generational loop:
    1. ``_initialize_infill``  — random initial population.
    2. ``_initialize_advance`` — single-pass IMAP scoring; set current_best.
    3. ``_infill``             — two-pass IMAP partitioning → elite + mutants + crossover.
    4. ``_advance``            — merge elites + evaluated infills; rescore; update tracking.

    Parameters
    ----------
    preference_functions:
        Outer key: stakeholder ID (int, 1-indexed).
        Inner key: objective name (``"cost"``, ``"distance"``, ``"duration"``, ``"fuel"``).
        Value: callable mapping one raw objective value → preference score in [0, 100].
    objective_weights:
        Per-stakeholder weight per objective.  Should sum to 1 per stakeholder.
    stakeholder_weights:
        One weight per stakeholder (normalised internally).
    pop_size:
        Total population size.  Equals ``n_elites + n_mutants + n_crossover``.
    elite_fraction:
        Fraction of population kept as elite each generation.
    mutant_fraction:
        Fraction of population replaced by random mutants each generation.
    crossover_bias:
        Probability that a crossover gene comes from the elite parent (default 0.7).
    **kwargs:
        Forwarded to :class:`~pymoo.core.algorithm.Algorithm` (e.g. ``seed``, ``verbose``).
    """

    def __init__(
        self,
        preference_functions: dict[int, dict[str, Callable]],
        objective_weights: dict[int, dict[str, float]],
        stakeholder_weights: list[float],
        pop_size: int = 200,
        elite_fraction: float = 0.1,
        mutant_fraction: float = 0.2,
        crossover_bias: float = 0.7,
        cv_pref_fn: Callable[[float], float] | None = None,
        cv_weight: float = 0.5,
        **kwargs,
    ) -> None:
        # Derive active objectives from preference_functions keys (Option A).
        # Sorted alphabetically so column order is deterministic and matches
        # IGSBRKGAProblem when both are built from the same preference_functions.
        self.all_objectives: list[str] = sorted(
            {obj for prefs in preference_functions.values() for obj in prefs}
        )
        self._f_col_map: list[str] = self.all_objectives

        kwargs.setdefault("output", IGSBRKGAOutput(self.all_objectives))
        super().__init__(**kwargs)

        self.preference_functions = preference_functions
        self.objective_weights = objective_weights
        self.stakeholder_weights = stakeholder_weights
        self.pop_size = pop_size
        self.n_elites = max(int(elite_fraction * pop_size), 1)
        self.n_mutants = max(int(mutant_fraction * pop_size), 1)
        self.n_crossover = pop_size - self.n_elites - self.n_mutants

        self.crossover = IGSBRKGACrossover(bias=crossover_bias)

        self.current_best: Individual | None = None
        self.current_best_gen: int = 0
        self._gen: int = 0
        self.hall_of_fame: list[Individual] = []
        self._current_elites: Population | None = None

        if cv_pref_fn is not None and not (0.0 < cv_weight < 1.0):
            raise ValueError(f"cv_weight must be in (0, 1), got {cv_weight}")
        self.cv_pref_fn: Callable[[float], float] | None = cv_pref_fn
        self.cv_weight: float = cv_weight

    # ------------------------------------------------------------------
    # Aggregation 
    # ------------------------------------------------------------------

    def get_aggregated_scores(
        self, objective_values: dict[str, list[float]]
    ) -> np.ndarray:
        """
        Compute IMAP preference scores for a pool of individuals.

        Parameters
        ----------
        objective_values:
            Maps each objective name to a list of raw values, one per individual.

        Returns
        -------
        np.ndarray
            1-D array of aggregated preference scores in [0, 100].
        """
        self.stakeholders = self.preference_functions.keys()
        total_weight = sum(self.stakeholder_weights)
        s_weights = [w / total_weight for w in self.stakeholder_weights]

        p: list[list[float]] = []
        w: list[float] = []

        for stakeholder in self.stakeholders:
            for objective in self.all_objectives:
                fn = self.preference_functions.get(stakeholder, {}).get(
                    objective, lambda x: 0
                )
                p.append([fn(x) for x in objective_values[objective]])
                w.append(
                    self.objective_weights[stakeholder][objective]
                    * s_weights[stakeholder - 1]
                )

        # ── Optional CV signal ──────────────────────────────────────────────
        # When a CV preference function is configured, scale all stakeholder
        # weights down to (1 - cv_weight) and append the CV criterion with
        # weight cv_weight.  The total then sums to 1 as required.
        # When the population is fully feasible all CV scores equal 100, giving
        # std ≈ 0; a_fine_aggregator replaces that with 1e-6 so the CV term
        # contributes essentially nothing — objective quality decides the rank.
        if self.cv_pref_fn is not None and "cv" in objective_values:
            scale = 1.0 - self.cv_weight
            w = [wi * scale for wi in w]
            w.append(self.cv_weight)
            p.append([self.cv_pref_fn(cv) for cv in objective_values["cv"]])

        return a_fine_aggregator(w, p) # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _f_to_objective_values(
        self,
        F: np.ndarray,
        G: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Map F array columns to objective_values dict expected by get_aggregated_scores.

        When cv_pref_fn is set and G is provided, adds a "cv" key containing
        the total constraint violation per individual (sum(max(0, g_j))).
        """
        F = np.atleast_2d(F)
        obj_vals: dict[str, list[float]] = {
            name: F[:, i].tolist() for i, name in enumerate(self._f_col_map)
        }
        if self.cv_pref_fn is not None and G is not None:
            G2d = np.atleast_2d(G)
            obj_vals["cv"] = np.sum(np.maximum(0.0, G2d), axis=1).tolist()
        return obj_vals

    def _update_current_best(
        self,
        pop: Population,
        scores: np.ndarray,
        CV: np.ndarray,
    ) -> None:
        """
        Update ``current_best`` and ``hall_of_fame`` from the scored population.

        Feasibility-first: prefers individuals with CV == 0; breaks ties by score.
        Appends old current_best to hall_of_fame only when a genuinely different
        individual displaces it.
        """
        feasible_mask = CV == 0
        if feasible_mask.any():
            candidate_idx = int(np.argmax(np.where(feasible_mask, scores, -np.inf)))
        else:
            candidate_idx = int(np.argmax(scores))

        candidate = pop[candidate_idx]

        if self.current_best is None:
            self.current_best = candidate
            self.current_best_gen = self._gen
        else:
            cb_x = self.current_best.get("X")
            if not np.array_equal(cb_x, candidate.get("X")): # type: ignore
                self.hall_of_fame.append(self.current_best)
                self.current_best = candidate
                self.current_best_gen = self._gen

    # ------------------------------------------------------------------
    # pymoo lifecycle hooks
    # ------------------------------------------------------------------

    def _initialize_infill(self) -> Population:
        """Generate the initial random population (not yet evaluated)."""
        X = self.random_state.random((self.pop_size, self.problem.n_var)) # type: ignore
        return Population.new(X=X)

    def _initialize_advance(
        self, infills: Population | None = None, **kwargs
    ) -> None:
        """
        Post-initialisation hook: single-pass IMAP scoring, no discard threshold.

        Called once by ``Algorithm.advance`` after the initial population has been
        evaluated.  Sets up ``current_best``, ``hall_of_fame``, and ``self.pop``.
        """
        self.hall_of_fame = []
        self.current_best = None
        self._gen = 0
        self.current_best_gen = 0

        F = infills.get("F") # type: ignore
        CV = infills.get("CV").flatten() # type: ignore
        obj_vals = self._f_to_objective_values(F, G=infills.get("G")) # type: ignore
        scores = np.clip(self.get_aggregated_scores(obj_vals), 0.0, 100.0)
        infills.set("imap_score", scores) # type: ignore

        self._update_current_best(infills, scores, CV) # type: ignore
        self.pop = infills

    def _infill(self) -> Population:
        """
        Two-pass IMAP partitioning; returns unevaluated mutants + crossover offspring.

        Steps
        -----
        0. Inject ``current_best`` into pool if not already present.
        1. Pass-1 aggregation on full pool.
        2. Discard phase (threshold = 40); force-keep ``current_best``.
        3. Pass-2 aggregation on survivors.
        4. Feasibility-first elite / non-elite partition from pass-2 scores.
        5. Generate ``n_mutants`` random immigrants.
        6. Generate ``n_crossover`` parametrized uniform crossover offspring.
        7. Return mutants + crossover as unevaluated ``Population``.
        """
        # --- Step 0: inject current_best ---
        pool = self.pop
        if self.current_best is not None:
            cb_x = self.current_best.get("X")
            already_in = any(
                np.array_equal(cb_x, pool[i].get("X")) # type: ignore
                for i in range(len(pool)) # type: ignore
            )
            if not already_in:
                pool = Population.merge(pool, Population.create(self.current_best))

        # --- Step 1: Pass-1 aggregation ---
        F1 = pool.get("F") # type: ignore
        CV = pool.get("CV").flatten() # type: ignore
        obj_vals1 = self._f_to_objective_values(F1, G=pool.get("G")) # type: ignore
        scores_pass1 = np.clip(self.get_aggregated_scores(obj_vals1), 0.0, 100.0)

        # --- Step 2: Discard phase (threshold = 40) ---
        keep_mask = scores_pass1 > 40.0
        n_natural_keep = int(keep_mask.sum())
        cb_x = self.current_best.get("X") if self.current_best is not None else None

        if n_natural_keep < self.pop_size:
            candidates = pool
            # force-keep patch redundant: candidates = pool already contains current_best
        else:
            # Force-keep current_best BEFORE slicing (not counted in n_natural_keep)
            if cb_x is not None:
                for i in range(len(pool)): # type: ignore
                    if np.array_equal(pool[i].get("X"), cb_x): # type: ignore
                        keep_mask[i] = True
                        break
            candidates = pool[keep_mask] # type: ignore

        # --- Step 3: Pass-2 aggregation ---
        if len(candidates) >= 2: # type: ignore
            F2 = candidates.get("F") # type: ignore
            CV2 = candidates.get("CV").flatten() # type: ignore
            obj_vals2 = self._f_to_objective_values(F2, G=candidates.get("G")) # type: ignore
            scores_pass2 = np.clip(self.get_aggregated_scores(obj_vals2), 0.0, 100.0)
        else:
            CV2 = candidates.get("CV").flatten() if len(candidates) > 0 else np.array([]) # type: ignore
            scores_pass2 = np.full(
                len(candidates), # type: ignore
                100.0 if len(candidates) == 1 else 50.0, # type: ignore
            )

        candidates.set("imap_score", scores_pass2) # type: ignore

        # --- Step 4: Feasibility-first elite / non-elite partition ---
        sort_idx = np.lexsort((-scores_pass2, (CV2 > 0).astype(int)))
        elite_idx = sort_idx[:self.n_elites]
        non_elite_idx = sort_idx[self.n_elites:]

        self._current_elites = candidates[elite_idx] # type: ignore
        non_elites = candidates[non_elite_idx] # type: ignore

        # --- Step 5: Mutants (fully random) ---
        X_mutants = self.random_state.random((self.n_mutants, self.problem.n_var)) # type: ignore

        # --- Step 6: Crossover (elite × non-elite) ---
        elite_X = self._current_elites.get("X") # type: ignore
        if len(non_elites) > 0:
            non_elite_X = non_elites.get("X") # type: ignore
            idx_B = self.random_state.integers(0, len(non_elites), size=self.n_crossover) # type: ignore
            X_B = non_elite_X[idx_B]
        else:
            idx_B_e = self.random_state.integers( # type: ignore
                0, len(self._current_elites), size=self.n_crossover # type: ignore
            )
            X_B = elite_X[idx_B_e]

        idx_A = self.random_state.integers( # type: ignore
            0, len(self._current_elites), size=self.n_crossover # type: ignore
        )
        X_A = elite_X[idx_A]

        X_cross = self.crossover._do(
            self.problem,
            np.stack([X_A, X_B]),      # shape (2, n_crossover, n_var)
            random_state=self.random_state,
        )[0]                            # shape (n_crossover, n_var)

        X_new = np.vstack([X_mutants, X_cross])
        return Population.new(X=X_new)

    def _advance(self, infills: Population | None = None, **kwargs) -> None:
        """
        Assemble next generation from elites + evaluated infills; update tracking.

        Parameters
        ----------
        infills:
            Evaluated mutants + crossover offspring from ``_infill``.
        """
        assert infills is not None, "_advance called without infills."
        self._gen += 1
        self.pop = Population.merge(self._current_elites, infills)

        # Rescore full new population
        F = self.pop.get("F") # type: ignore
        CV = self.pop.get("CV").flatten() # type: ignore
        obj_vals = self._f_to_objective_values(F, G=self.pop.get("G")) # type: ignore
        scores = np.clip(self.get_aggregated_scores(obj_vals), 0.0, 100.0)
        self.pop.set("imap_score", scores) # type: ignore

        self._update_current_best(self.pop, scores, CV) # type: ignore

        assert len(self.pop) == self.pop_size, ( # type: ignore
            f"Population size invariant violated after _advance: "
            f"{len(self.pop)} != {self.pop_size}" # type: ignore
        )

    def _set_optimum(self, **kwargs) -> None:
        """
        Set ``self.opt`` to the IMAP-tracked incumbent (``current_best``).

        Overrides pymoo's default so that ``res.opt`` reflects the best
        solution found across all generations.
        """
        if self.current_best is not None:
            self.opt = Population.create(self.current_best)
        elif self.pop is not None and len(self.pop) > 0:
            scores = self.pop.get("imap_score")
            if scores is not None:
                self.opt = self.pop[[int(np.argmax(scores))]]
            else:
                warnings.warn(
                    "_set_optimum called before imap_score was set on population; "
                    "falling back to first individual.",
                    stacklevel=2,
                )
                self.opt = self.pop[[0]]
