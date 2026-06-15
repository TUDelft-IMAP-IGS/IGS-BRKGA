# IGS-BRKGA

Multi-objective constrained optimisation using a **Biased Random-Key Genetic
Algorithm (BRKGA)** with **IMAP** preference-aggregation survival
selection.

---

## What is this?

The **IMAP** method evaluates design alternatives by converting raw performance
values (f1, f2, …) into *preference scores* via stakeholder-defined preference
functions, then aggregating those scores using a population-relative z-score
normalisation (`a_fine_aggregator`).  The key property: **the same solution can
receive a different score in a different generation**, because scoring is relative
to the current population - not absolute.

**IGS-BRKGA** (IMAP Genetic Algorithm with Biased Random-Key encoding) keeps the
strict BRKGA population partitioning - **elite**, **mutants**, **crossover** - but
replaces fitness-proportional ranking with a **two-pass IMAP affine aggregation**.
Solutions are encoded as random keys in `[0, 1]` and translated into feasible
solutions by a problem-specific **decoder**, so any combinatorial problem can be
solved by:

1. defining a decoder that maps a random-key chromosome to a solution
2. defining preference functions and objective bounds for each objective
3. configuring stakeholder weights and preference shapes
4. constructing an `IGSBRKGA` instance and passing it to `pymoo_minimize`

---

## Repository structure

```
IGS-BRKGA/
├── src/
│   ├── components/            # Vessel, Activity, Role, GA dataclasses
│   ├── logic/
│   │   ├── scheduling_model.py   # HVASP constraint model
│   │   └── Decoder.py            # random-key → schedule decoders (BRKGA strategies)
│   ├── optimization/
│   │   ├── igs_brkga.py          # IGSBRKGA algorithm + IGSBRKGAProblem
│   │   └── _aggregator.py        # affine aggregator kernel (no external deps)
│   ├── utils/
│   │   ├── imap_helpers.py       # build_imap_config, preference-function builders
│   │   ├── load_data.py          # Excel/JSON loaders → Vessel/Activity objects
│   │   └── helper_functions.py   # proxy bounds, metrics, post-processing
│   └── vrptw/                # VRPTW problem, decoder, and BRKGA wrapper
├── examples/
│   ├── HVASP/
│   │   ├── hvasp.ipynb           # demo notebook (vessel allocation & scheduling)
│   │   └── generate_experiments.py  # synthetic HVASP instance generator
│   └── VRPTW/
│       ├── vrptw.ipynb           # demo notebook (Solomon VRPTW)
│       └── data/                 # Solomon benchmark instances
└── pyproject.toml
```

---

## Installation

```bash
pip install -e .
```

Runtime dependencies: `numpy`, `pymoo`, `cpmpy`, `pandas`, `openpyxl`,
`loguru`, `werkzeug`, `searoute`, `seaborn`.

---

## Quick start

```python
import time
from datetime import date

import numpy as np
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from vrptw.data    import parse_solomon, compute_theoretical_bounds
from vrptw.decoder import VRPTWDecoder
from vrptw.igs_brkga_vrptw import IGSBRKGAVRPTWAlgorithm, VRPTWProblem
from utils.imap_helpers import build_imap_config, make_pref_cv

# 1. Parse a problem instance and derive analytical objective bounds
instance = parse_solomon("examples/VRPTW/data/C101.txt")
obj_names = ["distance", "n_vehicles", "route_balance"]
bounds = compute_theoretical_bounds(instance)

# 2. Configure stakeholders (one per objective here)
stakeholders = [
    {"obj_weights": [1.0, 0.0, 0.0], "weight": 1/3, "pref_shape": "linear"},
    {"obj_weights": [0.0, 1.0, 0.0], "weight": 1/3, "pref_shape": "linear"},
    {"obj_weights": [0.0, 0.0, 1.0], "weight": 1/3, "pref_shape": "linear"},
]

# 3. Build the IMAP preference structures
preference_functions, objective_weights, stakeholder_weights = build_imap_config(
    n_obj=len(obj_names),
    obj_bounds=[bounds[o] for o in obj_names],
    obj_names=obj_names,
    stakeholder_configs=stakeholders,
)

# 4. Decoder + problem + constraint-violation preference
decoder = VRPTWDecoder(instance, strategy="i1")
problem = VRPTWProblem(instance, decoder)
cv_pref_fn = make_pref_cv()

# 5. Run IGS-BRKGA
algorithm = IGSBRKGAVRPTWAlgorithm(
    preference_functions=preference_functions,
    objective_weights=objective_weights,
    stakeholder_weights=stakeholder_weights,
    pop_size=200,
    elite_fraction=0.15,
    mutant_fraction=0.10,
    crossover_bias=0.70,
    cv_pref_fn=cv_pref_fn,
    cv_weight=0.5,
    seed=42,
)

res = pymoo_minimize(problem, algorithm, get_termination("n_gen", 50), seed=42, copy_algorithm=False)

best   = res.algorithm.current_best
best_F = best.get("F").flatten()
print(f"Best found at gen {res.algorithm.current_best_gen}: "
      f"distance={best_F[0]:.2f}, vehicles={int(best_F[1])}, balance={best_F[2]:.2f}")
```

For the generic scheduling problem (HVASP), pair `IGSBRKGA` with
`IGSBRKGAProblem` and the scheduling `Decoder` instead — see
`examples/HVASP/hvasp.ipynb`.

---

## `IGSBRKGA` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `preference_functions` | `dict` | `{stakeholder_id: {obj_name: callable}}` — maps objective values to preference scores (0–100) |
| `objective_weights` | `dict` | `{stakeholder_id: {obj_name: float}}` — per-stakeholder objective weights (should sum to 1 per stakeholder) |
| `stakeholder_weights` | `list[float]` | Relative importance of each stakeholder (normalised internally) |
| `pop_size` | `int` | Total population size = `n_elites + n_mutants + n_crossover` (default `200`) |
| `elite_fraction` | `float` | Fraction of the population kept as elite each generation (default `0.1`) |
| `mutant_fraction` | `float` | Fraction replaced by random immigrants each generation (default `0.2`) |
| `crossover_bias` | `float` | Probability a crossover gene comes from the elite parent; must be `> 0.5` (default `0.7`) |
| `cv_pref_fn` | `callable \| None` | Preference function for total constraint violation; `None` disables the CV signal |
| `cv_weight` | `float` | Fraction of aggregation weight given to the CV signal, in `(0, 1)` (default `0.5`) |

> The active objective list is derived automatically from the
> `preference_functions` keys, sorted alphabetically.  Ensure your problem's `F`
> column order matches that sorted order (e.g. `IGSBRKGAProblem` defaults to
> `["cost", "distance", "duration", "fuel"]`).

### Stakeholder configuration pattern

```python
STAKEHOLDERS: dict[int, list[dict]] = {
    # Bi-objective problems
    2: [
        {"obj_weights": [0.5, 0.5], "weight": 0.3, "pref_shape": "linear"},
        {"obj_weights": [0.5, 0.5], "weight": 0.7, "pref_shape": "linear", "reverse": True},
    ],
    # Tri-objective problems
    3: [
        {"obj_weights": [1/3, 1/3, 1/3], "weight": 0.6, "pref_shape": "linear"},
        {"obj_weights": [0.2, 0.3, 0.5],  "weight": 0.4, "pref_shape": "linear", "reverse": True},
    ],
}
```

Each stakeholder dict supports:

| Key | Description |
|-----|-------------|
| `obj_weights` | Per-objective weight; must sum to 1 and have length `== n_obj` |
| `weight` | Importance of this stakeholder relative to others |
| `pref_shape` | `"linear"` \| `"convex"` \| `"concave"` \| `"sigmoid"` |
| `reverse` | If `True`, flips preference direction (higher objective value = better) |

`build_imap_config` turns a list of these dicts into the three structures
`IGSBRKGA` expects.

---

## Result attributes

After calling `pymoo_minimize`, the result algorithm object exposes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `res.algorithm.current_best` | `Individual` | Best feasible solution found across all generations |
| `res.algorithm.current_best_gen` | `int` | Generation in which `current_best` was last updated |
| `res.algorithm.hall_of_fame` | `list[Individual]` | All previous `current_best` solutions, in order |

Each scored individual also carries an `imap_score` attribute
(`ind.get("imap_score")`) holding its pool-relative aggregation score.

---

## Preference shape reference

| Shape | Formula | Behaviour |
|-------|---------|-----------|
| `linear` | $p(t) = t \times 100$ | Proportional — default |
| `convex` | $p(t) = t^2 \times 100$ | Rewards proximity to optimum |
| `concave` | $p(t) = \sqrt{t} \times 100$ | Diminishing returns |
| `sigmoid` | $p(t) = \dfrac{100}{1 + e^{-10(t - 0.5)}}$ | Threshold / S-curve behaviour |

where $t = \dfrac{x_{\max} - x}{x_{\max} - x_{\min}} \in [0, 1]$ (higher $t$ = lower, better objective value).

The constraint-violation preference (`make_pref_cv`) maps a feasible solution
(`cv == 0`) to `100` and decays sharply for any positive violation, so
infeasible solutions are always dominated by feasible ones.

---

## Demo notebooks

| Notebook | Problem | Highlights |
|----------|---------|------------|
| `examples/HVASP/hvasp.ipynb` | Heterogeneous Vessel Allocation & Scheduling | Synthetic instance generation, Excel/JSON loading, `SchedulingModel` + `Decoder`, schedule Gantt chart |
| `examples/VRPTW/vrptw.ipynb` | Solomon Vehicle Routing with Time Windows | Analytical bounds, insertion decoders, route plots |

Each notebook walks through the full workflow: derive objective bounds,
configure stakeholders, build the IMAP preference structures, run IGS-BRKGA, and
visualise convergence, the best solution, and the preference curves.

---

## License

This project is licensed under the **MIT License** — see the [`LICENSE`](./LICENSE) file for details.
