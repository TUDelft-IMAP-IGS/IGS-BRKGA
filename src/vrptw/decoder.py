"""
Solomon VRPTW Decoders for BRKGA.

Implements six of Solomon's constructive heuristics as BRKGA decoders, plus a
"random" meta-strategy that deterministically selects one decoder per individual.

Strategies
----------
``"i1"``      — Insertion heuristic I1 (original; c2 maximises depot-distance
                benefit).
``"i2"``      — Insertion heuristic I2 (c1 identical to I1; c2 minimises
                combined route distance + time).
``"i3"``      — Insertion heuristic I3 (c1 adds urgency term alpha_3*(l_u-b_u);
                c2 = c1, so we minimise c1 to pick the next customer).
``"savings"`` — Parallel Clarke-Wright savings with optional waiting-time limit.
``"nn"``      — Time-oriented nearest-neighbour (sequential).
``"sweep"``   — Polar-angle sweep clustering followed by I1 scheduling of each
                cluster.
``"random"``  — Deterministically selects one of the six base strategies per
                chromosome using a bitwise hash of the chromosome.

Chromosome layout  (N = number of customers, chromosome length = N + 4)
-----------------------------------------------------------------------
rk[0..3]   — heuristic parameters (interpretation is strategy-specific;
              see each ``_decode_*`` method for details).
rk[4..N+3] — per-customer bias keys: ``rk_bias[u-1]`` for customer u in 1..N.

Role of per-customer bias keys
-------------------------------
Each heuristic uses the bias keys to *perturb* its selection criterion so that
different chromosomes produce different route structures:

* I1  : ``c2_prime = c2 * rk_bias[u-1]``  (maximise → pick customer)
* I2  : ``c2_eff   = c2 * (1 + 0.2*(rk_bias[u-1] - 0.5))``  (minimise)
* I3  : ``c1_eff   = c1 * (1 + 0.1*(rk_bias[u-1] - 0.5))``  (minimise)
* NN  : ``c_ij_eff = c_ij * (0.5 + rk_bias[j-1])``           (minimise)
* Savings: ``s_eff = s * (1 + alpha*(rk_bias[i-1] + rk_bias[j-1] - 1))``
* Sweep: I1 sub-heuristic with the same rk_bias keys.

References
----------
Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling
problems with time window constraints. Operations Research, 35(2), 254-265.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import VRPTWInstance


# ── Public constants ───────────────────────────────────────────────────────────

#: All valid strategy names.
STRATEGIES = ("i1", "i2", "i3", "savings", "nn", "sweep", "random")

#: The six base (non-meta) strategies.
_BASE = ("i1", "i2", "i3", "savings", "nn", "sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Solution dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VRPTWSolution:
    """Decoded VRPTW solution."""
    routes:        list[list[int]]   # each route: [0, c1, c2, ..., 0]
    total_distance: float
    n_vehicles:    int
    route_balance: float             # d_max − d_min; 0 if only one route


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class VRPTWDecoder:
    """
    Decode a BRKGA chromosome into a :class:`VRPTWSolution` using one of
    Solomon's constructive heuristics.

    Parameters
    ----------
    instance:
        Parsed :class:`~allodyn.vrptw.data.VRPTWInstance`.
    strategy:
        One of ``"i1"``, ``"i2"``, ``"i3"``, ``"savings"``, ``"nn"``,
        ``"sweep"``, ``"random"``.  Default ``"i1"``.
    """

    def __init__(self, instance: VRPTWInstance, strategy: str = "i1") -> None:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}."
            )
        self.instance = instance
        self.strategy = strategy

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def decode(self, chromosome: np.ndarray) -> VRPTWSolution:
        """
        Decode *chromosome* into a feasible VRPTW solution.

        Parameters
        ----------
        chromosome:
            1-D array of length ``N + 4`` with values ∈ ``[0, 1]``.

        Returns
        -------
        VRPTWSolution
        """
        routes = self._build_routes(chromosome)
        return self._to_solution(routes)

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _build_routes(self, chromosome: np.ndarray) -> list[list[int]]:
        strategy = self.strategy
        if strategy == "random":
            # Deterministic but opaque to the GA: bitwise XOR hash of quantised
            # chromosome.  Different individuals land on different decoders even
            # when their chromosomes differ only slightly.
            bits = np.round(chromosome * 255).astype(np.uint8)
            key  = int(np.bitwise_xor.reduce(bits)) % len(_BASE)
            strategy = _BASE[key]

        _dispatch = {
            "i1":      self._decode_i1,
            "i2":      self._decode_i2,
            "i3":      self._decode_i3,
            "savings": self._decode_savings,
            "nn":      self._decode_nn,
            "sweep":   self._decode_sweep,
        }
        return _dispatch[strategy](chromosome)

    def _to_solution(self, routes: list[list[int]]) -> VRPTWSolution:
        D = self.instance.dist
        dists = [
            sum(D[r[k], r[k + 1]] for k in range(len(r) - 1))
            for r in routes
        ]
        total   = float(sum(dists))
        n_v     = len(routes)
        balance = float(max(dists) - min(dists)) if n_v > 1 else 0.0
        return VRPTWSolution(routes, total, n_v, balance)

    # ──────────────────────────────────────────────────────────────────────────
    # I1
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_i1(self, chromosome: np.ndarray) -> list[list[int]]:
        """
        Chromosome parameters
        ---------------------
        rk[0] → mu ∈ [0, 2]   distance-savings multiplier ∈ c11
        rk[1] → lambda ∈ [0, 2]   depot-distance weight ∈ c2
        rk[2], rk[3] → alpha_1, alpha_2  (normalised, alpha_1 + alpha_2=1)
        rk[4+] → per-customer bias for c2 scoring (maximise)
        """
        inst = self.instance
        D    = inst.dist
        mu, lam, alpha1, alpha2 = self._params_i1(chromosome)
        rk_bias = chromosome[4:]

        unrouted: set[int] = set(range(1, inst.n_customers + 1))
        routes: list[list[int]] = []

        while unrouted:
            seed = min(unrouted, key=lambda c: inst.nodes[c].due_date)
            unrouted.remove(seed)
            route = [0, seed, 0]
            b = self._compute_times(route)

            while True:
                best_u = best_pos = None
                best_c2p = -np.inf
                for u in unrouted:
                    c1, pos = self._best_insertion(route, b, u, mu, alpha1, alpha2, D)
                    if c1 is None:
                        continue
                    c2p = (lam * D[0, u] - c1) * rk_bias[u - 1]
                    if c2p > best_c2p:
                        best_c2p, best_u, best_pos = c2p, u, pos
                if best_u is None:
                    break
                route.insert(best_pos, best_u) # type: ignore
                unrouted.remove(best_u)
                b = self._compute_times(route)

            routes.append(route)

        return routes

    # ──────────────────────────────────────────────────────────────────────────
    # I2
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_i2(self, chromosome: np.ndarray) -> list[list[int]]:
        """
        c1 is identical to I1.  c2 minimises a weighted combination of the
        resulting route distance and route duration:

            c2(u) = beta_1 * R_d(u) + beta_2 * R_t(u)

        where R_d(u) is the total route distance and R_t(u) is the return-to-
        depot time after inserting u at its best c1 position.

        Chromosome parameters
        ---------------------
        rk[0] → mu ∈ [0, 2]
        rk[1], rk[2] → alpha_1, alpha_2  (normalised)
        rk[3] → beta_1 ∈ [0, 1]    (beta_2 = 1 - beta_1)
        rk[4+] → per-customer bias  (multiplies c2; higher bias → smaller
                  effective c2 → selected sooner in a minimise sense)
        """
        inst    = self.instance
        D       = inst.dist
        mu      = float(chromosome[0]) * 2.0
        a_sum   = float(chromosome[1]) + float(chromosome[2])
        alpha1  = float(chromosome[1]) / a_sum if a_sum > 1e-9 else 0.5
        alpha2  = 1.0 - alpha1
        beta1   = float(chromosome[3])
        beta2   = 1.0 - beta1
        rk_bias = chromosome[4:]

        unrouted: set[int] = set(range(1, inst.n_customers + 1))
        routes:   list[list[int]] = []

        while unrouted:
            seed = min(unrouted, key=lambda c: inst.nodes[c].due_date)
            unrouted.remove(seed)
            route = [0, seed, 0]
            b = self._compute_times(route)

            while True:
                best_u = best_pos = None
                best_c2p = np.inf  # minimise

                for u in unrouted:
                    c1, pos = self._best_insertion(route, b, u, mu, alpha1, alpha2, D)
                    if c1 is None:
                        continue

                    # Route metrics after tentative insertion
                    tentative = route[:pos] + [u] + route[pos:]
                    t_dist = sum(
                        D[tentative[k], tentative[k + 1]]
                        for k in range(len(tentative) - 1)
                    )
                    t_b    = self._compute_times(tentative)
                    t_time = t_b[-1]  # service-start at return depot

                    c2  = beta1 * t_dist + beta2 * t_time
                    # Bias: higher bias → smaller effective c2 → preferred
                    c2p = c2 * (1.0 + 0.2 * (rk_bias[u - 1] - 0.5))
                    if c2p < best_c2p:
                        best_c2p, best_u, best_pos = c2p, u, pos

                if best_u is None:
                    break
                route.insert(best_pos, best_u) # type: ignore
                unrouted.remove(best_u)
                b = self._compute_times(route)

            routes.append(route)

        return routes

    # ──────────────────────────────────────────────────────────────────────────
    # I3
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_i3(self, chromosome: np.ndarray) -> list[list[int]]:
        """
        Adds urgency to c1:

            c13(i, u, j) = l_u - b_u            (time slack at u after insertion)
            c1(i, u, j)  = alpha_1 * c11 + alpha_2 * c12 + alpha_3 * c13
            c2(i, u, j)  = c1(i, u, j)          (minimise c1 to choose customer)

        Chromosome parameters
        ---------------------
        rk[0] → mu ∈ [0, 2]
        rk[1], rk[2], rk[3] → alpha_1, alpha_2, alpha_3  (normalised, sum = 1)
        rk[4+] → per-customer bias  (mild +-10 % perturbation of c1; minimise)
        """
        inst    = self.instance
        D       = inst.dist
        mu      = float(chromosome[0]) * 2.0
        a_raw   = [float(chromosome[k]) for k in (1, 2, 3)]
        a_sum   = sum(a_raw)
        if a_sum < 1e-9:
            alpha1 = alpha2 = alpha3 = 1.0 / 3
        else:
            alpha1, alpha2, alpha3 = [a / a_sum for a in a_raw]
        rk_bias = chromosome[4:]

        unrouted: set[int] = set(range(1, inst.n_customers + 1))
        routes:   list[list[int]] = []

        while unrouted:
            seed = min(unrouted, key=lambda c: inst.nodes[c].due_date)
            unrouted.remove(seed)
            route = [0, seed, 0]
            b = self._compute_times(route)

            while True:
                best_u = best_pos = None
                best_c1p = np.inf  # minimise

                for u in unrouted:
                    c1, pos = self._best_insertion_i3(
                        route, b, u, mu, alpha1, alpha2, alpha3, D
                    )
                    if c1 is None:
                        continue
                    c1p = c1 * (1.0 + 0.1 * (rk_bias[u - 1] - 0.5))
                    if c1p < best_c1p:
                        best_c1p, best_u, best_pos = c1p, u, pos

                if best_u is None:
                    break
                route.insert(best_pos, best_u) # type: ignore
                unrouted.remove(best_u)
                b = self._compute_times(route)

            routes.append(route)

        return routes

    # ──────────────────────────────────────────────────────────────────────────
    # Savings (parallel Clarke-Wright)
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_savings(self, chromosome: np.ndarray) -> list[list[int]]:
        """
        Parallel savings heuristic (Clarke & Wright 1964, constrained variant
        from Solomon 1987).

        Savings formula:  s(i, j) = d(i, 0) + d(0, j) - mu * d(i, j)

        Merging link from the tail of one route (customer i) to the head of
        another (customer j) is accepted when:

        * capacity is not exceeded;
        * all time windows remain feasible after the merge;
        * if the waiting-time limit is active: the waiting time at j in the
          merged route does not exceed W.

        Per-customer bias perturbs the savings:

            s_eff(i, j) = s(i, j) * (1 + alpha * (rk_bias[i-1] + rk_bias[j-1] - 1))

        so different chromosomes sort the savings list differently.

        Chromosome parameters
        ---------------------
        rk[0] → mu ∈ [0, 2]
        rk[1] → waiting-time limit active when > 0.5
        rk[2] → W as fraction of the time horizon ∈ [0, 1]
        rk[3] → perturbation scale alpha ∈ [0, 0.5]
        rk[4+] → per-customer bias (perturbs savings)
        """
        inst    = self.instance
        D       = inst.dist
        N       = inst.n_customers
        mu      = float(chromosome[0]) * 2.0
        use_W   = float(chromosome[1]) > 0.5
        W_frac  = float(chromosome[2])
        sigma   = float(chromosome[3]) * 0.5
        rk_bias = chromosome[4:]

        horizon = max(inst.nodes[c].due_date for c in range(1, N + 1))
        W       = W_frac * horizon if use_W else float("inf")

        # Initialise: one singleton route per customer
        route_list: dict[int, list[int]] = {c: [0, c, 0] for c in range(1, N + 1)}
        route_of:   dict[int, int]       = {c: c         for c in range(1, N + 1)}

        # Compute perturbed savings for all (i, j) tail→head pairs
        savings: list[tuple[float, int, int]] = []
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i == j:
                    continue
                s     = D[i, 0] + D[0, j] - mu * D[i, j]
                s_eff = s * (1.0 + sigma * (rk_bias[i - 1] + rk_bias[j - 1] - 1.0))
                savings.append((s_eff, i, j))
        savings.sort(key=lambda t: -t[0])

        for _, i, j in savings:
            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = route_list[ri]
            route_j = route_list[rj]

            # i must be the tail customer, j the head customer
            if route_i[-2] != i or route_j[1] != j:
                continue

            # Capacity feasibility
            load_i = sum(inst.nodes[c].demand for c in route_i if c != 0)
            load_j = sum(inst.nodes[c].demand for c in route_j if c != 0)
            if load_i + load_j > inst.capacity:
                continue

            # Merge and check time feasibility
            merged = route_i[:-1] + route_j[1:]
            b      = self._compute_times(merged)
            if any(b[k] > inst.nodes[merged[k]].due_date for k in range(len(merged))):
                continue

            # Optional waiting-time limit at j (first customer of route_j)
            if use_W:
                j_pos    = len(route_i) - 1           # index of j in merged
                prev     = merged[j_pos - 1]
                arrival  = b[j_pos - 1] + inst.nodes[prev].service_time + D[prev, j]
                w_new    = b[j_pos] - arrival          # waiting time at j
                if w_new > W:
                    continue

            # Accept merge
            route_list[ri] = merged
            del route_list[rj]
            for c in route_j:
                if c != 0:
                    route_of[c] = ri

        return list(route_list.values())

    # ──────────────────────────────────────────────────────────────────────────
    # Nearest-neighbour (time-oriented, sequential)
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_nn(self, chromosome: np.ndarray) -> list[list[int]]:
        """
        Sequential time-oriented nearest-neighbour heuristic.

        At each step the "closest" unrouted customer to the last routed customer
        is added:

            c_ij = delta1 * d_ij + delta2 * T_ij + delta3 * v_ij

        where
            T_ij = b_j - (b_i + s_i)          time from end of service at i to
                                              start at j (includes travel + wait)
            v_ij = l_j - (b_i + s_i + t_i_j)    remaining slack at j measured from
                                              earliest possible arrival

        A smaller v_ij means the customer is more urgent.  Per-customer bias
        perturbs the metric so different chromosomes favour different customers:

            c_ij_eff = c_ij * (0.5 + rk_bias[j-1])    (minimise)

        A new route is started with the nearest unrouted customer to the depot
        when no feasible customer can be appended.

        Chromosome parameters
        ---------------------
        rk[0], rk[1], rk[2] → delta1, delta2, delta3  (normalised, sum = 1)
        rk[3] → unused
        rk[4+] → per-customer bias (perturbs c_ij)
        """
        inst    = self.instance
        D       = inst.dist
        N       = inst.n_customers
        d_raw   = [float(chromosome[k]) for k in (0, 1, 2)]
        d_sum   = sum(d_raw)
        if d_sum < 1e-9:
            delta1 = delta2 = delta3 = 1.0 / 3
        else:
            delta1, delta2, delta3 = [d / d_sum for d in d_raw]
        rk_bias = chromosome[4:]

        unrouted: set[int] = set(range(1, N + 1))
        routes:   list[list[int]] = []

        while unrouted:
            # Seed: NN metric from depot (b_depot = 0, s_depot = 0)
            seed      = None
            best_seed = np.inf
            for c in unrouted:
                b_c = max(inst.nodes[c].ready_time, D[0, c])
                T   = b_c                                      # T_{0,c}
                v   = inst.nodes[c].due_date - D[0, c]        # v_{0,c}
                if v < 0:
                    continue
                metric = (delta1 * D[0, c] + delta2 * T + delta3 * v) * (0.5 + rk_bias[c - 1])
                if metric < best_seed:
                    best_seed, seed = metric, c

            if seed is None:
                # Fallback: earliest deadline
                seed = min(unrouted, key=lambda c: inst.nodes[c].due_date)

            unrouted.remove(seed)
            route = [0, seed, 0]
            b     = self._compute_times(route)

            while True:
                last   = route[-2]
                b_last = b[-2]
                s_last = inst.nodes[last].service_time

                best_j    = None
                best_cij  = np.inf

                for j in unrouted:
                    # Capacity
                    load = sum(inst.nodes[c].demand for c in route if c != 0)
                    if load + inst.nodes[j].demand > inst.capacity:
                        continue

                    # Time at j if appended after last
                    arrival_j = b_last + s_last + D[last, j]
                    b_j       = max(inst.nodes[j].ready_time, arrival_j)
                    if b_j > inst.nodes[j].due_date:
                        continue

                    # Depot feasibility after j
                    if b_j + inst.nodes[j].service_time + D[j, 0] > inst.nodes[0].due_date:
                        continue

                    T_ij  = b_j - (b_last + s_last)
                    v_ij  = inst.nodes[j].due_date - arrival_j

                    c_ij  = delta1 * D[last, j] + delta2 * T_ij + delta3 * v_ij
                    c_eff = c_ij * (0.5 + rk_bias[j - 1])

                    if c_eff < best_cij:
                        best_cij, best_j = c_eff, j

                if best_j is None:
                    break

                # Append before depot
                route.insert(-1, best_j)
                unrouted.remove(best_j)
                b = self._compute_times(route)

            routes.append(route)

        return routes

    # ──────────────────────────────────────────────────────────────────────────
    # Sweep (cluster-first, route-second)
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_sweep(self, chromosome: np.ndarray) -> list[list[int]]:
        """
        Two-phase sweep heuristic.

        Phase 1 — Clustering
            Customers are sorted by polar angle (counterclockwise from the depot).
            The sweep begins at angle ``start_angle`` and assigns consecutive
            customers to a vehicle until its capacity is full, then starts a new
            cluster.

        Phase 2 — Scheduling
            Each cluster is scheduled with the I1 sub-heuristic (using mu, lamda,
            alpha from the chromosome).  Customers that cannot be feasibly
            inserted (time windows) are deferred to a leftover pool, which is
            then scheduled with additional I1 routes.

        Chromosome parameters
        ---------------------
        rk[0] → start_angle ∈ [0, 2 * pi)   sweep origin angle
        rk[1] → mu ∈ [0, 2]              I1 sub-heuristic parameter
        rk[2] → lambda ∈ [0, 2]              I1 sub-heuristic parameter
        rk[3] → alpha_1 ∈ [0, 1]            (alpha_2 = 1 - alpha_1) I1 sub-heuristic
        rk[4+] → per-customer bias for I1 sub-heuristic c2 scoring
        """
        inst         = self.instance
        D            = inst.dist
        N            = inst.n_customers
        start_angle  = float(chromosome[0]) * 2.0 * np.pi
        mu           = float(chromosome[1]) * 2.0
        lam          = float(chromosome[2]) * 2.0
        alpha1       = float(chromosome[3])
        alpha2       = 1.0 - alpha1
        rk_bias      = chromosome[4:]

        # Polar angle relative to depot, offset by start_angle
        depot = inst.nodes[0]
        def polar(c: int) -> float:
            nd = inst.nodes[c]
            a  = np.arctan2(nd.y - depot.y, nd.x - depot.x)
            return float((a - start_angle) % (2.0 * np.pi))

        sweep_order = sorted(range(1, N + 1), key=polar)

        # Phase 1: capacity-limited clusters in angular order
        clusters: list[list[int]] = []
        cluster:  list[int]       = []
        load      = 0
        for c in sweep_order:
            d = inst.nodes[c].demand
            if load + d <= inst.capacity:
                cluster.append(c)
                load += d
            else:
                if cluster:
                    clusters.append(cluster)
                cluster = [c]
                load    = d
        if cluster:
            clusters.append(cluster)

        # Phase 2: I1 on each cluster; collect time-infeasible leftovers
        routes:   list[list[int]] = []
        leftover: list[int]       = []

        for cluster_cust in clusters:
            remaining = set(cluster_cust)
            seed = min(remaining, key=lambda c: inst.nodes[c].due_date)
            remaining.remove(seed)
            route = [0, seed, 0]
            b     = self._compute_times(route)

            while True:
                best_u = best_pos = None
                best_c2p = -np.inf
                for u in remaining:
                    c1, pos = self._best_insertion(route, b, u, mu, alpha1, alpha2, D)
                    if c1 is None:
                        continue
                    c2p = (lam * D[0, u] - c1) * rk_bias[u - 1]
                    if c2p > best_c2p:
                        best_c2p, best_u, best_pos = c2p, u, pos
                if best_u is None:
                    break
                route.insert(best_pos, best_u) # type: ignore
                remaining.remove(best_u)
                b = self._compute_times(route)

            routes.append(route)
            leftover.extend(remaining)  # time-infeasible customers

        # Schedule leftover with standard I1
        leftover_set: set[int] = set(leftover)
        while leftover_set:
            seed = min(leftover_set, key=lambda c: inst.nodes[c].due_date)
            leftover_set.remove(seed)
            route = [0, seed, 0]
            b     = self._compute_times(route)

            while True:
                best_u = best_pos = None
                best_c2p = -np.inf
                for u in leftover_set:
                    c1, pos = self._best_insertion(route, b, u, mu, alpha1, alpha2, D)
                    if c1 is None:
                        continue
                    c2p = (lam * D[0, u] - c1) * rk_bias[u - 1]
                    if c2p > best_c2p:
                        best_c2p, best_u, best_pos = c2p, u, pos
                if best_u is None:
                    break
                route.insert(best_pos, best_u) # type: ignore
                leftover_set.remove(best_u)
                b = self._compute_times(route)

            routes.append(route)

        return routes

    # ──────────────────────────────────────────────────────────────────────────
    # Shared utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _params_i1(
        self, chromosome: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Extract (mu, lam, alpha1, alpha2) from the first four chromosome keys."""
        mu  = float(chromosome[0]) * 2.0
        lam = float(chromosome[1]) * 2.0
        a_sum = float(chromosome[2]) + float(chromosome[3])
        if a_sum < 1e-9:
            alpha1 = alpha2 = 0.5
        else:
            alpha1 = float(chromosome[2]) / a_sum
            alpha2 = float(chromosome[3]) / a_sum
        return mu, lam, alpha1, alpha2

    def _compute_times(self, route: list[int]) -> list[float]:
        """
        Compute service-start times ``b[k]`` for every node in *route*.

        b[k] = max(ready_time[k], b[k-1] + service_time[k-1] + d(k-1, k))
        """
        inst = self.instance
        D    = inst.dist
        b: list[float] = [float(inst.nodes[route[0]].ready_time)]
        for k in range(1, len(route)):
            prev, curr = route[k - 1], route[k]
            arrival    = b[k - 1] + inst.nodes[prev].service_time + D[prev, curr]
            b.append(max(float(inst.nodes[curr].ready_time), arrival))
        return b

    def _best_insertion(
        self,
        route:  list[int],
        b:      list[float],
        u:      int,
        mu:     float,
        alpha1: float,
        alpha2: float,
        D:      np.ndarray,
    ) -> tuple[float | None, int | None]:
        """
        Find the cheapest feasible I1/I2 c1 insertion position for customer *u*.

        Returns ``(c1_min, position)`` or ``(None, None)`` if infeasible.
        *position* is the index before which *u* is inserted.
        """
        inst   = self.instance
        node_u = inst.nodes[u]

        load = sum(inst.nodes[c].demand for c in route if c != 0)
        if load + node_u.demand > inst.capacity:
            return None, None

        best_c1: float | None = None
        best_pos: int | None  = None

        for p in range(1, len(route)):
            i, j = route[p - 1], route[p]

            arrival_u = b[p - 1] + inst.nodes[i].service_time + D[i, u]
            b_u       = max(float(node_u.ready_time), arrival_u)
            if b_u > node_u.due_date:
                continue

            arrival_j_new = b_u + node_u.service_time + D[u, j]
            b_j_new       = max(float(inst.nodes[j].ready_time), arrival_j_new)

            if not self._push_forward_feasible(route, b, p, b_j_new):
                continue

            c11 = D[i, u] + D[u, j] - mu * D[i, j]
            c12 = b_j_new - b[p]
            c1  = alpha1 * c11 + alpha2 * c12

            if best_c1 is None or c1 < best_c1:
                best_c1, best_pos = c1, p

        return best_c1, best_pos

    def _best_insertion_i3(
        self,
        route:  list[int],
        b:      list[float],
        u:      int,
        mu:     float,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        D:      np.ndarray,
    ) -> tuple[float | None, int | None]:
        """
        I3 variant of :meth:`_best_insertion`.

        Adds the urgency term: c13(i, u, j) = l_u - b_u  (slack at u).
        c1 = alpha_1·c11 + alpha_2·c12 + alpha_3·c13.
        """
        inst   = self.instance
        node_u = inst.nodes[u]

        load = sum(inst.nodes[c].demand for c in route if c != 0)
        if load + node_u.demand > inst.capacity:
            return None, None

        best_c1: float | None = None
        best_pos: int | None  = None

        for p in range(1, len(route)):
            i, j = route[p - 1], route[p]

            arrival_u = b[p - 1] + inst.nodes[i].service_time + D[i, u]
            b_u       = max(float(node_u.ready_time), arrival_u)
            if b_u > node_u.due_date:
                continue

            arrival_j_new = b_u + node_u.service_time + D[u, j]
            b_j_new       = max(float(inst.nodes[j].ready_time), arrival_j_new)

            if not self._push_forward_feasible(route, b, p, b_j_new):
                continue

            c11 = D[i, u] + D[u, j] - mu * D[i, j]
            c12 = b_j_new - b[p]
            c13 = float(node_u.due_date) - b_u          # urgency: slack at u
            c1  = alpha1 * c11 + alpha2 * c12 + alpha3 * c13

            if best_c1 is None or c1 < best_c1:
                best_c1, best_pos = c1, p

        return best_c1, best_pos

    def _push_forward_feasible(
        self,
        route:      list[int],
        b:          list[float],
        insert_pos: int,
        b_j_new:    float,
    ) -> bool:
        """
        Check that inserting a customer before ``route[insert_pos]`` does not
        push any subsequent node past its due date.

        Uses the standard Solomon push-forward propagation rule:

            push(k) = max(0, push(k-1) - wait(k))

        The push decreases monotonically and terminates early once absorbed by
        waiting time.
        """
        inst = self.instance
        D    = inst.dist
        push = b_j_new - b[insert_pos]

        if push <= 1e-9:
            return True

        for k in range(insert_pos, len(route)):
            if k == insert_pos:
                new_b_k = b_j_new
            else:
                prev    = route[k - 1]
                curr    = route[k]
                arrival = b[k - 1] + inst.nodes[prev].service_time + D[prev, curr]
                wait_k  = b[k] - arrival
                push    = max(0.0, push - wait_k)
                if push < 1e-9:
                    return True
                new_b_k = b[k] + push

            if new_b_k > inst.nodes[route[k]].due_date:
                return False

        return True
