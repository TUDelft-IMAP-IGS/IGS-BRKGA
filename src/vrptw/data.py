"""
Solomon VRPTW instance loader and theoretical bound computation.

Parses the standard Solomon benchmark format:

    INSTANCE_NAME

    VEHICLE
    NUMBER     CAPACITY
      25         200

    CUSTOMER
    CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
        0      40        50        0        0          1236        0
        1      45        68       10      912          967        90
        ...

Node 0 is the depot; nodes 1..N are customers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Theoretical bound helpers
# ---------------------------------------------------------------------------

def _mst_weight(dist: np.ndarray) -> float:
    """
    Prim's algorithm — O(n²) MST weight on the full node set.

    Parameters
    ----------
    dist:
        Square distance matrix of shape ``(n, n)``.

    Returns
    -------
    float
        Sum of the ``n-1`` MST edge weights.
    """
    n = dist.shape[0]
    in_mst  = np.zeros(n, dtype=bool)
    key     = np.full(n, np.inf)
    key[0]  = 0.0
    total   = 0.0
    for _ in range(n):
        # cheapest node not yet in MST
        tmp = np.where(~in_mst, key, np.inf)
        u   = int(np.argmin(tmp))
        in_mst[u] = True
        total += key[u]
        # relax edges from u
        mask       = ~in_mst
        better     = dist[u] < key
        key[mask & better] = dist[u][mask & better]
    return float(total)


def compute_theoretical_bounds(instance: "VRPTWInstance") -> dict[str, tuple[float, float]]:
    """
    Compute analytical lower and upper bounds for the three VRPTW objectives
    without evaluating any solutions.

    Bounds
    ------
    distance
        lo — MST weight on ``{depot} U {customers}`` (Prim's, O(n²)).
             Any set of routes covering every customer must form a connected
             spanning subgraph, which costs at least the MST.
        hi — ``2 * Σ d(depot, c)`` for all customers.
             Sending one vehicle per customer is always feasible for Solomon
             instances (single-customer routes satisfy capacity and TW), so
             this sum is a reachable upper bound.

    n_vehicles
        lo — ``⌈Σ demand / capacity⌉``.
             Classic bin-packing lower bound; no routing can beat it.
        hi — ``n_customers``.
             Absolute worst case: one vehicle per customer.

    route_balance  (d_max − d_min across routes)
        lo — ``0``.
             Theoretically achievable when all routes have equal length.
        hi — ``2 × (max d(depot, c) − min d(depot, c))``.
             In the worst case one route serves the farthest customer alone
             and another serves the nearest alone; their length difference
             is bounded by the spread of solo-trip distances.

    Parameters
    ----------
    instance:
        Parsed :class:`VRPTWInstance` with a precomputed distance matrix.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``{"distance": (lo, hi), "n_vehicles": (lo, hi), "route_balance": (lo, hi)}``
    """
    D  = instance.dist
    nc = instance.n_customers

    # ── distance ──────────────────────────────────────────────────────────────
    lo_dist = _mst_weight(D)                          # MST over all n+1 nodes
    hi_dist = 2.0 * float(np.sum(D[0, 1:nc + 1]))    # each customer served alone

    # ── n_vehicles ────────────────────────────────────────────────────────────
    total_demand = sum(c.demand for c in instance.customers)
    lo_veh = float(max(1, math.ceil(total_demand / instance.capacity)))
    hi_veh = float(nc)

    # ── route_balance ─────────────────────────────────────────────────────────
    # route_balance = d_max − d_min  ≤  d_max  ≤  max single-route distance.
    #
    # Upper bound on d_max: a route visits at most n_max customers (capacity).
    # By the triangle inequality, any route visiting customers c_1 … c_m has
    #   d(route) ≤ 2 × Σ d(depot, c_i)
    # The worst case is the n_max farthest customers from the depot.
    demands = np.array([c.demand for c in instance.customers])
    min_demand = float(demands[demands > 0].min()) if (demands > 0).any() else 1.0
    n_max_per_route = max(1, int(instance.capacity // min_demand))
    depot_dists = D[0, 1:nc + 1]
    farthest_n  = np.sort(depot_dists)[-n_max_per_route:]   # top-n_max distances
    hi_max_route = 2.0 * float(farthest_n.sum())

    lo_bal = 0.0
    # d_min ≥ 0, so route_balance ≤ d_max ≤ hi_max_route
    hi_bal = hi_max_route

    return {
        "distance":      (lo_dist, hi_dist),
        "n_vehicles":    (lo_veh,  hi_veh),
        "route_balance": (lo_bal,  hi_bal),
    }


@dataclass
class Node:
    """A single node (depot or customer) in a Solomon instance."""
    idx: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class VRPTWInstance:
    """
    A parsed Solomon VRPTW instance.

    Attributes
    ----------
    name:
        Instance identifier (e.g. ``"C101"``).
    n_vehicles:
        Maximum number of vehicles allowed (informational; I1 ignores this limit
        and uses as many routes as needed — minimising vehicle count is an objective).
    capacity:
        Vehicle load capacity.
    nodes:
        All nodes; index 0 = depot, indices 1..N = customers.
    dist:
        Precomputed Euclidean distance matrix, shape ``(N+1, N+1)``.
    """
    name: str
    n_vehicles: int
    capacity: float
    nodes: list[Node]
    dist: np.ndarray

    @property
    def depot(self) -> Node:
        return self.nodes[0]

    @property
    def customers(self) -> list[Node]:
        return self.nodes[1:]

    @property
    def n_customers(self) -> int:
        return len(self.nodes) - 1


def parse_solomon(path: str | Path) -> VRPTWInstance:
    """
    Parse a Solomon-format ``.txt`` file and return a :class:`VRPTWInstance`.

    Parameters
    ----------
    path:
        Path to the ``.txt`` file.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    name = lines[0].strip()

    # ── Vehicle capacity ───────────────────────────────────────────────────────
    n_vehicles = capacity = None
    for i, line in enumerate(lines):
        if line.strip().startswith("NUMBER"):
            parts = lines[i + 1].split()
            n_vehicles = int(parts[0])
            capacity = float(parts[1])
            break
    assert n_vehicles is not None, f"Could not parse VEHICLE section in {path}"

    # ── Customer data ──────────────────────────────────────────────────────────
    nodes: list[Node] = []
    reading = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CUST"):
            reading = True
            continue
        if not reading:
            continue
        parts = stripped.split()
        if len(parts) >= 7 and parts[0].lstrip("-").isdigit():
            nodes.append(Node(
                idx=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=float(parts[3]),
                ready_time=float(parts[4]),
                due_date=float(parts[5]),
                service_time=float(parts[6]),
            ))

    assert nodes, f"No customer data found in {path}"

    # ── Distance matrix (Euclidean) ────────────────────────────────────────────
    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = nodes[i].x - nodes[j].x
            dy = nodes[i].y - nodes[j].y
            dist[i, j] = math.sqrt(dx * dx + dy * dy)

    return VRPTWInstance(
        name=name,
        n_vehicles=n_vehicles,
        capacity=capacity, # type: ignore
        nodes=nodes,
        dist=dist,
    )
