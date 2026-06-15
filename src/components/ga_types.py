from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass

@dataclass(frozen=False, unsafe_hash=True)
class Chromosome:
    """
    Random-key chromosome for BRKGA. 
    The interpretation of these keys depends on the Decoder strategy:
    
    1. BRKGA-STT:
       - role_priority_keys: Not heavily used (or used as secondary tie-breakers).
       - start_time_keys: Used to sort Activities by insertion priority.
       - location_keys: Maps to maintenance location selection.
       - speed_keys:
            One key per role. Decoded as vessel-specific speed index for the incoming
            transition arc of that role's activity in a vessel sequence.
            (Anchor/first activity on a vessel has no incoming arc, key is ignored there.)
    """
    id: int = field(hash=True)
    role_priority_keys: Optional[np.ndarray] = field(default=None, hash=False)
    start_time_keys: Optional[np.ndarray] = field(default=None, hash=False)
    location_keys: Optional[np.ndarray] = field(default=None, hash=False)
    start_window_keys: Optional[np.ndarray] = field(default=None, hash=False)
    speed_keys: Optional[np.ndarray] = field(default=None, hash=False)
    
    fitness: Optional["FitnessVector"] = field(default=None, hash=False)
    decoded_solution: Optional["DecodedSolution"] = field(default=None, hash=False)
    _crowding_distance: float = field(default=0.0, hash=False)
    found_at_generation: Optional[int] = field(default=None, hash=False)

@dataclass
class DecodedSolution:
    """Decoded solution from chromosome."""
    assigned_vessels: np.ndarray
    start_times: np.ndarray
    locations: np.ndarray
    vessel_sequences: List[List[int]]
    is_next_matrix: Optional[np.ndarray]
    unassigned_penalty: float = 0.0
    speed_indices: Optional[np.ndarray] = None

@dataclass
class FitnessVector:
    """Multi-objective fitness values."""
    distance: float
    cost: float
    fuel: float
    duration: float
    result: tuple
    constraint_penalty: float = 0.0

@dataclass
class GenerationStats:
    """Statistics for a single generation - used for convergence tracking."""
    generation: int
    best_distance: float
    best_cost: float
    best_fuel: float
    best_duration: float
    best_penalty: float
    feasible_count: int
    population_size: int
    pareto_front_size: int = 0
    mean_distance: float = float('inf')
    std_distance: float = 0.0
    
    @property
    def feasibility_rate(self) -> float:
        return (self.feasible_count / self.population_size) * 100 if self.population_size > 0 else 0.0


@dataclass 
class GARunStats:
    """Complete statistics from a GA run."""
    ga_name: str
    total_generations: int
    total_runtime_seconds: float
    best_solution: Chromosome
    generation_stats: List[GenerationStats] = field(default_factory=list)
    
    # Final results
    final_pareto_front_size: int = 0
    found_feasible: bool = False
    best_distance: float = float('inf')
    best_cost: float = float('inf')
    best_fuel: float = float('inf')
    best_duration: float = float('inf')
    
    # Convergence milestones
    generation_first_feasible: int = -1
    generation_of_best: int = -1
    
    def get_convergence_curve(self, metric: str = "distance") -> List[float]:
        """Extract convergence curve for a specific metric."""
        if metric == "distance":
            return [s.best_distance for s in self.generation_stats]
        elif metric == "cost":
            return [s.best_cost for s in self.generation_stats]
        elif metric == "fuel":
            return [s.best_fuel for s in self.generation_stats]
        elif metric == "penalty":
            return [s.best_penalty for s in self.generation_stats]
        elif metric == "feasible_count":
            return [float(s.feasible_count) for s in self.generation_stats]
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_generation_to_threshold(self, threshold: float, metric: str = "distance") -> int:
        """Find first generation where metric drops below threshold."""
        curve = self.get_convergence_curve(metric)
        for i, val in enumerate(curve):
            if val <= threshold:
                return i
        return -1

# Type alias for generation callback
GenerationCallback = Callable[[int, GenerationStats], None]

@dataclass
class SolutionMetrics:
    """Metrics for a single solution."""
    source: str  # "CP" or "GA"
    solution_idx: int
    
    # Objective values
    distance: float
    cost: float
    fuel: float
    duration: float
    
    # Feasibility
    is_feasible: bool
    constraint_penalty: float
    
    # Assignments (for comparison)
    vessel_assignments: List[List[int]]  # Per vessel: list of activity indices
    start_times: List[int]  # Per activity
    locations: List[int]  # Per activity (for maintenance)
    
    def dominates(self, other: "SolutionMetrics") -> bool:
        """Check if this solution Pareto-dominates another."""
        if not self.is_feasible:
            return False
        if not other.is_feasible:
            return True
        
        dominated = False
        
        objectives = [
            (self.distance, other.distance),
            (self.cost, other.cost),
            (self.fuel, other.fuel),
            (self.duration, other.duration),
        ]
        
        for self_val, other_val in objectives:
            if self_val > other_val:
                return False  # Worse in at least one objective
            if self_val < other_val:
                dominated = True  # Better in at least one objective
        
        return dominated
    
    def is_same_assignment(self, other: "SolutionMetrics", tolerance: float = 0.01) -> bool:
        """Check if two solutions have the same vessel assignments."""
        if self.vessel_assignments != other.vessel_assignments:
            return False
        if self.start_times != other.start_times:
            return False
        if self.locations != other.locations:
            return False
        return True
    
    def is_similar(self, other: "SolutionMetrics", tolerance: float = 0.001) -> bool:
        """Check if two solutions are similar (within tolerance)."""
        return (
            abs(self.distance - other.distance) < self.distance * tolerance and
            abs(self.cost - other.cost) < self.cost * tolerance and
            abs(self.fuel - other.fuel) < self.fuel * tolerance and
            abs(self.duration - other.duration) < self.duration * tolerance
        )