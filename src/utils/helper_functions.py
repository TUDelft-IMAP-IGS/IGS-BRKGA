from typing import Optional

import numpy as np
from collections import defaultdict
from loguru import logger

from components.ga_types import Chromosome, DecodedSolution, FitnessVector
from logic.scheduling_model import SchedulingModel

def post_process_current_results(
    model,
    sorted_assignments_per_vessel,
    start_times,
    locations,
    idle_discount: float,
    speed_indices: Optional[np.ndarray] = None,
    assigned_vessels: Optional[np.ndarray] = None,
):
        """
        Post-process a single solution's results to compute sorted assignments and performance metrics.

        For the specified solution, extracts and sorts assigned activities per vessel,
        determines location configurations, calculates performance metrics (distance,
        duration, cost, fuel, speed), and aggregates totals.

        Parameters:
        idx_soln (int): Index of the solution being processed.
        case_assignment (list[list[int]]): Assignment matrix for vessels and activities.
        case_start_times (list[int]): Start times for activities.
        case_location (list[int]): Location assignments for activities.

        Returns:
            results_totals (dict): Aggregated totals across all vessels.
        """
        total_distance = 0
        total_cost = 0
        total_fuel = 0

        for idx_v, _ in enumerate(model.vessels):
            sorted_activities = sorted_assignments_per_vessel[idx_v]
            results_totals = calculate_performance_metrics(
                model=model,
                sorted_activities=sorted_activities,
                idx_v=idx_v,
                idle_discount=idle_discount,
                start_times=start_times,
                locations=locations,
                speed_indices=speed_indices,
                assigned_vessels=assigned_vessels,
            )

            total_distance += results_totals["distance"]
            total_cost += results_totals["cost"]
            total_fuel += results_totals["fuel"]

        # Calculate makespan
        if start_times and len(start_times) > 0:
            end_times = [
                start_times[i] + get_total_duration(model.activities[i])
                for i in range(len(start_times))
            ]
            total_duration = float(max(end_times) - min(start_times))
        else:
            total_duration = 0.0

        results_totals = {
            "distance": total_distance,
            "duration": total_duration,
            "cost": total_cost,
            "fuel": total_fuel,
        }
        return results_totals

def _role_for_activity_vessel(model, act_idx: int, v_idx: int, assigned_vessels: Optional[np.ndarray]) -> int:
    """Return role_id for (activity, vessel) if assigned; else fallback to first role of activity; else -1."""
    roles = [r for r in model.roles if r.parent_activity_idx == act_idx]
    if not roles:
        return -1

    if assigned_vessels is not None:
        for r in roles:
            if r.role_id < len(assigned_vessels) and int(assigned_vessels[r.role_id]) == v_idx:
                return r.role_id

    return roles[0].role_id

def calculate_performance_metrics(
    model,
    sorted_activities: list,
    idx_v: int,
    start_times: list,
    locations: list,
    idle_discount: float,
    speed_indices: Optional[np.ndarray] = None,
    assigned_vessels: Optional[np.ndarray] = None,
):
        """
        Calculate detailed performance metrics for a vessel's assigned activities.

        Evaluates distances, durations, costs, fuel consumption, and speeds based on
        the scheduled activities for a given vessel in a specific solution. Considers
        possible location configurations and selects the optimal one.

        Parameters:
        sorted_activities (list[int]): Activities assigned to the vessel, sorted by start time.
        idx_v (int): Index of the vessel.
        idx_soln (int): Index of the current solution.

        Returns:
            results_totals (dict): Totals of the above metrics for the vessel.
        """
        distance_total = 0
        cost_total = 0
        fuel_consumed_total = 0

        vessel = model.vessels[idx_v]

        if len(sorted_activities) > 1:
            dist_matrix_for_v = model.vessel_travel_dist_matrix_float[idx_v]
            idx_a_prev = sorted_activities[0]
            prev_activity = model.activities[idx_a_prev]

            for k in range(1, len(sorted_activities)):
                idx_a_current = sorted_activities[k]
                current_activity = model.activities[idx_a_current]

                prev_total_duration = get_total_duration(prev_activity)
                available_time = start_times[idx_a_current] - (start_times[idx_a_prev] + prev_total_duration)

                prev_loc_val = locations[idx_a_prev]
                end_loc_id = model._get_global_loc_id(prev_activity, prev_loc_val, is_start_loc=False)

                current_loc_val = locations[idx_a_current]
                start_loc_id = model._get_global_loc_id(current_activity, current_loc_val, is_start_loc=True)

                distance = dist_matrix_for_v[end_loc_id, start_loc_id]
                distance_total += distance

                idx_speed = None
                if speed_indices is not None:
                    role_id = _role_for_activity_vessel(model, idx_a_current, idx_v, assigned_vessels)
                    if role_id >= 0 and role_id < len(speed_indices):
                        try:
                            idx_speed = int(speed_indices[role_id])
                        except Exception:
                            idx_speed = None

                valid_speeds = vessel.possible_speeds
                if idx_speed is None:
                    # legacy fallback: infer from available gap
                    if available_time <= 0:
                        min_speed = vessel.sailing_speed_max
                    else:
                        min_speed = distance / (available_time * 24.0 + 1e-6)

                    if len(np.argwhere(valid_speeds >= min_speed - 1e-6)) == 0:
                        idx_speed = int(np.argmax(valid_speeds))
                    else:
                        idx_speed = int(np.argwhere(valid_speeds >= min_speed - 1e-6)[0][0])

                idx_speed = max(0, min(int(idx_speed), len(valid_speeds) - 1))
                speed = float(valid_speeds[idx_speed])

                # Sailing days used for billing/fuel with decoded speed
                if speed <= 0 or distance <= 0:
                    sailing_days = 0
                else:
                    sailing_days = int(np.ceil(distance / (speed * 24.0)))

                # Time gap between end(prev) and start(curr): delta_{r,r'}
                gap_days = int(max(0, available_time))
                idle_days = int(max(0, gap_days - sailing_days))

                fuel_rate = float(vessel.corresponding_fuel[idx_speed])
                fuel_amt = float(np.ceil(fuel_rate * float(sailing_days)))
                # gamma = c_v*tau + p_fuel*phi + alpha*c_v*(delta - tau)
                cost = (
                    float(vessel.day_rate_mob) * float(sailing_days)
                    + float(vessel.default_fuel_price) * fuel_amt
                    + idle_discount * float(vessel.day_rate_mob) * float(idle_days)
                )

                fuel_consumed_total += fuel_amt
                cost_total += cost

                idx_a_prev = idx_a_current
                prev_activity = current_activity

        return {
            "distance": distance_total,
            "cost": cost_total,
            "fuel": fuel_consumed_total,
        }

def get_total_duration(activity) -> int:
    """
    Get the total duration of an activity including both fixed and optional components.
    
    Total duration = duration_fixed + duration_option
    
    For tow activities:  duration = duration_fixed + duration_option
    For maintenance/current location: duration = duration (single value)
    """
    return activity.duration if activity.duration else 0

def build_chromosome(X: np.ndarray, id: int, model: SchedulingModel, F: np.ndarray) -> Chromosome:
    """
    Build a chromosome from the decision vector of the NSGAII class.
    """
    num_roles = model.num_roles
    num_acts = model.num_activities

    assigned_vessels = X[id][0:num_roles]
    start_times = X[id][num_roles:num_roles + num_acts]
    locations = X[id][num_roles + num_acts : num_roles + 2 * num_acts]
    speeds = X[id][num_roles + 2 * num_acts : 2 * num_roles + 2 * num_acts]

    cost, distance, duration, fuel = F[id]

    return Chromosome(
        id=0,
        fitness=FitnessVector(
            cost=cost,
            distance=distance,
            duration=duration,
            fuel=fuel,
            constraint_penalty=0,
            result=(cost, distance, duration, fuel)
        ),
        decoded_solution=DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=start_times,
            locations=locations,
            speed_indices=speeds,
            vessel_sequences=[[0]],
            is_next_matrix=None
        ),
        role_priority_keys=None
    )

def calculate_proxy_bounds(model: SchedulingModel) -> dict[str, int]:
    """
    Calculates theoretical 'worst-case' upper bounds for all objectives 
    using logic heuristics, executing instantly without the CP solver.
    """
    bounds = {}

    # 1. Proxy Max Distance
    max_single_trip = np.max(model.vessel_travel_dist_matrix_float)
    # Worst case: Max distance traveled between every single role
    proxy_max_dist = int(np.ceil(max_single_trip * model.num_roles))
    bounds['distance'] = max(1, proxy_max_dist) # Prevent division by zero

    # 2. Proxy Max Duration
    earliest_start = min([a.possible_start[0] for a in model.activities])
    latest_end = max([a.possible_start[-1] + a.duration for a in model.activities])
    
    # Add worst-case travel time (calculated using raw floats to avoid cpmpy Maximum overloads)
    max_travel_time = 0
    if model.vessels:
        # Find the absolute slowest speed any vessel can travel at
        global_min_speed = min([float(np.min(speeds)) for speeds in model.vessel_speed_values])
        if global_min_speed > 0:
            max_travel_time = int(np.ceil(max_single_trip / (global_min_speed * 24.0)))
    
    proxy_max_duration = int((latest_end - earliest_start) + (max_travel_time * model.num_roles))
    bounds['duration'] = max(1, proxy_max_duration)

    # 3. Proxy Max Fuel & Cost
    max_fuel_rate = max([v.fuel_consumed_max for v in model.vessels])
    max_day_rate = max([v.day_rate_mob for v in model.vessels])
    max_fuel_price = max([v.default_fuel_price for v in model.vessels])

    # Worst case: entire proxy duration is spent sailing at max fuel rate with the most expensive vessel
    proxy_max_fuel = int(np.ceil(max_fuel_rate * proxy_max_duration))
    proxy_max_cost = int((max_day_rate * proxy_max_duration) + (max_fuel_price * proxy_max_fuel))
    
    bounds['fuel'] = max(1, proxy_max_fuel)
    bounds['cost'] = max(1, proxy_max_cost)

    return bounds