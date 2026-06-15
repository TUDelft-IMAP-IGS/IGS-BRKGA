import time
import numpy as np
import cpmpy as cp
import json

from loguru import logger
from datetime import date
from itertools import combinations

from components.ga_types import Chromosome
from components.role import Role
from components.activity import Activity
from components.vessel import Vessel
from utils.distance_calculator import calculate_sailing_distance
from utils.validation import verify_path_logic

class SchedulingModel:
    """
    Scheduling model for vessel allocation and activity planning.

    This class initializes vessels, activities, and associated parameters,
    builds the constraint programming model, generates decision variables,
    and adds relevant constraints for scheduling.
    """
    def __init__(self, vessels: list[Vessel], 
                 activities: list[Activity], 
                 date_origin: date, 
                 objective: str = 'distance', 
                 verbose: bool = False,
                 idle_cost_discount: float = 1.0
    ) -> None:
        """
        Initialize the scheduling model with vessels, activities, and a reference date.

        Parameters:
        vessels (list): List of vessel objects involved in scheduling.
        activities (list): List of activity objects to be scheduled.
        date_origin (date): Reference start date for scheduling calculations.
        objective (str): Optimisation objective - one of: 'distance', 'cost', 'fuel', 'duration'

        Raises:
        ValueError: If objective is not one of the supported values.
        """
        # Validate objective parameter
        valid_objectives = ['distance', 'cost', 'fuel', 'duration']
        if objective not in valid_objectives:
            raise ValueError(f"Invalid objective '{objective}'. Must be one of {','.join(valid_objectives)}")
        if not (0.0 <= idle_cost_discount <= 1.0):
            raise ValueError(f"idle_cost_discount: {idle_cost_discount} must be in [0,1]")

        # incorporate model inputs
        self.vessels = vessels
        self.activities = activities
        self.objective = objective
        self.verbose = verbose
        self.idle_cost_discount = float(idle_cost_discount)

        # save number of vessels and number of activities
        self.num_vessels = len(vessels)
        self.num_activities = len(activities)

        # save origin date
        self.date_origin = date_origin

        # link vessel name to index
        self.vessel_name_to_idx = {v.vessel_name: i for i, v in enumerate(self.vessels)}

        # link activity name to index
        self.activity_name_to_idx = {a.activity_name: i for i, a in enumerate(self.activities)}

        # create roles for each activity
        self._explode_activities_to_roles()
        self.num_roles = len(self.roles)

        # possible locations for activities (specifically maintenance)
        self.number_allowed_locations = np.ones(self.num_activities)
        for i, a in enumerate(self.activities):
            if a.is_maintenance_type():
                self.number_allowed_locations[i] = len(a.allowed_locations)

        # initalize model
        self.model = cp.Model()

        # result containers
        self.assigned_vessel_results = []
        self.start_time_results = []
        self.location_results = []
        self.link_results = []
        self.objective_values = []
        self.speed_results = []
        self.ga_results = None

        # generate decision variables
        self._build_variables()

        # create travel matrix
        self._build_solver_travel_matrices()

        # precompute feasible pairs
        self._precompute_feasible_pairs()

        # add constraints
        self._add_constraints()

        # determine proxy distance
        self._determine_exact_path_dist()

    def _explode_activities_to_roles(self) -> None:
        """
        Populates self.roles by exploding each activity based on its vessel requirements.
        """
        # initialize self.roles
        self.roles = []
        current_role_id = 0

        for idx_a, activity in enumerate(self.activities):
            if (activity.is_maintenance_type() or activity.is_current_location) and hasattr(activity, "target_vessel"):
                # these activities require ONE specific vessel
                vessel_idx = self.vessel_name_to_idx.get(activity.target_vessel)
                if vessel_idx is not None:
                    domain = [vessel_idx]
                    group_id = f"act_{idx_a}_grp_0"  # Unique group for this single role
                    self.roles.append(Role(role_id=current_role_id, parent_activity=activity, parent_activity_idx=idx_a, vessel_domain=domain, role_group_id=group_id))
                    current_role_id += 1

            elif activity.is_tow_type():
                configuration_groups = activity.required_vessel_config[0]
                group_idx = 0
                # incorporates the 'config' logic
                for _, group_req in enumerate(configuration_groups):
                    vessel_names = group_req.get("vessels", [])
                    count = group_req.get("count", 0)

                    # Get the indices for the vessel names
                    domain = []
                    if "*" in vessel_names:
                        domain = list(range(self.num_vessels))  # all vessels
                    else:
                        domain = [self.vessel_name_to_idx[name] for name in vessel_names if name in self.vessel_name_to_idx]

                    group_id = f"act_{idx_a}_grp_{group_idx}"
                    if len(domain) == 0:
                        # no valid vessels are available
                        logger.error(f"{activity.activity_name} ({group_idx}): Under Saturated")
                        continue
                    elif len(domain) < count:
                        # not enough valid vessels for the number of vessels required
                        logger.error(f"{activity.activity_name} ({group_idx}): Under Saturated")
                        continue
                    elif len(domain) == count:
                        # perfect saturation -> we don't have to worry about having the full domain
                        for i in range(count):
                            self.roles.append(Role(role_id=current_role_id, parent_activity=activity, parent_activity_idx=idx_a, vessel_domain=[domain[i]], role_group_id=group_id))
                            current_role_id += 1
                            group_idx += 1
                    else:
                        # over saturated situation
                        for i in range(count):
                            self.roles.append(Role(role_id=current_role_id, parent_activity=activity, parent_activity_idx=idx_a, vessel_domain=domain, role_group_id=group_id))
                            current_role_id += 1
                        group_idx += 1

    def _precompute_feasible_pairs(self) -> None:
        self.feasible_pairs = []
        self.feasible_pairs_vessels = {}

        self.time_infeasible_pairs = []

        for i in range(self.num_roles):
            for j in range(self.num_roles):
                # skip if same role
                if i == j:
                    continue

                # get roles
                r_i = self.roles[i]
                r_j = self.roles[j]

                # skip if same parent activity
                if r_i.parent_activity_idx == r_j.parent_activity_idx:
                    continue

                # determine domain overlap
                overlap = list(set(r_i.vessel_domain) & set(r_j.vessel_domain))

                # skip if there is no overlap in vessel domain
                if len(overlap) == 0:
                    continue

                # get parent activities
                a_i = r_i.parent_activity
                a_j = r_j.parent_activity

                # skip if best case fails
                if a_j.possible_start[-1] < a_i.possible_start[0] + a_i.duration:
                    # symmetrically time infeasible pairs
                    if a_i.possible_start[-1] < a_j.possible_start[0] + a_j.duration:
                        self.time_infeasible_pairs.append((i, j))

                    continue

                self.feasible_pairs.append((i, j))
                self.feasible_pairs_vessels[(i, j)] = overlap

    def _build_variables(self) -> None:
        """
        Create decision variables for the scheduling model.
        """
        self.start_time_vars = []  # idx_activity -> cp.IntVar
        self.location_vars = []  # idx_activity -> cp.IntVar (for maintenance only, -1 otherwise)
        self.assigned_vessel_vars = []  # idx_role -> cp.IntVar (domain=vessel_indices)

        # populate per activity variables
        for _, activity in enumerate(self.activities):
            # initialise start times for activities
            self.start_time_vars.append(
                cp.intvar(
                    name=f"start_{activity.activity_name}",
                    lb=activity.possible_start[0],
                    ub=activity.possible_start[-1],
                )
            )

            if activity.is_tow_type():
                # check for predecessor
                if activity.predecessor_name and activity.predecessor_name != "":
                    predecessor_id = self.activity_name_to_idx[activity.predecessor_name]

                    constraint = self.start_time_vars[-1] >= (
                        self.start_time_vars[predecessor_id] + self.activities[predecessor_id].duration
                    )
                    constraint.set_description(
                        f"{activity.activity_name} must follow {self.activities[predecessor_id].activity_name} (PREDECESSOR)"
                    )
                    self.model += constraint

                # set value to be equal to -1 for tows
                var = cp.intvar(name=f"location_{activity.activity_name}", lb=-1, ub=-1)
                self.model += var == -1
                self.location_vars.append(var)
            elif activity.is_maintenance_type():
                # initialise decision variables for which location should be used for maintenance
                self.location_vars.append(
                    cp.intvar(
                        name=f"location_{activity.activity_name}",
                        lb=0,
                        ub=len(activity.allowed_locations) - 1,
                    )
                )
            elif activity.is_current_location:
                var = cp.intvar(name=f"location_{activity.activity_name}", lb=-1, ub=-1)
                self.model += var == -1
                self.location_vars.append(var)
            else:
                raise ValueError(f"Unknown activity type for {activity.activity_name}")

        # Populate per-role variables
        for idx_r, role in enumerate(self.roles):
            # domain is the pre-filtered list of allowed vessel indices
            domain = role.vessel_domain
            if not domain:
                raise ValueError(f"Role {role} has no possible vessels in its domain.")
            var = cp.intvar(
                name=f"role_{idx_r}_{role.parent_activity.activity_name}", 
                lb=min(domain), 
                ub=max(domain)
            )
            self.assigned_vessel_vars.append(var)

            # add the domain constraint
            if len(domain) == 1:
                # there is only one allowed vessel
                constraint = var == domain[0]
                constraint.set_description(f"{role.parent_activity.activity_name} <=> {self.vessels[domain[0]].vessel_name}")
                self.model += constraint
            else:
                self.model += cp.any([var == d for d in domain])
                # constraint.set_description(f"{role.parent_activity.activity_name} <=> ({[self.vessels[d].vessel_name for d in domain]})")

    def _build_solver_travel_matrices(self) -> None:
        """
        Pre-calculate distances and speed-dependent lookup tables.

        For each vessel v and location pair p=(i, j):
        - travel_time[v][p,s] = ceil(distance(i,j)/speed_s*24)
        - fuel[v][p,s] = cel(fuel_rate_s * travel_time[v][p,s])
        - cost[v][p,s] = day_rate * min(travel_time, eco_time) + fuel_price * fuel
        """
        # initialise location maps and coordinates
        self.all_locations_map = {}  # name -> id
        self.all_locations_coords = []  # id -> [lat, lon]
        loc_id = 0

        # iterate through activities
        for activity in self.activities:
            if activity.is_tow_type():
                # add start and end locations (if unique)
                for loc_name, loc_coords in [(activity.start_location_name, activity.start_location), (activity.end_location_name, activity.end_location)]:
                    if loc_name not in self.all_locations_map:
                        self.all_locations_map[loc_name] = loc_id
                        self.all_locations_coords.append(loc_coords)
                        loc_id += 1
            elif activity.is_maintenance_type():
                # maintenance activity (multiple possible locations)
                for loc_name, loc_coords in zip(activity.allowed_locations_names, activity.allowed_locations):
                    if loc_name not in self.all_locations_map:
                        self.all_locations_map[loc_name] = loc_id
                        self.all_locations_coords.append(loc_coords)
                        loc_id += 1
            elif activity.is_current_location:
                # current location
                key = f"current_loc_{activity.target_vessel}"
                if key not in self.all_locations_map:
                    self.all_locations_map[key] = loc_id
                    self.all_locations_coords.append(activity.start_location)
                    loc_id += 1

        # get total number of locations and generate travel matrices
        self.num_locs = len(self.all_locations_coords)
        self.num_loc_pairs = self.num_locs * self.num_locs

        self.vessel_travel_dist_matrix = []  # Integer NM
        self.vessel_travel_dist_matrix_float = []  # Float NM

        # Matrices for Time Bounds
        self.vessel_travel_time_fastest_matrix = []  # Integer Days (at max possible speed)
        self.vessel_travel_time_slowest_matrix = []  # Integer Days (at lowest possible speed)

        self.vessel_speed_values = []
        self.vessel_fuel_rate_values = []
        self.vessel_time_by_pair_speed = []
        self.vessel_fuel_by_pair_speed = []
        self.vessel_num_speed_steps = []
        self.max_num_speed_steps = 1

        global_dist_matrix = np.zeros((self.num_locs, self.num_locs), dtype=float)

        for i in range(self.num_locs):
            for j in range(self.num_locs):
                if i == j:
                    continue
                coords_i = self.all_locations_coords[i]
                coords_j = self.all_locations_coords[j]
                global_dist_matrix[i, j] = calculate_sailing_distance(
                    lon_init=coords_i[0], lat_init=coords_i[1], 
                    lon_final=coords_j[0], lat_final=coords_j[1]
                )[0]

        # round up all distances to get integers
        int_dist_matrix_for_solver = np.ceil(global_dist_matrix).astype(int)

        # calculate minimum AND maximum travel time
        for vessel in self.vessels:
            # Save distance matrices (invariant per vessel)
            self.vessel_travel_dist_matrix.append(cp.cparray(int_dist_matrix_for_solver))
            self.vessel_travel_dist_matrix_float.append(global_dist_matrix)

            speeds = np.array(vessel.possible_speeds, dtype=float)
            if speeds.size == 0:
                speeds = np.array([max(vessel.sailing_speed_max, 0.1)], dtype=float)

            fuel_rates = np.array(vessel.corresponding_fuel, dtype=float)
            if fuel_rates.size != speeds.size:
                fuel_rates = np.zeros_like(speeds) + float(vessel.fuel_consumed_eco)

            self.vessel_speed_values.append(speeds)
            self.vessel_fuel_rate_values.append(fuel_rates)

            S_v = int(len(speeds))
            self.vessel_num_speed_steps.append(S_v)
            self.max_num_speed_steps = max(self.max_num_speed_steps, S_v)

            max_speed_kn = float(np.max(speeds))
            min_speed_kn = float(np.min(speeds))

            time_fastest = np.ceil(global_dist_matrix / (max_speed_kn * 24.0)).astype(int) if max_speed_kn > 0 else np.zeros_like(global_dist_matrix, dtype=int)
            time_slowest = np.ceil(global_dist_matrix / (min_speed_kn * 24.0)).astype(int) if min_speed_kn > 0 else time_fastest.copy()

            self.vessel_travel_time_fastest_matrix.append(cp.cparray(time_fastest))
            self.vessel_travel_time_slowest_matrix.append(cp.cparray(time_slowest))

            # flattened [pair * S_v + speed_idx]
            time_flat = np.zeros((self.num_loc_pairs, S_v), dtype=int)
            fuel_flat = np.zeros((self.num_loc_pairs, S_v), dtype=int)

            for li in range(self.num_locs):
                for lj in range(self.num_locs):
                    p = li * self.num_locs + lj
                    d_nm = global_dist_matrix[li, lj]

                    for s_idx in range(S_v):
                        spd = float(speeds[s_idx])
                        fr = float(fuel_rates[s_idx])
                        t_days = int(np.ceil(d_nm / (spd * 24.0))) if d_nm > 0 and spd > 0 else 0

                        time_flat[p, s_idx] = t_days
                        fuel_flat[p, s_idx] = int(np.ceil(fr * t_days))

            self.vessel_time_by_pair_speed.append(cp.cparray(time_flat.flatten()))
            self.vessel_fuel_by_pair_speed.append(cp.cparray(fuel_flat.flatten()))

        # link roles to locations
        self.role_start_loc_idx = []  # [idx_r] -> location_id (or expression)
        self.role_end_loc_idx = []  # [idx_r] -> location_id (or expression)

        for role in self.roles:
            act = role.parent_activity
            idx_a = role.parent_activity_idx

            if act.is_tow_type():
                # get corresponding start and end locations
                start_loc_id = self.all_locations_map[act.start_location_name]
                end_loc_id = self.all_locations_map[act.end_location_name]
            elif act.is_maintenance_type():
                # get allowed start and end locations
                allowed_loc_ids = [self.all_locations_map[name] for name in act.allowed_locations_names]
                start_loc_id = cp.Element(allowed_loc_ids, self.location_vars[idx_a])
                end_loc_id = start_loc_id
            elif act.is_current_location:
                # get current location
                loc_id = self.all_locations_map[f"current_loc_{act.target_vessel}"]
                start_loc_id, end_loc_id = loc_id, loc_id
            else:
                raise ValueError(f"Unknown activity type for role {role.role_id}")

            self.role_start_loc_idx.append(start_loc_id)
            self.role_end_loc_idx.append(end_loc_id)

    def _add_constraints(self):
        """
        Add constraints to the scheduling model.
        """
        # split all roles into corresponding activities
        roles_by_activity = {}
        for idx_r, role in enumerate(self.roles):
            # add current role to corresponding group
            roles_by_activity.setdefault(role.parent_activity_idx, []).append(self.assigned_vessel_vars[idx_r])

        # prevent any duplicate vessles in a single activity
        for idx_a, role_vars_list in roles_by_activity.items():
            if len(role_vars_list) > 1:
                for v1, v2 in combinations(role_vars_list, 2):
                    constraint = v1 != v2
                    constraint.set_description(f"No duplicate vessels for activity: {self.activities[idx_a].activity_name}")
                    self.model += constraint

        # ensure there is no overlap between vessel assignments
        for idx_v, vessel in enumerate(self.vessels):
            # get list of all roles this vessel is allowed to perform
            potential_role_indices = [idx_r for idx_r, role in enumerate(self.roles) if idx_v in role.vessel_domain]
            if len(potential_role_indices) < 2:
                continue

            travel_time_matrix_for_v = self.vessel_travel_time_fastest_matrix[idx_v]

            # iterate over all pairs of vessels
            for i in range(len(potential_role_indices)):
                for j in range(i + 1, len(potential_role_indices)):
                    # get corresponding roles and activities for this pair
                    idx_r1 = potential_role_indices[i]
                    idx_r2 = potential_role_indices[j]
                    role1 = self.roles[idx_r1]
                    role2 = self.roles[idx_r2]

                    # trigger variable, true if vessel is assigned to this role
                    B1_var = cp.BoolVar(name=f"B_overlap_{idx_v}_{idx_r1}")
                    B2_var = cp.BoolVar(name=f"B_overlap_{idx_v}_{idx_r2}")

                    self.model += B1_var == (self.assigned_vessel_vars[idx_r1] == idx_v)
                    self.model += B2_var == (self.assigned_vessel_vars[idx_r2] == idx_v)
                    both_on_v = B1_var & B2_var

                    # get start and end times
                    start_1 = self.start_time_vars[role1.parent_activity_idx]
                    end_1 = start_1 + role1.parent_activity.duration
                    start_2 = self.start_time_vars[role2.parent_activity_idx]
                    end_2 = start_2 + role2.parent_activity.duration

                    # Look up Travel times from start loc 1 to end loc 2 and vice versa
                    end_loc_1 = self.role_end_loc_idx[idx_r1]
                    start_loc_2 = self.role_start_loc_idx[idx_r2]
                    travel_1_to_2 = travel_time_matrix_for_v[end_loc_1, start_loc_2]

                    end_loc_2 = self.role_end_loc_idx[idx_r2]
                    start_loc_1 = self.role_start_loc_idx[idx_r1]
                    travel_2_to_1 = travel_time_matrix_for_v[end_loc_2, start_loc_1]

                    # order literal
                    order_ij_v = cp.boolvar(name=f"ord_v{idx_v}_r{idx_r1}_r{idx_r2}")
                    self.model += order_ij_v <= both_on_v

                    # role1 before role2 on this vessel
                    constraint = order_ij_v.implies(start_2 >= end_1 + travel_1_to_2)
                    constraint.set_description(f"{vessel.vessel_name}: {role1.parent_activity.activity_name} -> {role2.parent_activity.activity_name} (sufficient mob time)")
                    self.model += constraint

                    # role2 before role1 on this vessel
                    constraint = (both_on_v & (~order_ij_v)).implies(start_1 >= end_2 + travel_2_to_1)
                    constraint.set_description(f"{vessel.vessel_name}: {role2.parent_activity.activity_name} -> {role1.parent_activity.activity_name} (sufficient mob time)")
                    self.model += constraint

    def _determine_exact_path_dist(self) -> None:
        """
        Calculates path cost and enforces topology.
        Implements Deterministic Index-Based Assignment for Saturated Active Groups.

        Refactoring Note:
        - Objective: Minimize Total Ballast Distance (Mobilization + Inter-project).
        - Logic: Cost is applied to the transition arc (End_A -> Start_B).
        - Exclusions: Operational travel (Start_A -> End_A) is time-constrained but cost-free.
        - Optimization: Distance cost is decoupled from vessel choice as distance is vessel-invariant.
        """
        self.is_next = {}
        self.speed_idx_vars = {}
        objective_terms = []

        # 1. Instantiate sparse variables only for strictly feasible pairs
        for (i, j) in self.feasible_pairs:
            self.is_next[(i, j)] = cp.boolvar(name=f"is_next_{i}_{j}")
            common = self.feasible_pairs_vessels.get((i, j), [])
            arc_speed_ub = int(min([self.vessel_num_speed_steps[v] - 1 for v in common])) if common else 0
            self.speed_idx_vars[(i, j)] = cp.intvar(lb=0, ub=arc_speed_ub, name=f"spd_idx_{i}_{j}")

        # 2. Add Routing Constraints directly from sparse mappings
        for i in range(self.num_roles):
            out_vars = [self.is_next[(i, j)] for j in range(self.num_roles) if (i, j) in self.is_next]
            in_vars = [self.is_next[(j, i)] for j in range(self.num_roles) if (j, i) in self.is_next]
            
            self.model += cp.sum(out_vars) <= 1
            if self.roles[i].parent_activity.is_current_activity:
                self.model += cp.sum(in_vars) == 0
            else:
                self.model += cp.sum(in_vars) == 1

        # 3. Add Time Infeasible bounds directly (skip loop logic over all N x N combinations)
        for (i, j) in self.time_infeasible_pairs:
            self.model += self.assigned_vessel_vars[i] != self.assigned_vessel_vars[j]

        # 4. Build constraint and objective relationships for feasible loops only
        reference_dist_matrix = self.vessel_travel_dist_matrix[0]

        for (i, j) in self.feasible_pairs:
            var_next = self.is_next[(i, j)]
            spd_idx = self.speed_idx_vars[(i, j)]
            var_v_i = self.assigned_vessel_vars[i]
            var_v_j = self.assigned_vessel_vars[j]

            # Tighter propagation: Only active if on same vessel
            self.model += var_next <= (var_v_i == var_v_j)

            end_loc_1 = self.role_end_loc_idx[i]
            start_loc_2 = self.role_start_loc_idx[j]
            pair_idx = end_loc_1 * self.num_locs + start_loc_2

            dur_i = self.roles[i].parent_activity.duration
            start_i = self.start_time_vars[self.roles[i].parent_activity_idx]
            start_j = self.start_time_vars[self.roles[j].parent_activity_idx]
            gap_ij = start_j - (start_i + dur_i)

            self.model += var_next.implies(start_j >= start_i + dur_i)

            if self.objective == "distance":
                dist_val = reference_dist_matrix[end_loc_1, start_loc_2]
                objective_terms.append(var_next * dist_val)

            # Use a single cost or fuel variable to prevent variable bloat
            if self.objective in ["cost", "fuel"]:
                arc_obj_term = cp.intvar(0, np.iinfo(np.int32).max, name=f"obj_term_{i}_{j}")
                self.model += (~var_next).implies(arc_obj_term == 0)
                objective_terms.append(arc_obj_term)

            for idx_v in self.feasible_pairs_vessels[(i, j)]:
                is_active_for_v = var_next & (var_v_i == idx_v)
                S_v = self.vessel_num_speed_steps[idx_v]
                
                flat_index = pair_idx * S_v + spd_idx
                t_lookup = cp.Element(self.vessel_time_by_pair_speed[idx_v], flat_index)
                f_lookup = cp.Element(self.vessel_fuel_by_pair_speed[idx_v], flat_index)

                self.model += is_active_for_v.implies(gap_ij >= t_lookup)

                if self.objective == "fuel":
                    self.model += is_active_for_v.implies(arc_obj_term == f_lookup)
                elif self.objective == "cost":
                    alpha_scaled = int(round(self.idle_cost_discount))
                    day_rate = int(round(self.vessels[idx_v].day_rate_mob))
                    fuel_price = int(round(self.vessels[idx_v].default_fuel_price))
                    
                    cost_expr = (day_rate * t_lookup) + (fuel_price * f_lookup) + (alpha_scaled * day_rate * (gap_ij - t_lookup))
                    self.model += is_active_for_v.implies(arc_obj_term == cost_expr)

        self.objective_var = cp.intvar(0, np.iinfo(np.int32).max, name=f"objective_{self.objective}")

        if self.objective == "duration":
            end_times = [self.start_time_vars[a] + int(self.activities[a].duration) for a in range(self.num_activities)]
            self.model += self.objective_var == (cp.max(end_times) - cp.min(self.start_time_vars))
        else:
            self.model += self.objective_var == cp.sum(objective_terms)

    def check_feasibility(self, time_limit: int | None = None) -> bool:
        """
        Pure satisfiability check — no objective, stops at the first feasible solution.

        Much faster than ``solve_k_best`` for seed screening because OR-Tools
        halts as soon as any assignment satisfies all constraints, without
        searching for an optimal value.

        Parameters
        ----------
        time_limit:
            Wall-clock seconds before giving up.  ``None`` means no limit.

        Returns
        -------
        bool
            ``True`` if a feasible solution exists, ``False`` otherwise.
        """
        solver_kwargs: dict = {"stop_after_first_solution": True}
        if time_limit is not None:
            solver_kwargs["time_limit"] = time_limit
        return bool(self.model.solve(solver="ortools", **solver_kwargs))

    def solve_single_solution(self):
        """
        Finds single solution.
        """
        logger.info(f"Checking input data")
        self.check_for_unassigned_vessels()

        solver = cp.SolverLookup.get("ortools", self.model)
        status = solver.solve()

        if status:
            self.assigned_vessel_results = []
            self.start_time_results = []
            self.location_results = []
            self.speed_results = []
            self._collect_current_solution()
            self.num_solutions = 1
            self.post_process_all_results()
            return status, "Valid input."
        else:
            return status, "Invalid input."

    def solve_min_max(self, run_time_per_sol: int = 60, num_workers: int = 14, verbose: bool = False):
        """
        Finds the Minimum and Maximum solutions (2 solutions total).
        """

        # Reset results
        self.assigned_vessel_results = []
        self.start_time_results = []
        self.location_results = []
        self.link_results = []
        self.objective_values = []
        self.speed_results = []

        solver_params = {
            "max_time_in_seconds": run_time_per_sol, 
            "num_search_workers": num_workers
        }

        solutions_found = 0
        
        # Define the two objectives we want to run
        # We store the method (minimize/maximize) and a label for logging
        phases = [("Min", self.model.minimize), ("Max", self.model.maximize)]

        for phase_name, objective_method in phases:
            objective_method(self.objective_var)

            if verbose: logger.debug('instantiating solver...')
            solver = cp.SolverLookup.get("ortools", self.model)
            
            if verbose: logger.info(f"Solving for {phase_name}imum...")
            status = solver.solve(**solver_params)

            if not status:
                # Handle cases where no solution is found or time limit hit
                if solver.status().runtime >= run_time_per_sol:
                    logger.warning(f"{phase_name} search: Time limit reached without optimal solution.")
                    if self.objective_var.value() is None:
                        continue # Skip to next phase if no feasible solution found
                else:
                    logger.warning(f"{phase_name} search: No feasible solution found.")
                    continue

            # 3. Validation
            is_valid, output = verify_path_logic(self, False)
            if not is_valid:
                logger.warning(output)
                raise Exception(f"Invalid path logic in {phase_name} solution.")

            # 4. Save solution
            current_value = self.objective_var.value()
            solutions_found += 1
            
            if verbose: logger.info(f"{phase_name} Solution: {self.objective} = {current_value}")

            self._collect_current_solution()
            self.objective_values.append(current_value)

        if verbose: logger.info(f"Finished. Found {solutions_found} solutions.")
        self.num_solutions = solutions_found
        
        if solutions_found > 0:
            self.post_process_all_results()

        return solutions_found

    def solve_k_best(self, k: int = 5, run_time_per_sol: int|None=None, max_runtime_seconds: float|None=None, num_workers: int = 14, verbose: bool = False):
        """
        Finds top K solutions.
        """

        # logger.info(f"Checking input data")
        self.check_for_unassigned_vessels()
        
        if verbose: logger.info(f"Searching for top {k} solutions...")

        # reset results
        self.assigned_vessel_results = []
        self.start_time_results = []
        self.location_results = []
        self.link_results = []
        self.objective_values = []
        self.speed_results = []

        self.model.minimize(self.objective_var)

        solver_params = {
            "num_search_workers": num_workers,
        }

        solutions_found = 0
        start_time = time.time()

        for i in range(k):
            if max_runtime_seconds is not None:
                elapsed = time.time() - start_time
                remaining = max_runtime_seconds - elapsed
                if remaining < 1:
                    logger.warning(f"Iteration {i+1}: Overall time limit reached before solving.")
                    break
                solver_params['max_time_in_seconds'] = int(remaining)
            elif run_time_per_sol:
                solver_params['max_time_in_seconds'] = run_time_per_sol

            solver = cp.SolverLookup.get("ortools", self.model)
            status = solver.solve(**solver_params)

            if not status:
                if max_runtime_seconds is not None or run_time_per_sol:
                    if solver.status().runtime >= solver_params.get('max_time_in_seconds', float('inf')):
                        logger.warning(f"Iteration {i+1}: Time limit reached.")
                        if self.objective_var.value() is None:
                            break
                else:
                    logger.info(f"Iteration {i+1}: No more feasible solutions.")
                    break

            # # check if valid solution
            # is_valid, output = verify_path_logic(self, False)
            # if not is_valid:
            #     logger.warning(output)
            #     raise Exception("Invalid path logic.")

            # save current solution
            current_value = self.objective_var.value()
            solutions_found += 1
            if verbose: logger.info(f"Rank {solutions_found}: {self.objective.capitalize()} = {current_value}")

            self._collect_current_solution()
            self.objective_values.append(current_value)

            # ensure unique solution
            exclusion_clause = []
            for idx_r, r in enumerate(self.roles):
                # get current assigned value for role
                val = self.assigned_vessel_vars[idx_r].value()

                # get corresponding activity
                idx_a = r.parent_activity_idx

                # prevent vessel from being assigned to any of the roles for this activity
                exclusion_clause.append(
                    cp.all([
                        self.assigned_vessel_vars[idx_role] != val 
                        for idx_role, role in enumerate(self.roles) 
                        if role.parent_activity_idx == idx_a
                    ])
                )

            for idx_a in range(self.num_activities):
                # alternatively, change the location of the maintenance
                val = self.location_vars[idx_a].value()
                exclusion_clause.append(self.location_vars[idx_a] != val)

            self.model += cp.any(exclusion_clause)
            solver += cp.any(exclusion_clause)

        if verbose: logger.info(f"Finished. Found {solutions_found} solutions.")
        self.num_solutions = solutions_found
        if solutions_found > 0:
            self.post_process_all_results()

        return solutions_found

    def _resolve_activity_locations(self, act, act_idx, locations, current_vessel_loc):
        """Helper to determine start and end location IDs for a vessel doing an activity."""
        start_id = -1
        end_id = -1
        
        if act.is_tow_type():
            start_id = self.all_locations_map[act.start_location_name]
            end_id = self.all_locations_map[act.end_location_name]
        elif act.is_maintenance_type():
            loc_idx = locations[act_idx]
            loc_name = act.allowed_locations_names[loc_idx]
            start_id = self.all_locations_map[loc_name]
            end_id = start_id
        elif act.is_current_location:
            start_id = current_vessel_loc
            end_id = current_vessel_loc
            
        return start_id, end_id
    
    def _calc_arrival(self, v_idx, state, dest_id):
        """Helper to calculate arrival time."""
        if not state['is_fixed']:
            # Floating start: Travel time is 0 (spawns at location)
            return 0
            
        origin_id = state['loc']
        if origin_id == -1 or dest_id == -1:
            return state['time'] # Should not happen in valid flow
            
        travel_time = self.vessel_travel_time_fastest_matrix[v_idx][origin_id, dest_id]
        return state['time'] + travel_time

    def _convert_ga_solutions(self, ga_solutions: list[Chromosome]) -> None:
        """
        Convert GA solutions to SchedulingModel result format.
        """
        self.assigned_vessel_results = []
        self.start_time_results = []
        self.location_results = []
        self.speed_results = []
        
        for chromosome in ga_solutions:
            sol = chromosome.decoded_solution

            self.assigned_vessel_results.append(sol.assigned_vessels.tolist())  # type: ignore
            self.start_time_results.append(sol.start_times.tolist())  # type: ignore
            self.location_results.append(sol.locations.tolist())  # type: ignore

            # Build dense role->role speed index matrix for compatibility with CP post-processing
            speed_dense = np.zeros((self.num_roles, self.num_roles), dtype=int)

            # If decoder produced per-role speed indices, map to vessel sequence arcs
            if hasattr(sol, "speed_indices") and sol.speed_indices is not None: # type: ignore
                # map (vessel, activity) -> role_id assigned to that vessel
                role_for_act_vessel = {}
                for r in self.roles:
                    v_assigned = int(sol.assigned_vessels[r.role_id])  # type: ignore
                    if v_assigned >= 0:
                        role_for_act_vessel[(v_assigned, r.parent_activity_idx)] = r.role_id

                # fill arc speeds: previous role -> current role
                for v_idx, seq in enumerate(sol.vessel_sequences):  # type: ignore
                    if len(seq) < 2:
                        continue
                    for k in range(1, len(seq)):
                        a_prev = seq[k - 1]
                        a_curr = seq[k]

                        r_prev = role_for_act_vessel.get((v_idx, a_prev), None)
                        r_curr = role_for_act_vessel.get((v_idx, a_curr), None)

                        if r_prev is None or r_curr is None:
                            continue

                        s_idx = int(sol.speed_indices[r_curr])  # type: ignore # incoming speed to current activity
                        speed_dense[r_prev, r_curr] = s_idx

            self.speed_results.append(speed_dense.tolist())

        self.num_solutions = len(ga_solutions)

        if self.num_solutions > 0:
            self.post_process_all_results()

    def debug_base_model(self):
        print("--- DEBUG: Testing Base Model Feasibility ---")
        solver = cp.SolverLookup.get("ortools", self.model)
        solver.minimize(0)
        status = solver.solve(time_limit=10)
        
        if status:
            print("✅ Base model is FEASIBLE. The issue is likely the repair objective or time limit.")
            print(f"Sample valid start times: {[v.value() for v in self.start_time_vars]}")
            print(f'Sample valid assignments: {[v.value() for v in self.assigned_vessel_vars]}')
            print(f'Sample valid locations: {[v.value() for v in self.location_vars]}')

        else:
            print("❌ Base model is INFEASIBLE. You have conflicting hard constraints.")

    def check_for_unassigned_vessels(self) -> bool:
        """
        Diagnose vessels that are unusable due to missing or overcrowded start nodes.
        Logic:
        - Specific Assignment (1 Vessel, 1 Role): Valid.
        - Saturated Group (N Vessels, N Roles): Valid (Perfect Match).
        - Ambiguous Group (N Vessels, <N Roles): Invalid (Ghosting).
        """

        # --- PRE-PROCESSING: Group Roles by Activity ---
        # We need to count how many roles exist for each activity to determine supply.
        activity_stats = {}  # act_idx -> {count: int, domain: list, name: str}

        for r in self.roles:
            if r.parent_activity.is_current_activity:
                idx = r.parent_activity_idx
                if idx not in activity_stats:
                    activity_stats[idx] = {"count": 0, "domain": r.vessel_domain, "name": r.parent_activity.activity_name}
                activity_stats[idx]["count"] += 1

        # --- CHECK 1: DETECT AMBIGUOUS ACTIVE PROJECTS ---
        found_ambiguous = False

        for idx, stats in activity_stats.items():
            role_count = stats["count"]
            domain_size = len(stats["domain"])

            # CASE A: Perfect Match (Saturated)
            # e.g., 5 Vessels in class, 5 Roles created.
            if domain_size == role_count:
                continue

            # CASE B: Ambiguous (Ghosting Risk)
            # e.g., 5 Vessels in class, but only 1 Role created.
            if domain_size > role_count:
                found_ambiguous = True
                vessel_names = [self.vessels[i].vessel_name for i in stats["domain"]]

        if found_ambiguous:
            return False

        # --- CHECK 2: DETECT OVERCROWDING (Global Supply/Demand) ---
        # Standard pigeonhole check for remaining logic
        vessel_to_anchors = {i: set() for i in range(self.num_vessels)}

        for r in self.roles:
            act = r.parent_activity
            if act.is_current_location or act.is_current_activity:
                for v_idx in r.vessel_domain:
                    vessel_to_anchors[v_idx].add(r.role_id)

        signatures = {}
        for v_idx, anchors in vessel_to_anchors.items():
            sig = tuple(sorted(list(anchors)))
            if sig not in signatures:
                signatures[sig] = []
            signatures[sig].append(v_idx)

        issues_found = False
        for anchors, v_indices in signatures.items():
            supply = len(anchors)
            demand = len(v_indices)

            if demand > supply:
                issues_found = True
                names = [self.vessels[i].vessel_name for i in v_indices]
                return False

        if not issues_found:
            return True
        return False

    def _collect_current_solution(self) -> None:
        """
        Helper function to extract and save the values from the
        most recent solution found by the solver.
        """
        try:
            # Extract results for this specific solution
            assigned_vessels = np.array([v.value() for v in self.assigned_vessel_vars])
            start_times = np.array([self.start_time_vars[idx_a].value() for idx_a in range(self.num_activities)])
            locations = np.array([self.location_vars[idx_a].value() for idx_a in range(self.num_activities)])
            
            speed_idx_dense = np.zeros((self.num_roles, self.num_roles), dtype=int)
            if isinstance(self.speed_idx_vars, dict):
                for (i, j), svar in self.speed_idx_vars.items():
                    val = svar.value()
                    speed_idx_dense[i, j] = int(val) if val is not None else 0
            else:
                speed_idx_dense = np.array([[self.speed_idx_vars[i, j].value() for j in range(self.num_roles)] for i in range(self.num_roles)])

            self.assigned_vessel_results.append(assigned_vessels.tolist())
            self.start_time_results.append(start_times.tolist())
            self.location_results.append(locations.tolist())
            self.speed_results.append(speed_idx_dense.tolist())
        except Exception as e:
            logger.error(f"Error collecting solution: {e}")

    def _get_global_loc_id(self, activity: Activity, solution_location_val: int, is_start_loc: bool) -> int:
        """
        Helper to get the global location ID from the all_locations_map.
        """
        if activity.is_tow_type():
            if is_start_loc:
                key = activity.start_location_name
            else:
                key = activity.end_location_name
            return self.all_locations_map[key]

        elif activity.is_maintenance_type():
            chosen_loc_index = solution_location_val
            chosen_loc_name = activity.allowed_locations_names[chosen_loc_index]
            return self.all_locations_map[chosen_loc_name]

        elif activity.is_current_location:
            key = f"current_loc_{activity.target_vessel}"
            return self.all_locations_map[key]

        return -1

    def _calculate_performance_metrics(self, sorted_role_indices: list[int], idx_v: int, idx_soln: int):
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
        tuple: A tuple containing:
            - results_lists (dict): Lists of metrics per activity segment (distance, duration, cost, fuel, speed).
            - results_totals (dict): Totals of the above metrics for the vessel.
        """

        distance_list = []
        duration_list = []
        mob_days_list = []
        cost_list = []
        fuel_list = []
        speed_list = []

        distance_total = 0
        duration_total = 0
        mob_days_total = 0
        cost_total = 0
        fuel_consumed_total = 0

        vessel = self.vessels[idx_v]
        dist_matrix_for_v = self.vessel_travel_dist_matrix_float[idx_v]

        if len(sorted_role_indices) > 1:
            for k in range(1, len(sorted_role_indices)):
                r_prev = sorted_role_indices[k - 1]
                r_curr = sorted_role_indices[k]

                role_prev = self.roles[r_prev]
                role_curr = self.roles[r_curr]

                idx_a_prev = role_prev.parent_activity_idx
                idx_a_curr = role_curr.parent_activity_idx

                act_prev = role_prev.parent_activity
                act_curr = role_curr.parent_activity

                prev_loc_val = self.location_results[idx_soln][idx_a_prev]
                curr_loc_val = self.location_results[idx_soln][idx_a_curr]

                end_loc_prev = self._get_global_loc_id(act_prev, prev_loc_val, is_start_loc=False)
                start_loc_curr = self._get_global_loc_id(act_curr, curr_loc_val, is_start_loc=True)

                distance = float(dist_matrix_for_v[end_loc_prev, start_loc_curr])
                distance_list.append(distance)
                distance_total += distance

                if (
                    idx_soln < len(self.speed_results)
                    and r_prev < len(self.speed_results[idx_soln])
                    and r_curr < len(self.speed_results[idx_soln][r_prev])
                ):
                    s_idx = int(self.speed_results[idx_soln][r_prev][r_curr])
                else:
                    s_idx = 0  # fallback eco

                s_idx = max(0, min(s_idx, len(vessel.possible_speeds) - 1))

                speed = float(vessel.possible_speeds[s_idx])
                speed_list.append(speed)

                if distance <= 0 or speed <= 0:
                    sailing_days = 0
                else:
                    sailing_days = int(np.ceil(distance / (speed * 24.0)))

                duration = 0.0 if speed <= 0 else (distance / (speed * 24.0) if distance > 0 else 0.0)

                fuel_rate = float(vessel.corresponding_fuel[s_idx])
                fuel_amt = float(np.ceil(fuel_rate * sailing_days))

                # Calculate the time gap (delta) between the end of the previous activity and start of current
                start_time_prev = self.start_time_results[idx_soln][idx_a_prev]
                start_time_curr = self.start_time_results[idx_soln][idx_a_curr]
                dur_prev = float(act_prev.duration)
                
                gap = start_time_curr - (start_time_prev + dur_prev)
                idle_time = max(0.0, gap - sailing_days)

                travel_cost = vessel.day_rate_mob * sailing_days
                fuel_cost = vessel.default_fuel_price * fuel_amt
                idle_cost = self.idle_cost_discount * vessel.day_rate_mob * idle_time
                
                mob_days = sailing_days + idle_time
                # cost = float(travel_cost + fuel_cost + idle_cost) / 1000
                cost = float(travel_cost + fuel_cost + idle_cost)

                duration_list.append(float(duration))
                mob_days_list.append(float(mob_days))
                fuel_list.append(float(fuel_amt))
                cost_list.append(float(cost))

                duration_total += duration
                mob_days_total += mob_days
                fuel_consumed_total += fuel_amt
                cost_total += cost

        results_lists = {
            "distance": distance_list,
            "duration": duration_list,
            "mob_days": mob_days_list,
            "cost": cost_list,
            "fuel": fuel_list,
            "speed": speed_list,
        }
        results_totals = {
            "distance": distance_total,
            "duration": duration_total,
            "mob_days": mob_days_total,
            "cost": cost_total,
            "fuel": fuel_consumed_total,
        }
        return results_lists, results_totals

    def post_process_current_results(self, idx_soln, sorted_assignments_per_vessel, sorted_roles_per_vessel):
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
        tuple: A tuple containing:
            - results_lists (dict): Lists of per-vessel metrics and assignments.
            - results_totals (dict): Aggregated totals across all vessels.
        """
        sorted_assignments = []
        sorted_distances = []
        sorted_durations = []
        sorted_costs = []
        sorted_mob_days = []
        sorted_fuels = []
        sorted_speeds = []

        total_distance = 0
        total_duration = 0
        total_mob_days = 0
        total_cost = 0
        total_fuel = 0

        for idx_v, _ in enumerate(self.vessels):
            sorted_activities = sorted_assignments_per_vessel[idx_v]
            sorted_roles = sorted_roles_per_vessel[idx_v]
            sorted_assignments.append(sorted_activities)

            results_lists, results_totals = self._calculate_performance_metrics(sorted_roles, idx_v, idx_soln)

            total_distance += results_totals["distance"]
            total_mob_days += results_totals["mob_days"]
            total_cost += results_totals["cost"]
            total_fuel += results_totals["fuel"]

            sorted_distances.append(results_lists["distance"])
            sorted_durations.append(results_lists["duration"])
            sorted_mob_days.append(results_lists["mob_days"])
            sorted_costs.append(results_lists["cost"])
            sorted_fuels.append(results_lists["fuel"])
            sorted_speeds.append(results_lists["speed"])

        # Calculate makespan for the entire schedule
        case_start_times = self.start_time_results[idx_soln]
        if case_start_times:
            end_times = [
                case_start_times[i] + self.activities[i].duration
                for i in range(self.num_activities)
                if case_start_times[i] is not None  # Only include scheduled activities
            ]
            if end_times:  # Only calculate if there are scheduled activities
                scheduled_start_times = [t for t in case_start_times if t is not None]
                total_duration = float(max(end_times) - min(scheduled_start_times))
            else:
                total_duration = 0.0

        results_lists = {
            "assignments": sorted_assignments,
            "distance": sorted_distances,
            "duration": sorted_durations,
            "mob_days": sorted_mob_days,
            "cost": sorted_costs,
            "fuel": sorted_fuels,
            "speeds": sorted_speeds,
        }
        results_totals = {
            "distance": total_distance,
            "duration": total_duration,
            "mob_days": total_mob_days,
            "cost": total_cost,
            "fuel": total_fuel,
        }
        return results_lists, results_totals

    def post_process_all_results(self) -> None:
        """
        Post-process all solutions after solving to compute overall metrics.

        Iterates over all solutions, processes each solution's results by sorting
        assignments and computing performance metrics. Stores aggregated results
        and totals for further analysis or reporting.
        """

        # tasks assigned to each vessel for each solution
        self.sorted_assignments = []
        self.sorted_distances = []
        self.sorted_durations = []
        self.sorted_costs = []
        self.sorted_mob_days = []
        self.sorted_fuels = []
        self.sorted_speeds = []

        # total results per solution
        self.total_distance = np.zeros(self.num_solutions)
        self.total_duration = np.zeros(self.num_solutions)
        self.total_mob_days = np.zeros(self.num_solutions)
        self.total_cost = np.zeros(self.num_solutions)
        self.total_fuel = np.zeros(self.num_solutions)

        # iterate through solutions
        for idx_soln in range(self.num_solutions):
            # extract results for current solution
            case_assigned_vessels = self.assigned_vessel_results[idx_soln]
            case_start_times = self.start_time_results[idx_soln]
            case_location = self.location_results[idx_soln]

            vessel_schedules_unsorted = [[] for _ in self.vessels]  # (act_idx, start_time, role_idx)

            for idx_r, assigned_vessel_idx in enumerate(case_assigned_vessels):
                role = self.roles[idx_r]
                act_idx = role.parent_activity_idx
                start_time = case_start_times[act_idx]
                vessel_schedules_unsorted[assigned_vessel_idx].append((act_idx, start_time, idx_r))

            sorted_assignments_per_vessel = []
            sorted_roles_per_vessel = []

            for schedule in vessel_schedules_unsorted:
                sorted_schedule = sorted(schedule, key=lambda x: x[1])
                sorted_activity_indices = [act_idx for (act_idx, _, _) in sorted_schedule]
                sorted_role_indices = [role_idx for (_, _, role_idx) in sorted_schedule]
                sorted_assignments_per_vessel.append(sorted_activity_indices)
                sorted_roles_per_vessel.append(sorted_role_indices)

            results_lists, results_totals = self.post_process_current_results(
                idx_soln,
                sorted_assignments_per_vessel,
                sorted_roles_per_vessel
            )
            
            self.sorted_assignments.append(results_lists["assignments"])
            self.sorted_distances.append(results_lists["distance"])
            self.sorted_durations.append(results_lists["duration"])
            self.sorted_mob_days.append(results_lists["mob_days"])
            self.sorted_costs.append(results_lists["cost"])
            self.sorted_fuels.append(results_lists["fuel"])
            self.sorted_speeds.append(results_lists["speeds"])

            self.total_distance[idx_soln] = results_totals["distance"]
            self.total_duration[idx_soln] = results_totals["duration"]
            self.total_mob_days[idx_soln] = results_totals["mob_days"]
            self.total_cost[idx_soln] = results_totals["cost"]
            self.total_fuel[idx_soln] = results_totals["fuel"]

        self.results = {
            "distance": self.total_distance.tolist(),
            "duration": self.total_duration.tolist(),
            "cost": self.total_cost.tolist(),
            "fuel": self.total_fuel.tolist(),
        }
        self.total_mob_days_list = self.total_mob_days.tolist()

    def results_to_dict(self):
        """
        Create a dictionary for the results.
        """
        results_dict = {
            "date_origin": date.strftime(self.date_origin, "%Y/%m/%d"),
            "num_vessels": self.num_vessels,
            "vessels": [v.to_dict() for v in self.vessels],
            "activities": [a.to_dict() for a in self.activities],
            "sorted_assignments": self.sorted_assignments,
            "start_time_results": self.start_time_results,
            "location_results": self.location_results,
            "speed_idx_results": self.speed_results,
            "sorted_distances": self.sorted_distances,
            "sorted_durations": self.sorted_durations,
            "sorted_costs": self.sorted_costs,
            "sorted_fuels": self.sorted_fuels,
            "sorted_speeds": self.sorted_speeds,
            "results": self.results,
        }
        return results_dict

    def results_to_json(self):
        """
        Create a json string for results dictionary.
        """
        results_dict = self.results_to_dict()
        return json.dumps(results_dict)