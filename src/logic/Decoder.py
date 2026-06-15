from collections import defaultdict
import math
import random

import numpy as np
from typing import List, Tuple
from components.ga_types import Chromosome, DecodedSolution
from logic.scheduling_model import SchedulingModel

class Decoder:
    """
    Decodes a Chromosome into a feasible schedule based on different heuristic strategies.
    Supported strategies: 'BRKGA-A' (Random Assignment), 'BRKGA-STT' (Shortest Travel Time), 'BRKGA-GRASP',
        'BRKGA-DIRECT', 'BRKGA-PERMKEY', 'BRKGA-RCPSP', 'BRKGA-PERMKEY-URGENCY', 'BRKGA-PERMKEY-TOWS-FIRST', 'BRKGA-RANDOM'.
    """
    def __init__(self, model: SchedulingModel, strategy: str = "BRKGA-PERMKEY"):
        self.model = model
        self.strategy = strategy

    def decode(self, chromosome: Chromosome) -> DecodedSolution:
        """Main routing method to direct the chromosome to the correct decoding strategy."""
        random.seed(hash(chromosome))
        p = random.randint(1, 5)
        if self.strategy == "BRKGA-A":
            return self._decode_brkga_a(chromosome)
        elif self.strategy == "BRKGA-STT" or (self.strategy == "BRKGA-RANDOM" and p == 1):
            return self._decode_stt(chromosome)
        elif self.strategy == "BRKGA-GRASP" or (self.strategy == "BRKGA-RANDOM" and p == 2):
            return self._decode_grasp(chromosome)
        elif self.strategy == "BRKGA-DIRECT" or (self.strategy == "BRKGA-RANDOM" and p == 3):
            return self._decode_direct(chromosome)
        elif self.strategy == "BRKGA-PERMKEY" or (self.strategy == "BRKGA-RANDOM" and p == 4):
            return self._decode_permkey(chromosome)
        elif self.strategy == "BRKGA-RCPSP" or (self.strategy == "BRKGA-RANDOM" and p == 5):
            return self._decode_rcpsp_extended(chromosome)
        elif self.strategy == "BRKGA-PERMKEY-URGENCY":
            return self._decode_permkey_urgency(chromosome)
        elif self.strategy == "BRKGA-PERMKEY-TOWS-FIRST":
            return self._decode_permkey_tows_first(chromosome)
        else:
            raise ValueError(f"Unknown decoding strategy: {self.strategy}")
        
    def _decode_rcpsp_extended(self, chromosome: Chromosome) -> DecodedSolution:
        """
        RCPSP-Extended Decoder — Priority-Based List-Scheduling / Insertion Heuristic.

        How it maps to the extended RCPSP concepts:
        1. Role Priorities: start_time_keys dictates the scheduling order (sorted + topo).
        2. Location Choice: location_keys pre-computes maintenance ports.
        3. Vessel Assignment: role_priority_keys selects the preferred vessel from the domain.
        4. Speed Choice: speed_keys assigns transit speed.
        5. Schedule & Append: Activities are strictly appended to the end of a vessel's 
           current path (List-Scheduling).
        6. Repair Mechanism: If the preferred vessel violates hard constraints (time windows, 
           travel), the decoder forward-wraps through the domain to find the next feasible vessel.
        """
        # ── Step 1: Decode Locations & Anchor Ongoing Activities ────────────────
        locations = self._decode_locations(chromosome)
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices = np.full(self.model.num_roles, -1, dtype=int)
        
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)
        
        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = self.model.activities[anchor_idx].possible_start[0]

        # ── Step 2: Sort Activities (Priority-based List Scheduling) ────────────
        # Use start_time_keys as the "Priority Keys" from the RCPSP logic
        activity_priorities = [
            (act_idx, chromosome.start_time_keys[act_idx]) for act_idx in future_activities # type: ignore
        ]
        # Sort descending by priority key, then strictly enforce precedence
        sorted_acts = sorted(activity_priorities, key=lambda x: x[1], reverse=True)
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]
        ordered_act_indices = self._topological_sort(sorted_acts, anchored_indices)
        
        # Fast map: activity -> roles
        activity_to_roles: dict[int, list] = defaultdict(list)
        for r in self.model.roles:
            if r.parent_activity_idx in set(future_activities):
                activity_to_roles[r.parent_activity_idx].append(r)
                
        # ── Step 3: Sequential Insertion ────────────────────────────────────────
        for act_idx in ordered_act_indices:
            act_roles = activity_to_roles.get(act_idx, [])
            if not act_roles:
                continue
                
            # Sort roles by ascending domain size (most constrained first)
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))
            
            activity_feasible = True
            temp_assignments = []
            planned_vessels_for_this_act: set[int] = set()
            
            for role in act_roles_sorted:
                domain = role.vessel_domain
                N_r = len(domain)
                if N_r == 0:
                    activity_feasible = False
                    break
                    
                # Decode Vessel: Map role_priority_keys to a preferred vessel index
                rk_vessel = float(chromosome.role_priority_keys[role.role_id]) # type: ignore
                rk_vessel = min(max(rk_vessel, 0.0), 0.999999999)
                start_domain_idx = int(rk_vessel * N_r)
                
                role_scheduled = False
                
                # ── Step 4: Feasibility Check & Repair (Forward-wrap) ───────────
                for offset in range(N_r):
                    domain_idx = (start_domain_idx + offset) % N_r
                    v_idx = domain[domain_idx]
                    
                    # C4 constraint: no two roles of the same activity on the same vessel
                    if v_idx in planned_vessels_for_this_act:
                        continue 
                        
                    seq = vessel_sequences[v_idx]
                    # List-scheduling appends to the end of the resource's current queue
                    pos = len(seq) 
                    
                    is_feasible, temp_starts = self._is_insertion_feasible(
                        v_idx=v_idx,
                        target_act_idx=act_idx,
                        target_role_id=role.role_id,
                        pos=pos,
                        seq=seq,
                        locations=locations,
                        current_start_times=committed_start_times,
                        assigned_vessels=assigned_vessels,
                        chromosome=chromosome
                    )
                    
                    if is_feasible:
                        temp_assignments.append({
                            "role_id": role.role_id,
                            "v_idx": v_idx,
                            "pos": pos,
                            "starts": temp_starts
                        })
                        planned_vessels_for_this_act.add(v_idx)
                        role_scheduled = True
                        break  # Found a valid vessel, stop repair loop
                        
                if not role_scheduled:
                    activity_feasible = False
                    break  # All-or-Nothing: if one role fails, drop the whole activity
                    
            # ── Step 5: Commit or Drop ──────────────────────────────────────────
            if activity_feasible:
                for a in temp_assignments:
                    rid = a["role_id"]
                    v_idx = a["v_idx"]
                    assigned_vessels[rid] = v_idx
                    speed_indices[rid] = self._decode_speed_index_for_role(chromosome, rid, v_idx)
                    vessel_sequences[v_idx].insert(a["pos"], act_idx)
                    committed_start_times.update(a["starts"])
                    
        # ── Step 6: Finalize & Evaluate ─────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices
        )

    def _decode_direct(self, chromosome: Chromosome) -> DecodedSolution:
        """
        BRKGA-DIRECT — Fully-expressive decoder.

        Unlike the greedy decoders (STT, BRKGA-A, GRASP), this decoder
        does NOT use sequential greedy insertion.  Instead it separates the
        three independent degrees of freedom and encodes each one with its
        own dedicated chromosome key, guaranteeing that every feasible
        schedule is representable.

        Degrees of freedom and their keys
        ──────────────────────────────────
        role_priority_keys[role_id]   → vessel assignment
            floor(key * |domain|) gives the target domain index.
            C4 (no two roles of the same activity on the same vessel)
            is resolved by processing roles in ascending domain-size order
            and forward-wrapping within the domain when a collision occurs.

        start_time_keys[act_idx]      → intra-vessel sequence ordering
            Activities assigned to the same vessel are sorted by
            start_time_keys ASCENDING.  A lower key means "earlier in
            the vessel's sequence".  Any target ordering is achievable by
            choosing keys with the correct relative magnitude.

        location_keys[maint_counter]  → maintenance location (unchanged)
        speed_keys[role_id]           → travel speed (unchanged)
        start_window_keys[act_idx]    → which possible_start[] slot to use

        Algorithm
        ─────────
        Step 1  Decode locations  (identical to all other decoders)
        Step 2  Anchor ongoing/current activities  (identical)
        Step 3  Assign vessels to future roles via role_priority_keys
                - most-constrained (smallest domain) first
                - forward-wrap on C4 collision
        Step 4  Build each vessel's activity sequence by sorting its
                assigned activities by start_time_keys[act_idx] ascending
        Step 5  Propagate start times through each vessel's sequence:
                - predecessor constraints (C5)
                - travel-time constraints (C6, speed from speed_keys)
                - pick the correct possible_start[] slot (start_window_keys)
                - if no valid slot exists → mark activity unassigned
                  (All-or-Nothing per activity)
        Step 6  Collect results and return DecodedSolution

        Completeness proof (summary)
        ────────────────────────────
        For any feasible schedule S* = (v*, seq*, t*, loc*, spd*):
        • role_priority_keys can be set so floor(key*|dom|) targets v*[r]
          for every role r, with the forward-wrap never needed (keys are
          chosen directly, not derived from a shared activity key).
        • start_time_keys can be set to any strictly-monotone sequence that
          reproduces the ordering in seq* — infinitely many such keys exist.
        • start_window_keys selects t*[a] from possible_start[]; because S*
          is feasible, t*[a] is in the valid window, so the key can target it.
        • Locations and speeds are independently encoded.
        Therefore every feasible schedule is reachable.  □
        """
        # ── Step 1: Decode locations ────────────────────────────────────
        locations = self._decode_locations(chromosome)

        # ── Step 2: Anchor ongoing/current activities ───────────────────
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices = np.full(self.model.num_roles, -1, dtype=int)
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        # Committed start times from anchors
        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = self.model.activities[anchor_idx].possible_start[0]

        future_set = set(future_activities)

        # ── Step 3: Vessel assignment via role_priority_keys ────────────
        # Build activity → future roles map
        activity_to_roles: dict[int, list] = defaultdict(list)
        for r in self.model.roles:
            if r.parent_activity_idx in future_set:
                activity_to_roles[r.parent_activity_idx].append(r)

        # Process each future activity; within each activity, process roles
        # in ascending domain-size order (most-constrained first) to
        # minimise unresolvable C4 conflicts.
        for act_idx in future_activities:
            act_roles = activity_to_roles.get(act_idx, [])
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            vessels_taken_this_act: set[int] = set()

            for role in act_roles_sorted:
                domain = role.vessel_domain
                N_r = len(domain)
                if N_r == 0:
                    continue  # no eligible vessel — stays -1

                # Map role_priority_keys[role_id] → domain index
                key = float(chromosome.role_priority_keys[role.role_id]) # type: ignore
                key = min(max(key, 0.0), 0.999999999)
                domain_idx = min(int(key * N_r), N_r - 1)
                vessel_id = domain[domain_idx]

                # C4: forward-wrap if chosen vessel already taken for this activity
                if vessel_id in vessels_taken_this_act:
                    resolved = False
                    for search_offset in range(1, N_r):
                        candidate_idx = (domain_idx + search_offset) % N_r
                        candidate = domain[candidate_idx]
                        if candidate not in vessels_taken_this_act:
                            vessel_id = candidate
                            resolved = True
                            break
                    if not resolved:
                        continue  # all domain vessels already taken — unassigned

                assigned_vessels[role.role_id] = vessel_id
                speed_indices[role.role_id] = self._decode_speed_index_for_role(
                    chromosome, role.role_id, vessel_id
                )
                vessels_taken_this_act.add(vessel_id)

        # ── Step 4: Build vessel sequences via start_time_keys ──────────
        # Collect (act_idx, start_time_key) for each vessel, sort ascending.
        # Ascending key → earlier position in sequence.
        vessel_act_keys: list[list[tuple[int, float]]] = [
            [] for _ in range(self.model.num_vessels)
        ]
        for act_idx in future_activities:
            act_roles = activity_to_roles.get(act_idx, [])
            # Find which vessel this activity's roles were assigned to.
            # For multi-role activities, all assigned roles must go on their
            # respective vessels; the activity itself appears in each vessel's
            # sequence exactly once.
            vessels_for_act: set[int] = set()
            for role in act_roles:
                v = assigned_vessels[role.role_id]
                if v >= 0:
                    vessels_for_act.add(v) # type: ignore
            key = float(chromosome.start_time_keys[act_idx]) # type: ignore
            for v in vessels_for_act:
                vessel_act_keys[v].append((act_idx, key))

        for v_idx in range(self.model.num_vessels):
            # Sort by key ascending → earlier key = earlier in sequence
            vessel_act_keys[v_idx].sort(key=lambda x: x[1])
            for act_idx, _ in vessel_act_keys[v_idx]:
                vessel_sequences[v_idx].append(act_idx)

        # ── Step 5: Propagate start times ───────────────────────────────
        # Process vessel sequences in topological order w.r.t. predecessor
        # constraints so that predecessor start times are committed before
        # their successors are processed.
        # We use iterative propagation (same spirit as _calculate_start_times)
        # but write into committed_start_times and respect possible_start[].

        # Determine a safe processing order: topological sort of future activities
        # by predecessor dependency, then by vessel sequence position.
        topo_order = self._topological_sort(
            [(a, chromosome.start_time_keys[a]) for a in future_activities], # type: ignore
            [seq[0] for seq in vessel_sequences if seq and seq[0] not in future_set],
        )

        # Track which activities had their start time successfully committed
        committed_ok: set[int] = set(committed_start_times.keys())

        for act_idx in topo_order:
            act = self.model.activities[act_idx]

            # Find which vessel and position this activity sits at
            # (it may appear in multiple vessels for multi-role activities,
            # but start time is shared — we use the first vessel's sequence
            # to determine travel constraints, honouring all vessels' travel).
            roles_for_act = activity_to_roles.get(act_idx, [])

            # Determine minimum required start from predecessor (C5)
            min_start = act.possible_start[0]
            if hasattr(act, 'predecessor_name') and act.predecessor_name:
                pred_idx = self.model.activity_name_to_idx.get(act.predecessor_name, -1)
                if pred_idx != -1:
                    if pred_idx in committed_start_times:
                        pred_end = (committed_start_times[pred_idx]
                                    + self.model.activities[pred_idx].duration)
                        min_start = max(min_start, pred_end)
                    else:
                        # Predecessor not yet committed — use earliest possible
                        pred_act = self.model.activities[pred_idx]
                        pred_end = pred_act.possible_start[0] + pred_act.duration
                        min_start = max(min_start, pred_end)

            # Determine minimum required start from travel (C6)
            # — for every vessel this activity is assigned to, check the
            # previous activity in that vessel's sequence.
            for role in roles_for_act:
                v_idx = assigned_vessels[role.role_id]
                if v_idx < 0:
                    continue
                seq = vessel_sequences[v_idx]
                if act_idx not in seq:
                    continue
                pos = seq.index(act_idx)
                if pos == 0:
                    continue  # first in sequence (after anchor handled separately)
                prev_act_idx = seq[pos - 1]
                prev_act = self.model.activities[prev_act_idx]
                prev_end = committed_start_times.get(
                    prev_act_idx, prev_act.possible_start[0]
                ) + prev_act.duration
                travel = self._travel_time_days_for_role_arc(
                    v_idx=v_idx, # type: ignore
                    prev_act_idx=prev_act_idx,
                    curr_act_idx=act_idx,
                    curr_role_id=role.role_id,
                    locations=locations,
                    chromosome=chromosome,
                )
                min_start = max(min_start, prev_end + travel)

            # Pick the correct possible_start[] slot (start_window_keys)
            valid_starts = [t for t in act.possible_start if t >= min_start]

            if not valid_starts:
                # No valid slot — activity cannot be feasibly scheduled.
                # Mark all its roles as unassigned (All-or-Nothing).
                for role in roles_for_act:
                    assigned_vessels[role.role_id] = -1
                    speed_indices[role.role_id] = -1
                # Remove from vessel sequences
                for role in roles_for_act:
                    v_idx = int(assigned_vessels[role.role_id]) if assigned_vessels[role.role_id] >= 0 else -1
                for v_idx_seq in range(self.model.num_vessels):
                    if act_idx in vessel_sequences[v_idx_seq]:
                        vessel_sequences[v_idx_seq].remove(act_idx)
                continue

            # Use start_window_keys to pick which valid slot
            wk = float(chromosome.start_window_keys[act_idx]) # type: ignore
            wk = min(max(wk, 0.0), 0.999999999)
            chosen_idx = min(int(wk * len(valid_starts)), len(valid_starts) - 1)
            committed_start_times[act_idx] = valid_starts[chosen_idx]
            committed_ok.add(act_idx)

        # ── Step 6: Finalise ────────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices,
        )
    
    def _decode_permkey(self, chromosome: Chromosome) -> DecodedSolution:
        """
        BRKGA-PERMKEY — Complete AND nearly-always-feasible decoder.

        Motivation
        ──────────
        The standard greedy decoders (STT, BRKGA-A, GRASP) are nearly always
        feasible because they only commit decisions that pass hard feasibility
        checks.  However, they are INCOMPLETE: they select the *cheapest*
        feasible option, so many chromosomes map to the same solution and
        large regions of the search space are unreachable.

        BRKGA-DIRECT is complete (every feasible schedule is reachable) but
        produces almost exclusively infeasible solutions because it assigns
        vessels and builds sequences globally, then checks feasibility
        afterwards — giving no guarantee that the sequence is satisfiable.

        BRKGA-PERMKEY resolves this tension with one key insight:

            At every decision point, ENUMERATE the feasible options and use
            the chromosome key to SELECT one of them by index.

        This guarantees:
        (a) FEASIBILITY  — only feasible options are ever selected.
        (b) COMPLETENESS — for any feasible schedule S*, a chromosome exists
            that reproduces it (proved below).

        Algorithm
        ---------
        Step 1  Decode maintenance locations
        Step 2  Anchor ongoing/current activities
        Step 3  Determine activity processing order from start_time_keys
                (descending, then topological sort).
        Step 4  For each activity in processing order:
                  For each role (ascending domain size, most-constrained first):
                    Enumerate ALL feasible (vessel, position) pairs.
                    Sort them by (vessel_index, position) for a stable,
                    deterministic ordering.
                    Use role_priority_keys[role_id] to pick one by index:
                        idx = floor(key x |feasible_set|)
                    If feasible_set is empty → activity fails (All-or-Nothing).
                  Atomically commit all roles of this activity.

        Multi-role consistency
        ----------------------
        For activities with multiple roles on different vessels, the first
        role's feasibility check determines the activity start time.
        Subsequent roles for the same activity see this pinned start time
        via `activity_pinned_starts`, ensuring all vessels agree on when
        the activity begins.  Candidates whose start time for act_idx
        disagrees with the pinned time are filtered out.

        Completeness proof
        ------------------
        Let S* = (v*, seq*, t*, loc*, spd*) be any feasible schedule.

        (i) Set start_time_keys to encode an S*-consistent processing order:
            assign key[A] > key[B] whenever A must be processed before B,
            i.e., whenever A appears before B in seq*[v] for any vessel v,
            OR A is a predecessor of B.  Such keys always exist because S*
            is feasible (no circular dependencies).

        (ii) When activity A is processed, all activities that precede A in
             every vessel's sequence under S* are already committed.  Therefore
             the target (v*[r], position_of_A_in_seq*[v*[r]]) is a valid
             feasible option for role r.

        (iii) The feasible set for role r is non-empty (contains the target).
              Set role_priority_keys[r] = (target_rank + 0.5) / |feasible_set|
              where target_rank is the index of the target pair in the sorted
              feasible set.  Then floor(key x |feasible_set|) == target_rank. 

        (iv) For multi-role activities, S* assigns the same start time to all
             roles.  The first role pins that time; subsequent roles' feasible
             sets still contain the target because S* is feasible on every
             vessel at that start time.

        (v) Locations, speeds are independently encoded (unchanged).

        Therefore every feasible schedule is reachable.

        Near-complete feasibility
        -------------------------
        Infeasibility only occurs when the feasible set is empty for some role,
        meaning no (vessel, position) pair satisfies all hard constraints given
        the activities already committed.  This mirrors the STT decoder (which
        is widely accepted as "nearly always feasible") but uses key-based
        selection instead of cost-based, giving a richer search landscape.

        Key usage
        ---------
        start_time_keys[act_idx]    processing priority (desc) + topo sort
        role_priority_keys[role_id] index into sorted feasible (v, pos) set
        location_keys               maintenance location (unchanged)
        speed_keys                  travel speed (unchanged)
        start_window_keys[act_idx]  which possible_start[] slot to use
        """
        # ── Step 1: Decode locations ────────────────────────────────────────
        locations = self._decode_locations(chromosome)

        # ── Step 2: Anchor ongoing/current activities ───────────────────────
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices    = np.full(self.model.num_roles, -1, dtype=int)
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = (
                    self.model.activities[anchor_idx].possible_start[0]
                )

        # ── Step 3: Determine processing order ─────────────────────────────
        activity_priorities = [
            (a_idx, chromosome.start_time_keys[a_idx]) for a_idx in future_activities # type: ignore
        ]
        sorted_acts = sorted(activity_priorities, key=lambda x: x[1], reverse=True)
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]
        ordered_act_indices = self._topological_sort(sorted_acts, anchored_indices)

        # Fast map: activity → roles
        activity_to_roles: dict[int, list] = defaultdict(list)
        future_set = set(future_activities)
        for r in self.model.roles:
            if r.parent_activity_idx in future_set:
                activity_to_roles[r.parent_activity_idx].append(r)

        # ── Step 4: Key-indexed feasible-set selection ──────────────────────
        for act_idx in ordered_act_indices:
            act_roles = activity_to_roles.get(act_idx, [])
            if not act_roles:
                continue

            # Most-constrained role first (smallest vessel domain)
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            temp_assignments: list[dict] = []
            planned_vessels_for_this_act: set[int] = set()
            activity_feasible = True

            # Track the pinned start time for this activity across
            #    all its roles.  The first role's chosen candidate determines
            #    the start time; subsequent roles must agree with it. 
            activity_pinned_starts: dict[int, int] = {}

            for role in act_roles_sorted:
                # ── Enumerate ALL feasible (vessel, position) pairs ─────────
                feasible_candidates: list[tuple[int, int, dict]] = []
                # tuple: (v_idx, pos, temp_start_times)

                # Merge globally committed times with the pinned start time
                # from earlier roles of this same activity, so that the
                # feasibility check sees the already-decided start time.
                merged_start_times = {**committed_start_times, **activity_pinned_starts}

                for v_idx in role.vessel_domain:
                    if v_idx in planned_vessels_for_this_act:
                        continue  # C4: same activity cannot reuse a vessel

                    seq = vessel_sequences[v_idx]
                    for pos in range(1, len(seq) + 1):
                        is_feasible, temp_starts = self._is_insertion_feasible(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            current_start_times=merged_start_times,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )
                        if not is_feasible:
                            continue

                        # If a previous role already pinned the start
                        #    time for act_idx, only accept candidates that
                        #    agree with that pinned time. 
                        if act_idx in activity_pinned_starts:
                            candidate_start = temp_starts.get(act_idx)
                            if candidate_start != activity_pinned_starts[act_idx]:
                                continue  # Start time mismatch — skip

                        feasible_candidates.append((v_idx, pos, temp_starts))

                if not feasible_candidates:
                    activity_feasible = False
                    break  # All-or-Nothing: abandon this activity

                # ── Deterministic sort for stable indexing ──────────────────
                # Sort by (vessel_index, position) — arbitrary but fixed order
                # so the same key always maps to the same candidate.
                feasible_candidates.sort(key=lambda c: (c[0], c[1]))

                # ── Key-indexed selection ───────────────────────────────────
                key = float(chromosome.role_priority_keys[role.role_id]) # type: ignore
                key = min(max(key, 0.0), 0.999999999)
                n   = len(feasible_candidates)
                idx = min(int(key * n), n - 1)

                chosen_v, chosen_pos, chosen_starts = feasible_candidates[idx]

                temp_assignments.append({
                    "role_id":  role.role_id,
                    "v_idx":    chosen_v,
                    "pos":      chosen_pos,
                    "starts":   chosen_starts,
                })
                planned_vessels_for_this_act.add(chosen_v)

                # Pin the start time for act_idx so that subsequent
                #    roles of this activity are forced to use the same time.
                if act_idx not in activity_pinned_starts and act_idx in chosen_starts:
                    activity_pinned_starts[act_idx] = chosen_starts[act_idx]

            # ── Atomic commit (All-or-Nothing) ──────────────────────────────
            if activity_feasible:
                for a in temp_assignments:
                    rid   = a["role_id"]
                    v_idx = a["v_idx"]
                    assigned_vessels[rid] = v_idx
                    speed_indices[rid]    = self._decode_speed_index_for_role(
                        chromosome, rid, v_idx
                    )
                    vessel_sequences[v_idx].insert(a["pos"], act_idx)
                    committed_start_times.update(a["starts"])
            # else: all roles for this activity stay at -1 (penalty applied later)

        # ── Finalise start times ────────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices,
        )

    def _decode_permkey_urgency(self, chromosome: Chromosome) -> DecodedSolution:
        """
        BRKGA-PERMKEY-URGENCY — identical to BRKGA-PERMKEY except that the
        activity processing order (Step 3) is determined by a static urgency
        score instead of start_time_keys.

        Urgency = (1 / window_size) + (1 / total_domain_size) + predecessor_bonus,
        where a higher score means the activity is harder to schedule and should
        be processed first.  start_time_keys are not used for ordering; they are
        still used by _is_insertion_feasible (via the chromosome object) for
        start-window selection, unchanged.
        """
        # ── Step 1: Decode locations ────────────────────────────────────────
        locations = self._decode_locations(chromosome)

        # ── Step 2: Anchor ongoing/current activities ───────────────────────
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices    = np.full(self.model.num_roles, -1, dtype=int)
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = (
                    self.model.activities[anchor_idx].possible_start[0]
                )

        # Fast map: activity → roles (built before urgency so domain sizes are available)
        activity_to_roles: dict[int, list] = defaultdict(list)
        future_set = set(future_activities)
        for r in self.model.roles:
            if r.parent_activity_idx in future_set:
                activity_to_roles[r.parent_activity_idx].append(r)

        # ── Step 3: Determine processing order via urgency score ────────────
        urgency_scores = []
        for a_idx in future_activities:
            act = self.model.activities[a_idx]
            act_roles_for_u = activity_to_roles.get(a_idx, [])
            w1 = 1.0 / max(len(act.possible_start), 1)
            w2 = 1.0 / max(sum(len(r.vessel_domain) for r in act_roles_for_u), 1)
            has_pred_bonus = 0.1 if getattr(act, 'predecessor_name', '') else 0.0
            urgency_scores.append((a_idx, w1 + w2 + has_pred_bonus))

        sorted_acts = sorted(urgency_scores, key=lambda x: x[1], reverse=True)
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]
        ordered_act_indices = self._topological_sort(sorted_acts, anchored_indices)

        # ── Step 4: Key-indexed feasible-set selection ──────────────────────
        for act_idx in ordered_act_indices:
            act_roles = activity_to_roles.get(act_idx, [])
            if not act_roles:
                continue

            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            temp_assignments: list[dict] = []
            planned_vessels_for_this_act: set[int] = set()
            activity_feasible = True
            activity_pinned_starts: dict[int, int] = {}

            for role in act_roles_sorted:
                feasible_candidates: list[tuple[int, int, dict]] = []
                merged_start_times = {**committed_start_times, **activity_pinned_starts}

                for v_idx in role.vessel_domain:
                    if v_idx in planned_vessels_for_this_act:
                        continue

                    seq = vessel_sequences[v_idx]
                    for pos in range(1, len(seq) + 1):
                        is_feasible, temp_starts = self._is_insertion_feasible(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            current_start_times=merged_start_times,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )
                        if not is_feasible:
                            continue

                        if act_idx in activity_pinned_starts:
                            candidate_start = temp_starts.get(act_idx)
                            if candidate_start != activity_pinned_starts[act_idx]:
                                continue

                        feasible_candidates.append((v_idx, pos, temp_starts))

                if not feasible_candidates:
                    activity_feasible = False
                    break

                feasible_candidates.sort(key=lambda c: (c[0], c[1]))

                key = float(chromosome.role_priority_keys[role.role_id]) # type: ignore
                key = min(max(key, 0.0), 0.999999999)
                n   = len(feasible_candidates)
                idx = min(int(key * n), n - 1)

                chosen_v, chosen_pos, chosen_starts = feasible_candidates[idx]

                temp_assignments.append({
                    "role_id":  role.role_id,
                    "v_idx":    chosen_v,
                    "pos":      chosen_pos,
                    "starts":   chosen_starts,
                })
                planned_vessels_for_this_act.add(chosen_v)

                if act_idx not in activity_pinned_starts and act_idx in chosen_starts:
                    activity_pinned_starts[act_idx] = chosen_starts[act_idx]

            if activity_feasible:
                for a in temp_assignments:
                    rid   = a["role_id"]
                    v_idx = a["v_idx"]
                    assigned_vessels[rid] = v_idx
                    speed_indices[rid]    = self._decode_speed_index_for_role(
                        chromosome, rid, v_idx
                    )
                    vessel_sequences[v_idx].insert(a["pos"], act_idx)
                    committed_start_times.update(a["starts"])

        # ── Finalise start times ────────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices,
        )

    def _decode_permkey_tows_first(self, chromosome: Chromosome) -> DecodedSolution:
        """
        BRKGA-PERMKEY-TOWS-FIRST — two sequential PERMKEY insertion passes over
        shared state.

        Pass 1 inserts tow activities (is_tow_type(), excluding is_current_location)
        using urgency-based ordering.  Pass 2 inserts maintenance activities
        (is_maintenance_type()) into the vessel sequences already updated by Pass 1,
        also using urgency-based ordering.  All state (vessel_sequences,
        assigned_vessels, committed_start_times) is shared; there is no reset
        between passes.  Maintenance activities therefore see all committed tow
        assignments and must insert around them.
        """
        # ── Step 1: Decode locations ────────────────────────────────────────
        locations = self._decode_locations(chromosome)

        # ── Step 2: Anchor ongoing/current activities ───────────────────────
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices    = np.full(self.model.num_roles, -1, dtype=int)
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = (
                    self.model.activities[anchor_idx].possible_start[0]
                )

        # Fast map: activity → roles
        activity_to_roles: dict[int, list] = defaultdict(list)
        future_set = set(future_activities)
        for r in self.model.roles:
            if r.parent_activity_idx in future_set:
                activity_to_roles[r.parent_activity_idx].append(r)

        # Initial anchored indices (current/ongoing) for topo sort seeding
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]

        # ── Shared urgency helper ───────────────────────────────────────────
        def _urgency(a_idx: int) -> float:
            act = self.model.activities[a_idx]
            act_roles_for_u = activity_to_roles.get(a_idx, [])
            w1 = 1.0 / max(len(act.possible_start), 1)
            w2 = 1.0 / max(sum(len(r.vessel_domain) for r in act_roles_for_u), 1)
            has_pred_bonus = 0.1 if getattr(act, 'predecessor_name', '') else 0.0
            return w1 + w2 + has_pred_bonus

        # ── Shared insertion loop ───────────────────────────────────────────
        def _run_insertion_pass(ordered_indices: list[int]) -> list[int]:
            """Run the PERMKEY feasible-set loop over ordered_indices.
            Returns the list of activity indices that were successfully committed."""
            committed = []
            for act_idx in ordered_indices:
                act_roles = activity_to_roles.get(act_idx, [])
                if not act_roles:
                    continue

                act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

                temp_assignments: list[dict] = []
                planned_vessels_for_this_act: set[int] = set()
                activity_feasible = True
                activity_pinned_starts: dict[int, int] = {}

                for role in act_roles_sorted:
                    feasible_candidates: list[tuple[int, int, dict]] = []
                    merged_start_times = {**committed_start_times, **activity_pinned_starts}

                    for v_idx in role.vessel_domain:
                        if v_idx in planned_vessels_for_this_act:
                            continue

                        seq = vessel_sequences[v_idx]
                        for pos in range(1, len(seq) + 1):
                            is_feasible, temp_starts = self._is_insertion_feasible(
                                v_idx=v_idx,
                                target_act_idx=act_idx,
                                target_role_id=role.role_id,
                                pos=pos,
                                seq=seq,
                                locations=locations,
                                current_start_times=merged_start_times,
                                assigned_vessels=assigned_vessels,
                                chromosome=chromosome,
                            )
                            if not is_feasible:
                                continue

                            if act_idx in activity_pinned_starts:
                                candidate_start = temp_starts.get(act_idx)
                                if candidate_start != activity_pinned_starts[act_idx]:
                                    continue

                            feasible_candidates.append((v_idx, pos, temp_starts))

                    if not feasible_candidates:
                        activity_feasible = False
                        break

                    feasible_candidates.sort(key=lambda c: (c[0], c[1]))

                    key = float(chromosome.role_priority_keys[role.role_id]) # type: ignore
                    key = min(max(key, 0.0), 0.999999999)
                    n   = len(feasible_candidates)
                    idx = min(int(key * n), n - 1)

                    chosen_v, chosen_pos, chosen_starts = feasible_candidates[idx]

                    temp_assignments.append({
                        "role_id":  role.role_id,
                        "v_idx":    chosen_v,
                        "pos":      chosen_pos,
                        "starts":   chosen_starts,
                    })
                    planned_vessels_for_this_act.add(chosen_v)

                    if act_idx not in activity_pinned_starts and act_idx in chosen_starts:
                        activity_pinned_starts[act_idx] = chosen_starts[act_idx]

                if activity_feasible:
                    committed.append(act_idx)
                    for a in temp_assignments:
                        rid   = a["role_id"]
                        v_idx = a["v_idx"]
                        assigned_vessels[rid] = v_idx
                        speed_indices[rid]    = self._decode_speed_index_for_role(
                            chromosome, rid, v_idx
                        )
                        vessel_sequences[v_idx].insert(a["pos"], act_idx)
                        committed_start_times.update(a["starts"])

            return committed

        # ── Pass 1: Tow activities ──────────────────────────────────────────
        tow_indices = [
            a_idx for a_idx in future_activities
            if self.model.activities[a_idx].is_tow_type()
            and not self.model.activities[a_idx].is_current_location
        ]
        tow_urgency = [(a_idx, _urgency(a_idx)) for a_idx in tow_indices]
        sorted_tow = sorted(tow_urgency, key=lambda x: x[1], reverse=True)
        ordered_tow_indices = self._topological_sort(sorted_tow, anchored_indices)

        committed_tow_indices = _run_insertion_pass(ordered_tow_indices)

        # ── Pass 2: Maintenance activities ──────────────────────────────────
        maint_indices = [
            a_idx for a_idx in future_activities
            if self.model.activities[a_idx].is_maintenance_type()
        ]
        maint_urgency = [(a_idx, _urgency(a_idx)) for a_idx in maint_indices]
        sorted_maint = sorted(maint_urgency, key=lambda x: x[1], reverse=True)
        # Pass 2 topo sort seeds include initial anchors AND all committed tow activities
        anchored_indices_pass2 = anchored_indices + committed_tow_indices
        ordered_maint_indices = self._topological_sort(sorted_maint, anchored_indices_pass2)

        _run_insertion_pass(ordered_maint_indices)

        # ── Finalise start times ────────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices,
        )

    def _decode_permkey1(self, chromosome: Chromosome) -> DecodedSolution:
        """
        BRKGA-PERMKEY — Complete AND nearly-always-feasible decoder.

        Motivation
        ──────────
        The standard greedy decoders (STT, BRKGA-A, GRASP) are nearly always
        feasible because they only commit decisions that pass hard feasibility
        checks.  However, they are INCOMPLETE: they select the *cheapest*
        feasible option, so many chromosomes map to the same solution and
        large regions of the search space are unreachable.

        BRKGA-DIRECT is complete (every feasible schedule is reachable) but
        produces almost exclusively infeasible solutions because it assigns
        vessels and builds sequences globally, then checks feasibility
        afterwards — giving no guarantee that the sequence is satisfiable.

        BRKGA-PERMKEY resolves this tension with one key insight:

            At every decision point, ENUMERATE the feasible options and use
            the chromosome key to SELECT one of them by index.

        This guarantees:
        (a) FEASIBILITY  — only feasible options are ever selected.
        (b) COMPLETENESS — for any feasible schedule S*, a chromosome exists
            that reproduces it (proved below).

        Algorithm
        ─────────
        Step 1  Decode maintenance locations (identical to all decoders).
        Step 2  Anchor ongoing/current activities (identical).
        Step 3  Determine activity processing order from start_time_keys
                (descending, then topological sort — identical to STT).
        Step 4  For each activity in processing order:
                  For each role (ascending domain size, most-constrained first):
                    Enumerate ALL feasible (vessel, position) pairs.
                    Sort them by (vessel_index, position) for a stable,
                    deterministic ordering.
                    Use role_priority_keys[role_id] to pick one by index:
                        idx = floor(key × |feasible_set|)
                    If feasible_set is empty → activity fails (All-or-Nothing).
                  Atomically commit all roles of this activity.

        Completeness proof
        ──────────────────
        Let S* = (v*, seq*, t*, loc*, spd*) be any feasible schedule.

        (i) Set start_time_keys to encode an S*-consistent processing order:
            assign key[A] > key[B] whenever A must be processed before B,
            i.e., whenever A appears before B in seq*[v] for any vessel v,
            OR A is a predecessor of B.  Such keys always exist because S*
            is feasible (no circular dependencies).

        (ii) When activity A is processed, all activities that precede A in
             every vessel's sequence under S* are already committed.  Therefore
             the target (v*[r], position_of_A_in_seq*[v*[r]]) is a valid
             feasible option for role r.

        (iii) The feasible set for role r is non-empty (contains the target).
              Set role_priority_keys[r] = (target_rank + 0.5) / |feasible_set|
              where target_rank is the index of the target pair in the sorted
              feasible set.  Then floor(key × |feasible_set|) == target_rank. ✓

        (iv) Locations, speeds are independently encoded (unchanged). ✓

        Therefore every feasible schedule is reachable.  □

        Near-complete feasibility
        ─────────────────────────
        Infeasibility only occurs when the feasible set is empty for some role,
        meaning no (vessel, position) pair satisfies all hard constraints given
        the activities already committed.  This mirrors the STT decoder (which
        is widely accepted as "nearly always feasible") but uses key-based
        selection instead of cost-based, giving a richer search landscape.

        Key usage
        ─────────
        start_time_keys[act_idx]    processing priority (desc) + topo sort
        role_priority_keys[role_id] index into sorted feasible (v, pos) set
        location_keys               maintenance location (unchanged)
        speed_keys                  travel speed (unchanged)
        start_window_keys[act_idx]  which possible_start[] slot to use
        """
        # ── Step 1: Decode locations ────────────────────────────────────────
        locations = self._decode_locations(chromosome)

        # ── Step 2: Anchor ongoing/current activities ───────────────────────
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices    = np.full(self.model.num_roles, -1, dtype=int)
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = (
                    self.model.activities[anchor_idx].possible_start[0]
                )

        # ── Step 3: Determine processing order ─────────────────────────────
        activity_priorities = [
            (a_idx, chromosome.start_time_keys[a_idx]) for a_idx in future_activities # type: ignore
        ]
        sorted_acts = sorted(activity_priorities, key=lambda x: x[1], reverse=True)
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]
        ordered_act_indices = self._topological_sort(sorted_acts, anchored_indices)

        # Fast map: activity → roles
        activity_to_roles: dict[int, list] = defaultdict(list)
        future_set = set(future_activities)
        for r in self.model.roles:
            if r.parent_activity_idx in future_set:
                activity_to_roles[r.parent_activity_idx].append(r)

        # ── Step 4: Key-indexed feasible-set selection ──────────────────────
        for act_idx in ordered_act_indices:
            act_roles = activity_to_roles.get(act_idx, [])
            if not act_roles:
                continue

            # Most-constrained role first (smallest vessel domain)
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            temp_assignments: list[dict] = []
            planned_vessels_for_this_act: set[int] = set()
            activity_feasible = True

            for role in act_roles_sorted:
                # ── Enumerate ALL feasible (vessel, position) pairs ─────────
                feasible_candidates: list[tuple[int, int, dict]] = []
                # tuple: (v_idx, pos, temp_start_times)

                for v_idx in role.vessel_domain:
                    if v_idx in planned_vessels_for_this_act:
                        continue  # C4: same activity cannot reuse a vessel

                    seq = vessel_sequences[v_idx]
                    for pos in range(1, len(seq) + 1):
                        is_feasible, temp_starts = self._is_insertion_feasible(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            current_start_times=committed_start_times,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )
                        if is_feasible:
                            feasible_candidates.append((v_idx, pos, temp_starts))

                if not feasible_candidates:
                    activity_feasible = False
                    break  # All-or-Nothing: abandon this activity

                # ── Deterministic sort for stable indexing ──────────────────
                # Sort by (vessel_index, position) — arbitrary but fixed order
                # so the same key always maps to the same candidate.
                feasible_candidates.sort(key=lambda c: (c[0], c[1]))

                # ── Key-indexed selection ───────────────────────────────────
                key = float(chromosome.role_priority_keys[role.role_id]) # type: ignore
                key = min(max(key, 0.0), 0.999999999)
                n   = len(feasible_candidates)
                idx = min(int(key * n), n - 1)

                chosen_v, chosen_pos, chosen_starts = feasible_candidates[idx]

                temp_assignments.append({
                    "role_id":  role.role_id,
                    "v_idx":    chosen_v,
                    "pos":      chosen_pos,
                    "starts":   chosen_starts,
                })
                planned_vessels_for_this_act.add(chosen_v)

            # ── Atomic commit (All-or-Nothing) ──────────────────────────────
            if activity_feasible:
                for a in temp_assignments:
                    rid   = a["role_id"]
                    v_idx = a["v_idx"]
                    assigned_vessels[rid] = v_idx
                    speed_indices[rid]    = self._decode_speed_index_for_role(
                        chromosome, rid, v_idx
                    )
                    vessel_sequences[v_idx].insert(a["pos"], act_idx)
                    committed_start_times.update(a["starts"])
            # else: all roles for this activity stay at -1 (penalty applied later)

        # ── Finalise start times ────────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices,
        )
        
    def _decode_grasp(self, chromosome: Chromosome) -> DecodedSolution:
        """
        GRASP-style decoder (deterministic, chromosome-driven).

        Construction:
        1) Decode maintenance locations.
        2) Anchor ongoing/current activities.
        3) Process future activities in topological order guided by start_time_keys.
        4) For each role, evaluate feasible (vessel, position) insertions.
            Build an RCL from best candidates and pick via key-mapped index.

        Local Search (bounded):
        - Intra-vessel relocate for activities in future set.
        - Accept first improving feasible move on added travel proxy.
        - Deterministic scan order and tie-breaking.

        Returns fully feasible schedule.
        """
        # ---------------------------
        # Tunable GRASP parameters
        # ---------------------------
        ALPHA = 0.60         # RCL quality threshold in [0,1]
        MAX_LS_PASSES = 2     # bounded LS passes
        MAX_RCL_SIZE = 8      # cap for speed / stability

        locations = self._decode_locations(chromosome)
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices = np.full(self.model.num_roles, -1, dtype=int)

        # Anchor ongoing/current activities
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = self.model.activities[anchor_idx].possible_start[0]

        # Priority + topo sort (same spirit as STT)
        activity_priorities = [(a_idx, chromosome.start_time_keys[a_idx]) for a_idx in future_activities] # type: ignore
        sorted_acts = sorted(activity_priorities, key=lambda x: x[1], reverse=True)
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]
        ordered_act_indices = self._topological_sort(sorted_acts, anchored_indices)

        # Fast map: activity -> roles
        activity_to_roles: dict[int, list] = defaultdict(list)
        for r in self.model.roles:
            if r.parent_activity_idx in set(future_activities):
                activity_to_roles[r.parent_activity_idx].append(r)

        # ---------------------------
        # GRASP construction
        # ---------------------------
        for act_idx in ordered_act_indices:
            act_roles = activity_to_roles.get(act_idx, [])
            if not act_roles:
                continue

            # C4-friendly: constrained roles first
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            # temporary plan for all roles of this activity (atomic commit)
            temp_assignments = []
            planned_vessels_for_this_act: set[int] = set()

            activity_feasible = True

            for role in act_roles_sorted:
                candidates = []  # list[(cost, v_idx, pos, temp_starts)]

                # Build candidate list
                for v_idx in role.vessel_domain:
                    if v_idx in planned_vessels_for_this_act:
                        continue  # C4 inside activity

                    seq = vessel_sequences[v_idx]
                    # insert after anchor position 0 if anchor exists; allow pos=1..len(seq)
                    for pos in range(1, len(seq) + 1):
                        feasible, temp_starts = self._is_insertion_feasible(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            current_start_times=committed_start_times,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )
                        if not feasible:
                            continue

                        greedy_score = self._calculate_added_cost(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                            committed_start_times=committed_start_times,
                        )

                        # small deterministic tie breaker
                        tie = (v_idx * 1e-6) + (pos * 1e-7)
                        cost = float(greedy_score) + tie
                        candidates.append((cost, v_idx, pos, temp_starts))

                if not candidates:
                    activity_feasible = False
                    break

                # Sort candidates by greedy score
                candidates.sort(key=lambda x: x[0])

                # Build RCL by alpha-threshold + cap
                c_min = candidates[0][0]
                c_max = candidates[-1][0]
                threshold = c_min + ALPHA * (c_max - c_min)
                rcl = [c for c in candidates if c[0] <= threshold]
                if len(rcl) > MAX_RCL_SIZE:
                    rcl = rcl[:MAX_RCL_SIZE]

                # Deterministic "randomized" pick using chromosome keys
                # combine role key + activity key for stable spread
                pick_key = 0.5 * float(chromosome.role_priority_keys[role.role_id]) + 0.5 * float(chromosome.start_time_keys[act_idx]) # type: ignore
                pick_key = min(max(pick_key, 0.0), 0.999999999)
                idx = min(int(pick_key * len(rcl)), len(rcl) - 1)

                chosen = rcl[idx]
                _, best_v_idx, best_pos, best_temp_starts = chosen

                temp_assignments.append({
                    "role_id": role.role_id,
                    "v_idx": best_v_idx,
                    "pos": best_pos,
                    "starts": best_temp_starts
                })
                planned_vessels_for_this_act.add(best_v_idx)

            # Atomic commit (same policy as STT)
            if activity_feasible:
                for a in temp_assignments:
                    rid = a["role_id"]
                    v_idx = a["v_idx"]
                    assigned_vessels[rid] = v_idx
                    speed_indices[rid] = self._decode_speed_index_for_role(chromosome, rid, v_idx)
                    vessel_sequences[v_idx].insert(a["pos"], act_idx)
                    committed_start_times.update(a["starts"])
            # else: drop activity (roles remain -1)

        # ---------------------------
        # Local Search (bounded, deterministic)
        # Intra-vessel relocate moves for future activities only
        # ---------------------------
        future_set = set(future_activities)

        for _ in range(MAX_LS_PASSES):
            improved = False

            for v_idx in range(self.model.num_vessels):
                seq = vessel_sequences[v_idx]
                if len(seq) <= 2:
                    continue

                base_cost = self._sequence_proxy_cost_for_objective(
                    v_idx, seq, locations, assigned_vessels, chromosome, committed_start_times
                )

                for from_pos in range(1, len(seq)):
                    act_idx = seq[from_pos]
                    if act_idx not in future_set:
                        continue

                    role_id = self._select_role_id_for_activity_vessel(act_idx, v_idx, assigned_vessels)
                    if role_id == -1:
                        continue

                    reduced = seq[:from_pos] + seq[from_pos + 1:]

                    for to_pos in range(1, len(reduced) + 1):
                        if to_pos == from_pos:
                            continue

                        feasible, temp_starts = self._is_insertion_feasible(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role_id,
                            pos=to_pos,
                            seq=reduced,
                            locations=locations,
                            current_start_times=committed_start_times,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )
                        if not feasible:
                            continue

                        trial_seq = reduced.copy()
                        trial_seq.insert(to_pos, act_idx)
                        new_cost = self._sequence_proxy_cost_for_objective(
                            v_idx, trial_seq, locations, assigned_vessels, chromosome, committed_start_times
                        )

                        if new_cost + 1e-9 < base_cost:
                            vessel_sequences[v_idx] = trial_seq
                            committed_start_times.update(temp_starts)
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        # Phase 2: Inter-vessel relocate
        if self.model.objective in ('cost', 'fuel'):
            self._inter_vessel_relocate(
                vessel_sequences, assigned_vessels, committed_start_times,
                locations, chromosome, future_set, max_passes=MAX_LS_PASSES
            )

        # ---------------------------
        # Finalize start times
        # ---------------------------
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)

        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices
        )

    def _decode_locations(self, chromosome: Chromosome) -> np.ndarray:
        """
        Decodes location_keys into specific local location indices for maintenance activities.
        Returns an array of size `num_activities`. Non-maintenance activities are set to -1.
        """
        locations = np.full(self.model.num_activities, -1, dtype=int)
        maint_counter = 0
        
        for act_idx, act in enumerate(self.model.activities):
            if act.is_maintenance_type():
                loc_key = chromosome.location_keys[maint_counter] # type: ignore
                num_locs = len(act.allowed_locations)
                
                # Scale [0, 1) to [0, num_locs - 1]
                chosen_idx = min(int(loc_key * num_locs), num_locs - 1)
                locations[act_idx] = chosen_idx
                maint_counter += 1                
        return locations
    
    def _inter_vessel_relocate(
        self,
        vessel_sequences: list[list[int]],
        assigned_vessels: np.ndarray,
        committed_start_times: dict,
        locations: np.ndarray,
        chromosome: Chromosome,
        future_set: set[int],
        max_passes: int = 1,
    ) -> None:
        """
        Deterministic inter-vessel relocate for single-role future activities.
        Fully reproducible — no random calls, fixed scan order.
        """
        # Pre-compute single-role activities
        act_role_map: dict[int, list[int]] = defaultdict(list)
        for r in self.model.roles:
            act_role_map[r.parent_activity_idx].append(r.role_id)
        single_role_acts = {a for a, rids in act_role_map.items() if len(rids) == 1}

        for _ in range(max_passes):
            improved = False

            # Deterministic scan: vessel 0..V, position 1..len(seq)
            for src_v in range(self.model.num_vessels):
                src_seq = vessel_sequences[src_v]
                if len(src_seq) <= 1:
                    continue

                for src_pos in range(1, len(src_seq)):
                    act_idx = src_seq[src_pos]
                    if act_idx not in future_set or act_idx not in single_role_acts:
                        continue

                    role_id = act_role_map[act_idx][0]
                    role = self.model.roles[role_id]

                    src_cost = self._sequence_proxy_cost_for_objective(
                        src_v, src_seq, locations, assigned_vessels,
                        chromosome, committed_start_times
                    )

                    reduced_src = src_seq[:src_pos] + src_seq[src_pos + 1:]
                    new_src_cost = self._sequence_proxy_cost_for_objective(
                        src_v, reduced_src, locations, assigned_vessels,
                        chromosome, committed_start_times
                    )
                    removal_saving = src_cost - new_src_cost

                    best_net = 0.0
                    best_dst_v = -1
                    best_dst_pos = -1
                    best_starts: dict = {}

                    # Deterministic scan: destination vessels in index order
                    for dst_v in role.vessel_domain:
                        if dst_v == src_v:
                            continue

                        dst_seq = vessel_sequences[dst_v]
                        old_dst_cost = self._sequence_proxy_cost_for_objective(
                            dst_v, dst_seq, locations, assigned_vessels,
                            chromosome, committed_start_times
                        )

                        for dst_pos in range(1, len(dst_seq) + 1):
                            feasible, temp_starts = self._is_insertion_feasible(
                                v_idx=dst_v,
                                target_act_idx=act_idx,
                                target_role_id=role_id,
                                pos=dst_pos,
                                seq=dst_seq,
                                locations=locations,
                                current_start_times=committed_start_times,
                                assigned_vessels=assigned_vessels,
                                chromosome=chromosome,
                            )
                            if not feasible:
                                continue

                            trial = dst_seq.copy()
                            trial.insert(dst_pos, act_idx)
                            new_dst_cost = self._sequence_proxy_cost_for_objective(
                                dst_v, trial, locations, assigned_vessels,
                                chromosome, committed_start_times
                            )
                            net = removal_saving - (new_dst_cost - old_dst_cost)

                            if net > best_net + 1e-9:
                                best_net = net
                                best_dst_v = dst_v
                                best_dst_pos = dst_pos
                                best_starts = temp_starts

                    if best_dst_v != -1:
                        vessel_sequences[src_v] = reduced_src
                        vessel_sequences[best_dst_v].insert(best_dst_pos, act_idx)
                        assigned_vessels[role_id] = best_dst_v
                        committed_start_times.update(best_starts)
                        improved = True
                        break  # restart scan

                if improved:
                    break

            if not improved:
                break

    def _sequence_proxy_cost_for_objective(
        self, v_idx: int, seq: list[int], 
        locations: np.ndarray, assigned_vessels: np.ndarray,
        chromosome: Chromosome, committed_start_times: dict
    ) -> float:
        """
        Proxy cost for a vessel sequence, aligned with the model's objective.
        """
        if len(seq) < 2:
            return 0.0

        vessel = self.model.vessels[v_idx]
        total = 0.0

        for i in range(1, len(seq)):
            prev_a = seq[i - 1]
            curr_a = seq[i]
            curr_role_id = self._select_role_id_for_activity_vessel(curr_a, v_idx, assigned_vessels)
            if curr_role_id == -1:
                continue

            prev_act = self.model.activities[prev_a]
            curr_act = self.model.activities[curr_a]

            prev_end_loc = self.model._get_global_loc_id(prev_act, locations[prev_a], False)
            curr_start_loc = self.model._get_global_loc_id(curr_act, locations[curr_a], True)

            dist = float(self.model.vessel_travel_dist_matrix_float[v_idx][prev_end_loc, curr_start_loc])
            speed_idx = self._decode_speed_index_for_role(chromosome, curr_role_id, v_idx)
            speed_kn = float(vessel.possible_speeds[speed_idx])

            if dist <= 0 or speed_kn <= 0:
                continue

            travel_days = math.ceil(dist / (speed_kn * 24.0))

            if self.model.objective == 'distance':
                total += dist

            elif self.model.objective == 'duration':
                total += travel_days

            elif self.model.objective == 'fuel':
                fuel_rate = float(vessel.corresponding_fuel[speed_idx])
                total += math.ceil(fuel_rate * travel_days)

            elif self.model.objective == 'cost':
                day_rate = float(vessel.day_rate_mob)
                fuel_price = float(vessel.default_fuel_price)
                idle_discount = float(self.model.idle_cost_discount)

                eco_speed = float(vessel.possible_speeds[0])
                eco_days = math.ceil(dist / (eco_speed * 24.0)) if eco_speed > 0 else travel_days
                mob_days = min(travel_days, eco_days)

                fuel_rate = float(vessel.corresponding_fuel[speed_idx])
                fuel_amt = math.ceil(fuel_rate * travel_days)

                arc_cost = day_rate * mob_days + fuel_price * fuel_amt

                # Idle cost from committed start times
                prev_end_time = committed_start_times.get(
                    prev_a, prev_act.possible_start[0]
                ) + prev_act.duration
                curr_start_time = committed_start_times.get(
                    curr_a, curr_act.possible_start[0]
                )
                gap = max(0, curr_start_time - prev_end_time)
                idle_days = max(0, gap - travel_days)
                arc_cost += idle_discount * day_rate * idle_days

                total += arc_cost

        return total

    def _decode_brkga_a(self, chromosome: Chromosome) -> DecodedSolution:
        """
        Random Assignment & Routes Decoder — HVASP-adapted BRKGA-A.

        Key departure from standard BRKGA-A
        ─────────────────────────────────────
        Standard BRKGA-A uses one role_priority_key PER ROLE for vessel
        selection. On the HVASP, all roles of the same activity share an
        identical vessel domain (all vessels are eligible for all tasks).
        This causes two compounding failures:

        1. Linkage disruption: BRKGA crossover inherits each role's key
            independently. Even if both parents have collision-free
            assignments, ~40% of offspring re-introduce C4 collisions
            because the keys are mixed across roles of the same activity.

        2. psi_r degeneracy: When all roles share the same domain and
            receive similar keys (e.g. uniform initialisation), psi_r
            values collapse to the same value, making the Step 3 ordering
            uninformative.

        Fix: use start_time_keys[act_idx] — ONE key per ACTIVITY — to
        govern vessel selection for all roles of that activity. Because
        crossover operates on the key of the activity as a unit, the
        assignment of all roles of the same activity always derives from
        the same parent, eliminating linkage disruption entirely.

        role_priority_keys[role_id] is then repurposed exclusively for
        psi_r — the within-vessel sequencing index — which is its correct
        theoretical role.

        Algorithm steps
        ───────────────
        Step 1: Vessel assignment — use start_time_keys[act_idx] + offset
                to map each role to a distinct vessel (C4-safe by
                construction). Roles sorted smallest-domain-first before
                assignment to minimise unresolvable conflicts.
        Step 2: Sequencing index — compute psi_r from role_priority_keys
                [role_id], which is now a pure sequence-ordering signal.
        Step 3: Per-vessel sort by psi_r ASC.
        Step 4: Hard feasibility insertion (C2, C5, C6) with All-or-Nothing
                rollback.
        """
        locations = self._decode_locations(chromosome)
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices = np.full(self.model.num_roles, -1, dtype=int)

        # ── Anchor ongoing activities ──────────────────────────────────────────────
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)

        committed_start_times: dict[int, int] = {}
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = self.model.activities[anchor_idx].possible_start[0]

        # ── Build activity → roles map (future activities only) ───────────────────
        future_set = set(future_activities)
        activity_to_roles: dict[int, list] = defaultdict(list)
        for role in self.model.roles:
            if role.parent_activity_idx in future_set:
                activity_to_roles[role.parent_activity_idx].append(role)

        # ── temp_vessel_roles[v_idx] = list of (act_idx, role_id, psi_r) ──────────
        temp_vessel_roles: list[list[tuple[int, int, float]]] = [
            [] for _ in range(self.model.num_vessels)
        ]

        # ══════════════════════════════════════════════════════════════════════════
        # Steps 1 & 2 — Vessel assignment + sequencing index
        # ══════════════════════════════════════════════════════════════════════════
        for act_idx, act_roles in activity_to_roles.items():

            # Sort roles by ascending domain size — most constrained first.
            # This minimises the chance that a small-domain role is left with
            # no available vessel after larger-domain roles have claimed theirs.
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            # Track vessels already claimed by earlier roles of this activity (C4)
            vessels_taken: set[int] = set()

            for role in act_roles_sorted:
                domain = role.vessel_domain
                N_r = len(domain)
                if N_r == 0:
                    continue  # No eligible vessels — role stays unassigned

                # ── Step 1: vessel selection via per-ACTIVITY key ──────────────────
                # Use start_time_keys[act_idx] as the activity-level assignment key.
                # All roles of the same activity derive their vessel from the SAME
                # key, so crossover cannot split an activity's assignment across
                # two parents (no linkage disruption).
                #
                # An integer offset (number of roles already claimed for this
                # activity) is added before the modulo so successive roles of the
                # same activity naturally land on successive domain slots, giving
                # a collision-free base assignment without any fallback needed
                # in the common case.
                rk_act = chromosome.start_time_keys[act_idx] # type: ignore
                offset = len(vessels_taken)
                tc_r = max(1, min(math.ceil(rk_act * N_r), N_r))
                domain_idx = (tc_r - 1 + offset) % N_r
                vessel_id = domain[domain_idx]
                tc_r = domain_idx + 1

                # ── C4: forward-wrap conflict resolution ───────────────────────────
                # The offset mechanism handles the common case, but if the domain
                # is small relative to vessels_taken, an explicit conflict check
                # is still needed.
                if vessel_id in vessels_taken:
                    resolved = False
                    for search_offset in range(1, N_r):
                        candidate_idx = (domain_idx + search_offset) % N_r
                        candidate = domain[candidate_idx]
                        if candidate not in vessels_taken:
                            domain_idx = candidate_idx
                            tc_r = domain_idx + 1
                            vessel_id = candidate
                            resolved = True
                            break
                    if not resolved:
                        continue

                # ── Step 2: sequencing index from role_priority_keys ───────────────
                # role_priority_keys[role_id] is now a pure sequence-ordering signal,
                # independent of vessel assignment. This gives psi_r meaningful
                # variation even when all roles share the same domain.
                rk_seq = chromosome.role_priority_keys[role.role_id] # type: ignore
                # _compute_tc_and_psi expects the full domain; we pass N_r as a
                # synthetic single-element domain to extract psi_r in [0,1).
                # Equivalent: psi_r = N_r * rk_seq - ceil(rk_seq * N_r) + 1
                _, psi_r, _ = self._compute_tc_and_psi(rk_seq, list(range(N_r))) # type: ignore

                vessels_taken.add(vessel_id)
                temp_vessel_roles[vessel_id].append((act_idx, role.role_id, psi_r))

        # ══════════════════════════════════════════════════════════════════════════
        # Step 3 — Sort each vessel's roles by psi_r ASC
        # ══════════════════════════════════════════════════════════════════════════
        for v_idx in range(self.model.num_vessels):
            temp_vessel_roles[v_idx].sort(key=lambda x: x[2])

        # ══════════════════════════════════════════════════════════════════════════
        # Step 4 — Hard feasibility insertion + All-or-Nothing rollback
        # ══════════════════════════════════════════════════════════════════════════
        # activity_tentative[act_idx] = list of role_ids committed so far for
        # that activity. Used by _rollback_activity to undo partial commits.
        activity_tentative: dict[int, list[int]] = defaultdict(list)

        for v_idx in range(self.model.num_vessels):
            for act_idx, role_id, _ in temp_vessel_roles[v_idx]:

                # Skip if this activity was already rolled back by a previous
                # vessel's processing of a different role in the same activity
                if act_idx in activity_tentative and activity_tentative[act_idx] == []:
                    # Sentinel: activity was rolled back — skip remaining roles
                    continue

                seq = vessel_sequences[v_idx]

                best_pos = -1
                best_temp_starts: dict[int, int] = {}
                min_travel = float("inf")

                for pos in range(1, len(seq) + 1):
                    is_feasible, temp_starts = self._is_insertion_feasible(
                        v_idx=v_idx,
                        target_act_idx=act_idx,
                        target_role_id=role_id,
                        pos=pos,
                        seq=seq,
                        locations=locations,
                        current_start_times=committed_start_times,
                        assigned_vessels=assigned_vessels,
                        chromosome=chromosome,
                    )
                    if not is_feasible:
                        continue

                    added_travel = self._calculate_added_travel(
                        v_idx=v_idx,
                        target_act_idx=act_idx,
                        target_role_id=role_id,
                        pos=pos,
                        seq=seq,
                        locations=locations,
                        assigned_vessels=assigned_vessels,
                        chromosome=chromosome,
                    )
                    if added_travel < min_travel:
                        min_travel = added_travel
                        best_pos = pos
                        best_temp_starts = temp_starts

                if best_pos == -1:
                    # ── All-or-Nothing: roll back every role committed so far
                    #    for this activity across ALL vessels ─────────────────
                    self._rollback_activity(
                        act_idx, vessel_sequences, committed_start_times,
                        assigned_vessels, activity_tentative
                    )
                    # Mark as rolled-back with empty sentinel so other vessels
                    # processing remaining roles of this activity skip them
                    activity_tentative[act_idx] = []
                else:
                    # Tentative commit
                    assigned_vessels[role_id] = v_idx
                    seq.insert(best_pos, act_idx)
                    committed_start_times.update(best_temp_starts)
                    activity_tentative[act_idx].append(role_id)

        # ── Finalise start times ───────────────────────────────────────────────────
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)
        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices
        )
    
    def _rollback_activity(
        self,
        act_idx: int,
        vessel_sequences: list[list[int]],
        committed_start_times: dict,
        assigned_vessels: np.ndarray,
        activity_tentative: dict
    ) -> None:
        """
        Rolls back ALL tentative insertions for act_idx.
        Called when Any-or-Nothing enforcement triggers for a multi-role activity.

        Removes act_idx from every vessel_sequence it was inserted into,
        clears its start time from committed_start_times, and resets
        assigned_vessels for all its roles back to -1.
        """
        # Remove act_idx from vessel sequences (it may appear in multiple vessels
        # if multi-role activities are split across vessels)
        for seq in vessel_sequences:
            if act_idx in seq:
                seq.remove(act_idx)

        # Clear committed start time
        committed_start_times.pop(act_idx, None)

        # Reset all role assignments for this activity
        for role_id in activity_tentative.get(act_idx, []):
            assigned_vessels[role_id] = -1

        # Clear tentative record
        activity_tentative.pop(act_idx, None)

    def _decode_stt(self, chromosome: Chromosome) -> DecodedSolution:
        """
        Shortest Travel Time Decoder — strict 4-step BRKGA-STT algorithm.

        Implements the theoretical algorithm exactly:
        Step 1: Sort activities by random key (desc), then topological sort.
        Step 2: For each activity, evaluate all (vessel, position) pairs using
                HARD feasibility (C2 time windows + C6 travel times).
        Step 3: All-or-Nothing commit: if ANY role in R_a has no feasible
                insertion, drop the ENTIRE activity (all roles unassigned).
        Step 4: Continue until all activities processed.

        Unassigned activities accumulate a financial penalty in DecodedSolution.
        """
        locations = self._decode_locations(chromosome)
        assigned_vessels = np.full(self.model.num_roles, -1, dtype=int)
        speed_indices = np.full(self.model.num_roles, -1, dtype=int)
        committed_start_times = {}

        # --- Step 1a: Anchor ongoing activities ---
        vessel_sequences, future_activities = self._anchor_ongoing_activities(assigned_vessels)
        for v_idx, seq in enumerate(vessel_sequences):
            if seq:
                anchor_idx = seq[0]
                committed_start_times[anchor_idx] = self.model.activities[anchor_idx].possible_start[0]

        # --- Step 1b: Sort by random key (desc), then topological sort ---
        activity_priorities = [
            (act_idx, chromosome.start_time_keys[act_idx]) for act_idx in future_activities # type: ignore
        ]
        sorted_acts = sorted(activity_priorities, key=lambda x: x[1], reverse=True)
        anchored_indices = [seq[0] for seq in vessel_sequences if seq]
        ordered_act_indices = self._topological_sort(sorted_acts, anchored_indices)

        # --- Steps 2, 3, 4: Greedy insertion with hard feasibility ---
        for act_idx in ordered_act_indices:
            act_roles = [r for r in self.model.roles if r.parent_activity_idx == act_idx]
            
            # Collect the best feasible (vessel, position) per role.
            # Key invariant: if ANY role has no feasible candidate, we skip ALL roles.
            best_per_role = []
            activity_feasible = True
            planned_vessels_for_this_act = set()

            # Sort roles by ascending domain size — most constrained first (C4 safeguard)
            act_roles_sorted = sorted(act_roles, key=lambda r: len(r.vessel_domain))

            for role in act_roles_sorted:
                best_v_idx = -1
                best_insert_pos = -1
                best_temp_start_times = {}
                min_travel = float('inf')

                for v_idx in role.vessel_domain:
                    if v_idx in planned_vessels_for_this_act:
                        continue

                    seq = vessel_sequences[v_idx]
                    for pos in range(1, len(seq) + 1):
                        is_feasible, temp_starts = self._is_insertion_feasible(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            current_start_times=committed_start_times,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )
                        if not is_feasible:
                            continue

                        added_travel = self._calculate_added_travel(
                            v_idx=v_idx,
                            target_act_idx=act_idx,
                            target_role_id=role.role_id,
                            pos=pos,
                            seq=seq,
                            locations=locations,
                            assigned_vessels=assigned_vessels,
                            chromosome=chromosome,
                        )

                        if added_travel < min_travel:
                            min_travel = added_travel
                            best_v_idx = v_idx
                            best_insert_pos = pos
                            best_temp_start_times = temp_starts

                if best_v_idx == -1:
                    # No feasible vessel found for this role — entire activity fails (Step 3)
                    activity_feasible = False
                    break  # No need to evaluate remaining roles

                best_per_role.append({
                    'role_id': role.role_id,
                    'v_idx': best_v_idx,
                    'pos': best_insert_pos,
                    'starts': best_temp_start_times,
                })
                # Reserve this vessel so subsequent roles in R_a cannot claim it (C4)
                planned_vessels_for_this_act.add(best_v_idx)

            # --- Step 3: All-or-Nothing atomic commit ---
            if activity_feasible:
                for assignment in best_per_role:
                    rid = assignment['role_id']
                    v_idx = assignment['v_idx']
                    assigned_vessels[rid] = v_idx
                    speed_indices[rid] = self._decode_speed_index_for_role(chromosome, rid, v_idx)
                    vessel_sequences[v_idx].insert(assignment['pos'], act_idx)
                    committed_start_times.update(assignment['starts'])
            # If not feasible: assigned_vessels stays at -1 for all roles in R_a.
            # vessel_sequences and committed_start_times are NOT touched (no partial state).

        # --- Finalize start times ---
        final_start_times = np.zeros(self.model.num_activities, dtype=int)
        for idx in range(self.model.num_activities):
            final_start_times[idx] = committed_start_times.get(
                idx, self.model.activities[idx].possible_start[0]
            )

        penalty = self._calculate_unassigned_penalty(assigned_vessels)
        return DecodedSolution(
            assigned_vessels=assigned_vessels,
            start_times=final_start_times,
            locations=locations,
            vessel_sequences=vessel_sequences,
            is_next_matrix=np.zeros((1, 1)),
            unassigned_penalty=penalty,
            speed_indices=speed_indices
        )

    def _calculate_added_travel(
        self,
        v_idx: int,
        target_act_idx: int,
        target_role_id: int,
        pos: int,
        seq: List[int],
        locations: np.ndarray,
        assigned_vessels: np.ndarray,
        chromosome: Chromosome
    ) -> float:
        """
        Helper function to calculate the extra travel time incurred by inserting `target_act_idx` at `pos`.
        Uses travel in days using speed-dependent arc times:
            prev->target uses target role speed
            target->next uses next activity role speed on this vessel
            minus broken prev->next (uses next activity role speed)
        """
        added = 0.0

        if pos > 0:
            prev_act_idx = seq[pos - 1]
            added += self._travel_time_days_for_role_arc(
                v_idx, prev_act_idx, target_act_idx, target_role_id, locations, chromosome
            )

        if pos < len(seq):
            next_act_idx = seq[pos]
            next_role_id = self._select_role_id_for_activity_vessel(next_act_idx, v_idx, assigned_vessels)
            added += self._travel_time_days_for_role_arc(
                v_idx, target_act_idx, next_act_idx, next_role_id, locations, chromosome
            )

            if pos > 0:
                prev_act_idx = seq[pos - 1]
                added -= self._travel_time_days_for_role_arc(
                    v_idx, prev_act_idx, next_act_idx, next_role_id, locations, chromosome
                )

        return max(0.0, float(added))

    def _calculate_added_cost(
        self,
        v_idx: int,
        target_act_idx: int,
        target_role_id: int,
        pos: int,
        seq: List[int],
        locations: np.ndarray,
        assigned_vessels: np.ndarray,
        chromosome: Chromosome,
        committed_start_times: dict,
    ) -> float:
        """
        Calculate the added COST of inserting target_act_idx at pos in vessel v_idx's sequence.
        
        Cost components:
        1. Travel cost:  day_rate * mob_days (capped at eco-speed travel time)
        2. Fuel cost:    fuel_price * fuel_consumed (speed-dependent)
        3. Idle cost:    idle_cost_discount * day_rate * idle_days
        
        This replaces _calculate_added_travel as the greedy metric when objective == 'cost'.
        """
        vessel = self.model.vessels[v_idx]
        day_rate = float(vessel.day_rate_mob)
        fuel_price = float(vessel.default_fuel_price)
        idle_discount = float(self.model.idle_cost_discount)

        def _arc_cost(prev_act_idx: int, curr_act_idx: int, curr_role_id: int) -> float:
            """Cost of a single arc: prev -> curr on vessel v_idx."""
            prev_act = self.model.activities[prev_act_idx]
            curr_act = self.model.activities[curr_act_idx]

            prev_end_loc = self.model._get_global_loc_id(prev_act, locations[prev_act_idx], False)
            curr_start_loc = self.model._get_global_loc_id(curr_act, locations[curr_act_idx], True)

            dist = float(self.model.vessel_travel_dist_matrix_float[v_idx][prev_end_loc, curr_start_loc])

            # Speed for this arc
            speed_idx = self._decode_speed_index_for_role(chromosome, curr_role_id, v_idx)
            speed_kn = float(vessel.possible_speeds[speed_idx])
            
            if dist <= 0 or speed_kn <= 0:
                return 0.0

            travel_days = math.ceil(dist / (speed_kn * 24.0))

            # Eco-speed travel time (for mob_days cap)
            eco_speed = float(vessel.possible_speeds[0]) if len(vessel.possible_speeds) > 0 else speed_kn
            eco_days = math.ceil(dist / (eco_speed * 24.0)) if eco_speed > 0 else travel_days
            mob_days = min(travel_days, eco_days)

            # Fuel
            fuel_rate = float(vessel.corresponding_fuel[speed_idx])
            fuel_amt = math.ceil(fuel_rate * travel_days)

            travel_cost = day_rate * mob_days
            fuel_cost = fuel_price * fuel_amt

            return travel_cost + fuel_cost

        def _idle_cost(prev_act_idx: int, curr_act_idx: int, curr_role_id: int) -> float:
            """Idle cost: vessel waiting between finishing transit and activity start."""
            prev_act = self.model.activities[prev_act_idx]
            curr_act = self.model.activities[curr_act_idx]

            prev_end_loc = self.model._get_global_loc_id(prev_act, locations[prev_act_idx], False)
            curr_start_loc = self.model._get_global_loc_id(curr_act, locations[curr_act_idx], True)
            dist = float(self.model.vessel_travel_dist_matrix_float[v_idx][prev_end_loc, curr_start_loc])

            speed_idx = self._decode_speed_index_for_role(chromosome, curr_role_id, v_idx)
            speed_kn = float(vessel.possible_speeds[speed_idx])

            if dist <= 0 or speed_kn <= 0:
                travel_days = 0
            else:
                travel_days = math.ceil(dist / (speed_kn * 24.0))

            # Compute the gap (actual elapsed time between activities)
            prev_end_time = committed_start_times.get(
                prev_act_idx, prev_act.possible_start[0]
            ) + prev_act.duration
            curr_start_time = committed_start_times.get(
                curr_act_idx, curr_act.possible_start[0]
            )
            
            gap = max(0, curr_start_time - prev_end_time)
            idle_days = max(0, gap - travel_days)

            return idle_discount * day_rate * idle_days

        # --- Compute cost delta of insertion ---
        added_cost = 0.0

        # Arc: prev -> target
        if pos > 0:
            prev_act_idx = seq[pos - 1]
            added_cost += _arc_cost(prev_act_idx, target_act_idx, target_role_id)

        # Arc: target -> next
        if pos < len(seq):
            next_act_idx = seq[pos]
            next_role_id = self._select_role_id_for_activity_vessel(next_act_idx, v_idx, assigned_vessels)
            added_cost += _arc_cost(target_act_idx, next_act_idx, next_role_id)

            # Subtract the broken arc: prev -> next (which no longer exists)
            if pos > 0:
                prev_act_idx = seq[pos - 1]
                added_cost -= _arc_cost(prev_act_idx, next_act_idx, next_role_id)

        return max(0.0, added_cost)

    def _calculate_start_times(self, vessel_sequences: List[List[int]], locations: np.ndarray, assigned_vessels: np.ndarray) -> np.ndarray:
        """
        Simulates the schedule temporally. Calculates the earliest feasible start time for every
        activity respecting precedences (C5) and minimum travel times.
        Any constraint violations will inherently push times out of `possible_start` bounds, 
        which the GA's `compute_fitness` will penalize.
        """
        start_times = np.zeros(self.model.num_activities, dtype=int)
        
        # We process activities in topological order (sequence dependencies + predecessor dependencies)
        # To handle complex graphs simply, we iteratively push start times until convergence.
        changed = True
        iterations = 0
        max_iterations = self.model.num_activities * 2 # Prevent infinite loop in case of cyclic precedence
        
        # Initialize all to their earliest possible start window
        for act_idx, act in enumerate(self.model.activities):
            start_times[act_idx] = act.possible_start[0]

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for act_idx, act in enumerate(self.model.activities):
                current_start = start_times[act_idx]
                min_required_start = act.possible_start[0]
                
                # 1. Check Predecessor Constraints (C5)
                if hasattr(act, 'predecessor_name') and act.predecessor_name:
                    pred_idx = self.model.activity_name_to_idx.get(act.predecessor_name, -1)
                    if pred_idx != -1:
                        pred_end = start_times[pred_idx] + self.model.activities[pred_idx].duration
                        min_required_start = max(min_required_start, pred_end)

                # 2. Check Sequence Constraints (Travel from previous activity in vessel route)
                act_roles = [r for r in self.model.roles if r.parent_activity_idx == act_idx]
                
                for r in act_roles:
                    v_idx = assigned_vessels[r.role_id]
                    if v_idx == -1: continue
                    
                    seq = vessel_sequences[v_idx]
                    if act_idx in seq:
                        pos = seq.index(act_idx)
                        if pos > 0:
                            prev_act_idx = seq[pos - 1]
                            prev_act = self.model.activities[prev_act_idx]
                            
                            prev_end = start_times[prev_act_idx] + prev_act.duration
                            p_end_loc = self.model._get_global_loc_id(prev_act, locations[prev_act_idx], False)
                            t_start_loc = self.model._get_global_loc_id(act, locations[act_idx], True)
                            
                            travel_time = self.model.vessel_travel_time_fastest_matrix[v_idx][p_end_loc, t_start_loc]
                            vessel_arrival = prev_end + travel_time
                            
                            min_required_start = max(min_required_start, vessel_arrival)

                # If the required start time is pushed later, update and trigger another loop
                if min_required_start > current_start:
                    start_times[act_idx] = min_required_start
                    changed = True

        return start_times
    
    def _decode_speed_index_for_role(self, chromosome: Chromosome, role_id: int, vessel_idx: int) -> int:
        """
        Decode vessel-specific speed index from chromosome.speed_keys.
        Backward-compatible fallback:
        - if speed_keys missing/None -> fastest speed index
        """
        vessel = self.model.vessels[vessel_idx]
        n = len(vessel.possible_speeds)
        if n <= 0:
            return 0

        if chromosome.speed_keys is None or role_id >= len(chromosome.speed_keys):
            return n - 1  # fallback: fastest

        key = float(chromosome.speed_keys[role_id])
        key = min(max(key, 0.0), 0.999999999)
        idx = min(int(key * n), n - 1)
        return idx

    def _travel_time_days_for_role_arc(
        self,
        v_idx: int,
        prev_act_idx: int,
        curr_act_idx: int,
        curr_role_id: int,
        locations: np.ndarray,
        chromosome: Chromosome
    ) -> int:
        """
        Travel time for arc prev->curr using speed key of current role.
        """
        prev_act = self.model.activities[prev_act_idx]
        curr_act = self.model.activities[curr_act_idx]

        prev_end_loc = self.model._get_global_loc_id(prev_act, locations[prev_act_idx], False)
        curr_start_loc = self.model._get_global_loc_id(curr_act, locations[curr_act_idx], True)

        dist = float(self.model.vessel_travel_dist_matrix_float[v_idx][prev_end_loc, curr_start_loc])

        speed_idx = self._decode_speed_index_for_role(chromosome, curr_role_id, v_idx)
        speed_kn = float(self.model.vessels[v_idx].possible_speeds[speed_idx])

        if dist <= 0 or speed_kn <= 0:
            return 0
        return int(math.ceil(dist / (speed_kn * 24.0)))

    def _select_role_id_for_activity_vessel(self, act_idx: int, v_idx: int, assigned_vessels: np.ndarray) -> int:
        """
        Returns role_id in activity assigned to vessel. If none yet assigned, returns first role_id of activity.
        """
        act_roles = [r for r in self.model.roles if r.parent_activity_idx == act_idx]
        for r in act_roles:
            if assigned_vessels[r.role_id] == v_idx:
                return r.role_id
        return act_roles[0].role_id if act_roles else -1
    
    def _anchor_ongoing_activities(self, assigned_vessels: np.ndarray) -> Tuple[List[List[int]], List[int]]:
        """
        Identifies current_location and ongoing activities, locks their vessel assignments,
        and anchors them at the start of the respective vessel sequences.
        Returns the initialized sequences and a list of the remaining (future) activities to schedule.
        """
        vessel_sequences = [[] for _ in range(self.model.num_vessels)]
        future_activities = []
        
        for act_idx, act in enumerate(self.model.activities):
            # In activity.py, is_current_activity captures both current_location and ongoing tows/maintenance
            if getattr(act, 'is_current_activity', False):
                # Find the role(s) for this activity
                act_roles = [r for r in self.model.roles if r.parent_activity_idx == act_idx]
                assigned_to_this_act = set()

                for role in act_roles:
                    assert len(role.vessel_domain) == 1, (
                        f'The activity must specify which vessel will perform which role.'
                        f'Activity: {act.activity_name}'
                    )
                    # Assign the first available vessel in the domain not yet working on this act
                    for v_idx in role.vessel_domain:
                        if v_idx not in assigned_to_this_act:
                            assigned_vessels[role.role_id] = v_idx
                            vessel_sequences[v_idx].append(act_idx)
                            assigned_to_this_act.add(v_idx)
                            break
            else:
                future_activities.append(act_idx)
                
        return vessel_sequences, future_activities
    
    def _calculate_unassigned_penalty(self, assigned_vessels: np.ndarray) -> float:
        """
        Calculates a financial penalty for any roles that were left unassigned.
        Penalty = Activity Duration * Max Day Rate of Allowed Vessels
        """
        penalty = 0.0
        for role in self.model.roles:
            if assigned_vessels[role.role_id] == -1:
                # The role was not scheduled
                duration = role.parent_activity.duration
                
                if role.vessel_domain:
                    # Find the maximum day rate among vessels that COULD have done this role
                    max_rate = max(self.model.vessels[v_idx].day_rate_mob for v_idx in role.vessel_domain) * 10
                else:
                    # Fallback if no vessels were in domain
                    max_rate = 1000000.0 
                    
                # Add to total penalty
                penalty += duration * max_rate
                
        return penalty

    def _is_insertion_feasible(
        self,
        v_idx: int,
        target_act_idx: int,
        target_role_id: int,
        pos: int,
        seq: List[int],
        locations: np.ndarray,
        current_start_times: dict,
        assigned_vessels: np.ndarray,
        chromosome: Chromosome
    ) -> Tuple[bool, dict]:
        """
        Hard feasibility with speed-dependent travel times.
        """
        temp_seq = seq.copy()
        temp_seq.insert(pos, target_act_idx)

        temp_start_times = {}

        for i, act_idx in enumerate(temp_seq):
            act = self.model.activities[act_idx]
            min_start = act.possible_start[0]

            # C5: Predecessor constraint
            if hasattr(act, 'predecessor_name') and act.predecessor_name:
                pred_idx = self.model.activity_name_to_idx.get(act.predecessor_name, -1)
                if pred_idx != -1:
                    # Look in globally committed times first, then in times computed
                    # earlier in this same vessel's sequence (temp_start_times).
                    if pred_idx in current_start_times:
                        pred_end = current_start_times[pred_idx] + self.model.activities[pred_idx].duration
                        min_start = max(min_start, pred_end)
                    elif pred_idx in temp_start_times:
                        # Predecessor was just processed earlier in this same vessel sequence
                        pred_end = temp_start_times[pred_idx] + self.model.activities[pred_idx].duration
                        min_start = max(min_start, pred_end)
                    else:
                        # Predecessor is unscheduled — successor cannot be feasibly placed
                        return False, {}

            # C6: Travel time from previous activity in sequence
            if i > 0:
                prev_act_idx = temp_seq[i - 1]
                prev_end = temp_start_times[prev_act_idx] + self.model.activities[prev_act_idx].duration

                # incoming arc to act_idx uses speed key of role assigned to act_idx on this vessel
                if act_idx == target_act_idx:
                    role_for_arc = target_role_id
                else:
                    role_for_arc = self._select_role_id_for_activity_vessel(act_idx, v_idx, assigned_vessels)

                travel_time = self._travel_time_days_for_role_arc(
                    v_idx=v_idx,
                    prev_act_idx=prev_act_idx,
                    curr_act_idx=act_idx,
                    curr_role_id=role_for_arc,
                    locations=locations,
                    chromosome=chromosome
                )
                min_start = max(min_start, prev_end + travel_time)

            valid_starts = [t for t in act.possible_start if t >= min_start]
            if not valid_starts:
                return False, {}

            # Use the chromosome key to pick which valid start time
            window_key = float(chromosome.start_window_keys[act_idx]) # type: ignore
            window_key = min(max(window_key, 0.0), 0.999999999)
            chosen_idx = min(int(window_key * len(valid_starts)), len(valid_starts) - 1)
            temp_start_times[act_idx] = valid_starts[chosen_idx]

        return True, temp_start_times
    
    def _topological_sort(
        self,
        sorted_acts: List[Tuple[int, float]],
        anchored_act_indices: List[int]
    ) -> List[int]:
        """
        Topological sort.
        
        1. Activities with NO predecessor (predecessor_name == "" or None) are 
        immediately schedulable. They must not be trapped in the deadlock path.
        2. Uses explicit set membership check rather than string-in-set.
        3. Preserves random-key priority order among schedulable activities.
        """
        ordered_act_indices = []
        
        # Track by name — but explicitly handle the "no predecessor" case
        scheduled_names = {
            self.model.activities[idx].activity_name for idx in anchored_act_indices
        }
        # Add sentinel for "no predecessor required"
        scheduled_names.add("")
        scheduled_names.add(None)
        
        remaining_acts = list(sorted_acts)
        
        while remaining_acts:
            progress_made = False
            
            for i, (act_idx, prio) in enumerate(remaining_acts):
                act = self.model.activities[act_idx]
                pred_name = getattr(act, 'predecessor_name', None)
                
                # Schedulable if: no predecessor OR predecessor already scheduled
                is_schedulable = (
                    pred_name is None or 
                    pred_name == "" or 
                    pred_name in scheduled_names
                )
                
                if is_schedulable:
                    ordered_act_indices.append(act_idx)
                    scheduled_names.add(act.activity_name)
                    remaining_acts.pop(i)
                    progress_made = True
                    break  # Restart from highest remaining priority
            
            if not progress_made:
                # True deadlock: circular dependency or missing predecessor in dataset.
                # Force-schedule the highest-priority remaining activity.
                act_idx, _ = remaining_acts.pop(0)
                ordered_act_indices.append(act_idx)
                scheduled_names.add(self.model.activities[act_idx].activity_name)
        
        return ordered_act_indices

    def _compute_tc_and_psi(self, rk: float, domain: list) -> tuple[int, float, int]:
        """
        Computes TC_r (1-indexed vessel selection), the corresponding global vessel ID,
        and psi_r (sequencing index) from a single random key.

        Returns:
            (global_vessel_id, psi_r, tc_r)
            - global_vessel_id: the selected vessel's global ID
            - psi_r:            sequencing index in [0, 1)
            - tc_r:             1-indexed position in domain (needed for C4 resolution)
        """
        N_r = len(domain)

        tc_r = max(1, min(math.ceil(rk * N_r), N_r))  # 1-indexed, edge-case clamped
        domain_idx = tc_r - 1
        global_vessel_id = domain[domain_idx]
        psi_r = N_r * rk - tc_r + 1  # in [0, 1) by construction

        return global_vessel_id, psi_r, tc_r