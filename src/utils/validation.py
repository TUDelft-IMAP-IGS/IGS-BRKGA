def verify_path_logic(model_instance, verbose=False):
    """
    Verifies if the solver's 'is_next' variables match the actual chronological
    schedule of the vessels.
    """
    output_text = f"\n--- VERIFYING PATH LOGIC ({model_instance.objective}: {model_instance.objective_var.value()}) ---"

    issues_found = 0
    total_links_expected = 0
    total_links_found = 0

    # 1. Reconstruct the schedule from the raw variable values
    assignments = [v.value() for v in model_instance.assigned_vessel_vars]
    start_times = [v.value() for v in model_instance.start_time_vars]

    # 2. Iterate per Vessel
    for v_idx, vessel in enumerate(model_instance.vessels):
        # Get all roles assigned to this vessel
        vessel_roles = []
        for r_idx, role in enumerate(model_instance.roles):
            if assignments[r_idx] == v_idx:
                # Store (Start Time, Role Index, Activity Name)
                # We look up the start time of the PARENT activity
                s_time = start_times[role.parent_activity_idx]
                vessel_roles.append((s_time, r_idx, role.parent_activity.activity_name))

        # Sort roles by start time to see the physical sequence
        vessel_roles.sort(key=lambda x: x[0])

        if not vessel_roles:
            continue

        output_text += f"\n\n🚢 Vessel: {vessel.vessel_name} ({len(vessel_roles)} tasks)"

        # 3. Check the links between consecutive tasks
        for i in range(len(vessel_roles) - 1):
            total_links_expected += 1

            current_task = vessel_roles[i]
            next_task = vessel_roles[i + 1]

            r_id_curr = current_task[1]
            r_id_next = next_task[1]

            # Check what the solver decided for this link
            # Note: is_next is a boolean variable, we need .value()
            link_active = model_instance.is_next[r_id_curr, r_id_next].value()

            start_arrow = " -> " if link_active else " -X-> "
            status = "✅ OK" if link_active else "❌ BROKEN LINK (Teleportation Detected)"

            if not link_active:
                issues_found += 1
            else:
                total_links_found += 1

            output_text += f"\n   {current_task[2]} (t={current_task[0]}) {start_arrow} {next_task[2]} (t={next_task[0]}) | {status}"

    output_text += "\n\n--- SUMMARY ---"
    output_text += f"\nLinks Expected (Chronological): {total_links_expected}"
    output_text += f"\nLinks Found (Solver):           {total_links_found}"

    if issues_found > 0:
        output_text += f"\n⚠️  CRITICAL: Found {issues_found} missing links."
        output_text += "\n    The solver is 'cheating' to minimize cost. The Proxy Distance is UNDERESTIMATED."
    else:
        output_text += "\n✅  SUCCESS: All physical paths are correctly accounted for in the cost."

    if verbose:
        print(output_text)

    return issues_found == 0, output_text
