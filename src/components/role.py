from components.activity import Activity


class Role:
    """
    Represents a single vessel slot required for an activity.
    """

    def __init__(self, role_id: int, parent_activity: Activity, parent_activity_idx: int, vessel_domain: list[int], role_group_id: str) -> None:
        self.role_id = role_id  # unique id for this role
        self.parent_activity = parent_activity  # the activity object it belongs to
        self.parent_activity_idx = parent_activity_idx  # the id of its parent
        self.vessel_domain = vessel_domain  # list of vessel ids that can fill this role
        self.role_group_id = role_group_id  # id to link roles
        self.is_current_role = parent_activity.is_current_activity # set role as current role if parent is active role

    def __repr__(self) -> str:
        return f"Role_{self.role_id} (Activity: {self.parent_activity.activity_name})"
