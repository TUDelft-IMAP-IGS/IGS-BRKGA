import numpy as np
from datetime import date, datetime

from loguru import logger


class Activity:
    """Represents an activity (tow, maintenance, or current vessel location) in the scheduling model."""

    def __init__(self, activity_dict: dict, date_origin: date):
        """
        Initialize an Activity instance.

        Parameters:
        activity_dict (dict): Dictionary containing the activity's parameters.
        date_origin (date): The reference date used to convert string dates into relative integer days.

        Returns:
        -
        """

        def get_corresponding_dates(d1_str: str | datetime, d2_str: str | datetime | None) -> np.ndarray:
            """
            Convert one/two date strings into an array of possible start days (integers).

            Parameters:
            d1_str (str): The first date string in format "%d/%m/%Y".
            d2_str (str | None): Optional second date string in format "%d/%m/%Y".

            Returns:
            np.ndarray: Array of possible start days relative to `date_origin`.
            """

            # extract date from string/datetime and convert into integer
            if isinstance(d1_str, str):
                d1 = datetime.strptime(d1_str, "%d/%m/%Y").date()
            else:
                d1 = d1_str.date()

            start_window = (d1 - date_origin).days

            if d2_str is not None:
                # extract date from string/datetime and convert into integer
                if isinstance(d2_str, str):
                    d2 = datetime.strptime(d2_str, "%d/%m/%Y").date()
                else:
                    d2 = d2_str.date()

                end_window = (d2 - date_origin).days + self.duration

                if end_window >= 0:
                    possible_starts = np.arange(max(0, start_window), end_window - self.duration + 1, 1)
                else:
                    raise Exception(f"Activity start window in the past. No actual start date provided. {activity_dict.get('activity_name', '')}")
            else:
                possible_starts = np.array([start_window])

            return possible_starts

        # get activity type
        self.activity_type = activity_dict.get("activity_type", "tow")
        self.is_current_location = activity_dict.get("is_current_location", False)

        # check that the dictionary has all the required input parameters
        self._validate_params(activity_dict)

        # get relevant input parameters
        if self.is_current_location:
            # VESSEL START LOCATION

            self.activity_type = "current_location"
            # corresponding vessel
            self.target_vessel = activity_dict.get("target_vessel")
            self.allowed_vessels = [self.target_vessel]
            self.required_vessel_config = [[{"vessels": [self.target_vessel], "count": 1}]]

            # set activity name
            activity_dict["activity_name"] = f"Current Location - {self.target_vessel}"

            # set the duration and dates
            self.duration = 0
            self.duration_fixed = 0
            self.duration_option = 0

            self.possible_start = [0]

            # tow start and end location
            self.start_location = activity_dict.get("location", [0, 0])
            self.end_location = self.start_location

            self.route_description = ""

            # indicator that current activity is already underway
            self.is_current_activity = True

            # get activity name
            self.activity_name = activity_dict.get("activity_name", "")

        elif self.is_tow_type():
            # TOW ACTIVITY

            # activity window and duration
            self.duration_fixed = activity_dict.get("duration_fixed", 0)
            self.duration_option = activity_dict.get("duration_options", 0)

            self.project_code = activity_dict.get("project_code", "")

            if not self.duration_fixed:
                self.duration_fixed = 0

            if not self.duration_option:
                self.duration_option = 0

            self.duration = activity_dict.get("duration_opex", self.duration_fixed + self.duration_option)

            if not self.duration:
                self.duration = self.duration_fixed + self.duration_option

            # tow start and end location
            self.start_location = activity_dict.get("start_location", [0, 0])
            self.end_location = activity_dict.get("end_location", [0, 0])

            # start and end location names
            self.start_location_name = activity_dict.get("start_location_name")
            self.end_location_name = activity_dict.get("end_location_name")

            self.route_description = activity_dict.get("route_description", "")

            if self.route_description == "":
                self.route_description = f"{self.start_location_name} - {self.end_location_name}"

            # required vessel configuration
            self.required_vessel_config = activity_dict.get("required_vessel_config", [[{"vessels": "*", "count": 0}]])

            self.allowed_vessels = list(dict.fromkeys(v for group in self.required_vessel_config for subgroup in group for v in subgroup["vessels"]))

            self.use_full_window = activity_dict.get("use_full_window", False)

            # get start window
            if (activity_dict.get("actual_start") is not None) and (activity_dict.get("actual_start") != ""):
                self.possible_start = get_corresponding_dates(activity_dict.get("actual_start", ""), None)
            else:
                if self.use_full_window:
                    # ALLOW START DATE TO BE ANY TIME IN WINDOW
                    self.possible_start = get_corresponding_dates(
                        activity_dict.get("begin_start_window", ""),
                        activity_dict.get("end_start_window"),
                    )
                else:
                    # USE THE START OF THE WINDOW
                    self.possible_start = get_corresponding_dates(
                        activity_dict.get("begin_start_window", ""),
                        None,
                    )

            # determine if this activity follows another
            self.predecessor_name = activity_dict.get("predecessor_name", "")

            # logger.debug(activity_dict.get("activity_name", ""))
            # determine whether this activity is currently underway
            self.is_current_activity = self.possible_start[-1] <= 0

            # get activity name
            self.activity_name = activity_dict.get("activity_name", "")

        elif self.is_maintenance_type():
            # MAINTENANCE ACTIVITY

            # corresponding vessel
            self.target_vessel = activity_dict.get("target_vessel")
            self.allowed_vessels = [self.target_vessel]

            # maintenance locations
            self.allowed_locations = activity_dict.get("allowed_locations", [])
            self.allowed_locations_names = activity_dict.get("allowed_locations_names", [""] * len(self.allowed_locations))

            # maintenance duration
            self.duration = activity_dict.get("duration", 0)

            # get start window
            if (activity_dict.get("actual_start") is not None) and (activity_dict.get("actual_start") != ""):
                self.possible_start = get_corresponding_dates(activity_dict.get("actual_start", ""), None)
            else:
                self.possible_start = get_corresponding_dates(
                    activity_dict.get("begin_start_window", ""),
                    activity_dict.get("end_start_window"),
                )

            # determine whether this activity is currently underway
            self.is_current_activity = (self.possible_start[-1] <= 0) and (self.possible_start[-1] + self.duration >= 0)

            if self.possible_start[-1] <= 0:
                self.possible_start = [self.possible_start[-1]]

            # get activity name
            self.activity_name = activity_dict.get("activity_name", "") + f"_{self.target_vessel}"

        else:
            raise ValueError(f"Unknown activity type: {self.activity_type}")

    def is_tow_type(self) -> bool:
        """Return True if activity type is 'tow'."""
        return self.activity_type == "tow"

    def is_maintenance_type(self) -> bool:
        """Return True if activity type is 'maintenance'."""
        return self.activity_type == "maintenance"
    
    def is_current_location_type(self) -> bool:
        """"Return True if activity type is 'current location'."""
        return self.activity_type == "current_location"

    def _validate_params(self, activity_dict) -> None:
        """
        Validate that all required parameters are present for the given activity type.

        Parameters:
        activity_dict (dict[str, Any]): The dictionary containing activity parameters.

        Raises:
        ValueError: If required parameters are missing.
        """
        if self.is_current_location:
            required = ["target_vessel", "location"]
        elif self.is_tow_type():
            required = ["start_location", "end_location", "required_vessel_config"]
        elif self.is_maintenance_type():
            required = ["target_vessel", "allowed_locations"]
        else:
            required = []

        missing = [param for param in required if activity_dict.get(param) in (None, "")]
        if missing:
            raise ValueError(f"Missing required parameters for {self.activity_type}: {', '.join(missing)}")

    def check_valid(self) -> bool:
        """
        Check whether the activity object that has been created is valid.

        Returns:
        bool: Is a valid activity.
        """
        # check whether activity has already past
        if self.possible_start[-1] + self.duration < 0:
            print(f"Activity ({self.activity_name}) is in the past.")
            return False

        return True

    def to_dict(self) -> dict:
        if self.is_current_location:
            activity_dict = {
                "activity_name": self.activity_name,
                "is_tow": self.is_tow_type(),
                "is_current_location": self.is_current_location,
                "duration_fixed": self.duration_fixed,
                "duration_option": self.duration_option,
                "duration": self.duration,
                "route_description": self.route_description,
                "start_location": self.start_location,
                "end_location": self.end_location,
                "start_location_name": f"Current Location: {self.target_vessel}",
                "end_location_name": f"Current Location: {self.target_vessel}",
            }
        elif self.is_tow_type():
            activity_dict = {
                "activity_name": self.activity_name,
                "project_code": self.project_code,
                "is_tow": self.is_tow_type(),
                "is_current_location": self.is_current_location,
                "duration_fixed": self.duration_fixed,
                "duration_option": self.duration_option,
                "duration": self.duration,
                "route_description": self.route_description,
                "start_location": self.start_location,
                "end_location": self.end_location,
                "start_location_name": self.start_location_name,
                "end_location_name": self.end_location_name,
            }
        else:
            activity_dict = {
                "activity_name": self.activity_name,
                "is_tow": self.is_tow_type(),
                "is_current_location": False,
                "duration": self.duration,
                "allowed_vessels": self.allowed_vessels,
                "allowed_locations": self.allowed_locations,
                "allowed_locations_names": self.allowed_locations_names,
            }

        return activity_dict

    def __repr__(self) -> str:
        """
        Return a minimal string representation of the Activity.

        Returns:
        str: The activity name.
        """
        return self.activity_name

    def __str__(self) -> str:
        """
        Return a comprehensive string representation of the Activity.

        Returns:
        str: Full details of activity.
        """
        if self.is_tow_type() or self.is_current_location:
            output = (
                f"Activity Name: {self.activity_name}\n"
                + f"Activity Type: {self.activity_type} {'*' if self.is_current_location else ''}\n"
                + f"Allowed Vessels: {self.allowed_vessels}\n"
                + f"Possible Starts: {self.possible_start[0]} -> {self.possible_start[-1]}\n"
                + f"Duration: {self.duration} ({self.duration_fixed} + {self.duration_option})\n"
                + f"Route: ({self.start_location[0]}, {self.start_location[1]})"
                + f" -> ({self.end_location[0]}, {self.end_location[1]})\n"
                + f"Route Description: {self.route_description}\n"
                + f"Is Active: {self.is_current_activity}\n"
            )
        else:
            output = (
                f"Activity Name: {self.activity_name}\n"
                + f"Activity Type: {self.activity_type}\n"
                + f"Assigned Vessels: {self.target_vessel}\n"
                + f"Possible Starts: {self.possible_start[0]} -> {self.possible_start[-1]}\n"
                + f"Allowed Locations: {self.allowed_locations}\n"
                + f"Allowed Locations: {self.allowed_locations_names}\n"
                + f"Is Active: {self.is_current_activity}\n"
            )

        return output
