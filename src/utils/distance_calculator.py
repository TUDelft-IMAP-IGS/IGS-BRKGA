import searoute as sr


def calculate_sailing_distance(
    lon_init: float,
    lat_init: float,
    lon_final: float,
    lat_final: float,
    restrictions: list = ["suez", "northwest"],
    return_coords: bool = False,
) -> tuple[float, list | None]:
    """
    Paramters:
    lon_init (float): Origin longitude (-180 < lon < 180)
    lat_init (float): Origin latitude (-90 < lon < 90)
    lon_final (float): Destination longitude (-180 < lon < 180)
    lat_final (float): Destination latitude (-90 < lon < 90)
    restrictions (list): List of restricted routes.

    Return:
    dist (float): Distance travelled in nautical miles.
    coords (list): Waypoints of route travelled [(lon, lat)]***

    *** Returns None if return_coords == False
    """

    # generate route based on searoute package
    route = sr.searoute(
        origin=(lon_init, lat_init),
        destination=(lon_final, lat_final),
        units="naut",
        restrictions=restrictions,
    )

    # extract useful information from route
    dist = route.properties.get("length")

    if return_coords:
        coords = route.geometry.coordinates
        return dist, coords
    else:
        # only return distance and duration
        return dist, None


def calculate_mobilization_matrix(activities: list, vessels: list):
    """
    TO DO:
    Make doc string
    """
    # create a list of all speeds
    all_speeds = []

    # max distance
    max_distance = 0

    for v in vessels:
        all_speeds.extend(v.possible_speeds)

    all_speeds = list(set(all_speeds))

    # dictionary of all locations
    location_starts = {}
    location_ends = {}

    for activity in activities:
        if activity.is_tow() or activity.is_current_location:
            # extract start and end location
            location_starts[f"{activity.activity_name}_start"] = activity.start_location
            location_ends[f"{activity.activity_name}_end"] = activity.end_location

        elif activity.is_maintenance():
            # extract possible locations
            for idx, loc in enumerate(activity.allowed_locations):
                # start and end location are the same
                location_starts[f"{activity.activity_name}_location_{idx}"] = loc
                location_ends[f"{activity.activity_name}_location_{idx}"] = loc

    # compute all possible mobilization times between all tasks based on speed
    durations = {}

    for location1 in location_ends:
        # extract coordinates
        lon1, lat1 = location_ends[location1]
        task1 = location1.split("_")[0]

        durations[location1] = {}

        for location2 in location_starts:
            # extract coordinates
            lon2, lat2 = location_starts[location2]
            task2 = location2.split("_")[0]

            if task1 == task2:
                continue
            else:
                dist, _ = calculate_sailing_distance(lon1, lat1, lon2, lat2)

            durations[location1][location2] = {}

            durations[location1][location2][0] = dist

            max_distance += dist

            for speed in all_speeds:
                durations[location1][location2][speed] = dist / (speed * 24)

    return durations, max_distance
