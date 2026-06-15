import numpy as np


class Vessel:
    """
    Represents a vessel with sailing speed and fuel consumption characteristics.

    This class initializes a vessel object from a dictionary of parameters, calculating
    possible sailing speeds and corresponding fuel consumption for different operational modes.
    It also stores economic parameters such as day rate while mobilizing.
    """

    def __init__(self, vessel_dict: dict) -> None:
        """
        Initialize a Vessel instance from a dictionary of vessel attributes.

        Parameters:
        vessel_dict (dict): A dictionary containing vessel parameters.
            - vessel_name (str): Name of the vessel.
            - IMO (int): IMO number of the vessel.
            - sailing_speed_eco (float): Economical sailing speed in knots.
            - sailing_speed_cruise (float): Cruise sailing speed in knots.
            - sailing_speed_max (float): Maximum sailing speed in knots.
            - fuel_consumed_eco (float): Fuel consumed at economical speed [mt/24hr].
            - fuel_consumed_cruise (float): Fuel consumed at cruise speed [mt/24hr].
            - fuel_consumed_max (float): Fuel consumed at max speed [mt/24hr].
            - day_rate_mob (float): Day rate while mobilizing in USD.

        Raises:
        KeyError: If any required parameters are missing.
        """

        # check that the dictionary has all the required input parameters
        required_keys = [
            "vessel_name",
            "vessel_IMO",
            "sailing_speed_eco",
            "sailing_speed_cruise",
            "sailing_speed_max",
            "fuel_consumed_eco",
            "fuel_consumed_cruise",
            "fuel_consumed_max",
            "day_rate_mob",
        ]

        missing_keys = [key for key in required_keys if key not in vessel_dict]

        if missing_keys:
            raise KeyError(f"Missing required vessel parameters: {', '.join(missing_keys)}")

        # get vessel name and IMO
        self.vessel_name = vessel_dict.get("vessel_name", "")
        self.IMO = vessel_dict.get("IMO")
        self.asset_id = vessel_dict.get("asset_id", "")

        # get default fuel costs
        self.default_fuel_price = vessel_dict.get("default_fuel_price", 600)

        self.get_speed_fuel_arrays(vessel_dict)

        # day rate while mobilizing [USD]
        self.day_rate_mob = vessel_dict.get("day_rate_mob", 0)

    def get_speed_fuel_arrays(self, vessel_dict):
        """
        Generate possible speeds array and corresponding fuel consumptions.
        """
        # get sailing speeds [kt]
        self.sailing_speed_eco = vessel_dict.get("sailing_speed_eco", 15)
        self.sailing_speed_cruise = vessel_dict.get("sailing_speed_cruise", 15)
        self.sailing_speed_max = vessel_dict.get("sailing_speed_max", 15)

        self.possible_speeds = np.arange(self.sailing_speed_eco, self.sailing_speed_max + 0.5, 0.5)

        # corresponding fuel consumption [mt/24hr]
        self.fuel_consumed_eco = vessel_dict.get("fuel_consumed_eco", 0)
        self.fuel_consumed_cruise = vessel_dict.get("fuel_consumed_cruise", 0)
        self.fuel_consumed_max = vessel_dict.get("fuel_consumed_max", 0)

        if self.sailing_speed_max > self.sailing_speed_cruise:
            dydx = (self.fuel_consumed_max - self.fuel_consumed_cruise) / (self.sailing_speed_max - self.sailing_speed_cruise)

            corresp_fuel_cruise_max = np.array(dydx * (self.possible_speeds - self.sailing_speed_cruise) + self.fuel_consumed_cruise)
        else:
            corresp_fuel_cruise_max = np.zeros_like(self.possible_speeds) + self.fuel_consumed_cruise

        if self.sailing_speed_cruise > self.sailing_speed_eco:
            dydx = (self.fuel_consumed_cruise - self.fuel_consumed_eco) / (self.sailing_speed_cruise - self.sailing_speed_eco)

            corresp_fuel_eco_cruise = np.array(dydx * (self.possible_speeds - self.sailing_speed_eco) + self.fuel_consumed_eco)
        else:
            corresp_fuel_eco_cruise = np.zeros_like(self.possible_speeds) + self.fuel_consumed_eco

        # assign
        self.corresponding_fuel = np.where(self.possible_speeds <= self.sailing_speed_cruise, corresp_fuel_eco_cruise, corresp_fuel_cruise_max)

    def to_dict(self) -> dict:
        """
        Returns a dictionary with all information regarding the vessel object.
        """
        vessel_dict = {
            "vessel_name": self.vessel_name,
            "IMO": self.IMO,
            "sailing_speed_eco": self.sailing_speed_eco,
            "sailing_speed_cruise": self.sailing_speed_cruise,
            "sailing_speed_max": self.sailing_speed_max,
            "possible_speeds": self.possible_speeds.tolist(),
            "fuel_consumed_eco": self.fuel_consumed_eco,
            "fuel_consumed_cruise": self.fuel_consumed_cruise,
            "fuel_consumed_max": self.fuel_consumed_max,
            "corresponding_fuel": self.corresponding_fuel.tolist(),
            "day_rate_mob": self.day_rate_mob,
        }

        return vessel_dict

    def __repr__(self) -> str:
        """
        Return a minimal string representation of the Vessel object.

        Returns:
        str: The vessel's name.
        """
        return self.vessel_name

    def __str__(self) -> str:
        """
        Return a comprehensive string representation of the Vessel object.

        Returns:
        str: Full vessel details.
        """
        output = (
            f"Vessel Name: {self.vessel_name}\n"
            + f"IMO: {self.IMO}\n"
            + f"Eco: {self.sailing_speed_eco} kn ({self.fuel_consumed_eco} T/day)\n"
            + f"Cruise: {self.sailing_speed_cruise} kn ({self.fuel_consumed_cruise} T/day)\n"
            + f"Max: {self.sailing_speed_max} kn ({self.fuel_consumed_max} T/day)\n"
            + f"Day Rate: {self.day_rate_mob} euro\n"
        )
        return output