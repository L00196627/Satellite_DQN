import glob
import json


files = glob.glob("data/configurations_AP_*.json")

if not files:
    print("No generated configuration files found.")
    raise SystemExit


for path in files:

    with open(path, "r") as file:
        data = json.load(file)

    max_access_point_candidates = 0
    max_all_candidates = 0

    max_access_point_name = None
    max_satellite_name = None

    for config_entry in data["configurations"]:

        for satellite in config_entry["configuration"]:

            # SearchNode considers neighbours + the current SEN itself
            candidate_count = len(satellite["neighbors"]) + 1

            # Maximum count for any satellite
            if candidate_count > max_all_candidates:
                max_all_candidates = candidate_count
                max_satellite_name = satellite["satellite"]

            # Maximum count specifically for access-point SENs
            if satellite.get("is_access_point", False):

                if candidate_count > max_access_point_candidates:
                    max_access_point_candidates = candidate_count
                    max_access_point_name = satellite["satellite"]

    print()
    print(f"File: {path}")
    print(
        f"Maximum candidates for an access point SEN: "
        f"{max_access_point_candidates}"
    )
    print(
        f"Access point SEN: {max_access_point_name}"
    )
    print(
        f"Maximum candidates for any SEN: "
        f"{max_all_candidates}"
    )
    print(
        f"Satellite: {max_satellite_name}"
    )