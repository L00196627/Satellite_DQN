"""
This module converts live COSMOS scheduling information into a fixed-size numeric state for the neural network.
For each candidate SEN, it includes CPU and network waiting time, transfer time, orbital sunset, elevation angle,
expected completion time, energy, and whether the candidate exists.

It also includes task information such as CPU requirement, deadline, data size, and bandwidth.

The current design supports up to 50 candidate SENs.
"""




import math

# maximum number of candidate SENs observed for an Access Point SEN
MAX_CANDIDATES = 50

# each candidate SEN is represented by:
# CPU wait, network wait, transfer time, orbital sunset,
# elevation angle, expected completion time, energy,
# and a flag showing whether the candidate exists
FEATURES_PER_CANDIDATE = 8

# task information:
# CPU requirement, deadline, data size, bandwidth
TASK_FEATURES = 4


# Total DQN input size:
# (50 candidates * 8 features) + 4 task features = 404
STATE_SIZE = (
    MAX_CANDIDATES * FEATURES_PER_CANDIDATE
    + TASK_FEATURES
)

# the DQN can choose one of the 50 candidate positions
ACTION_SIZE = MAX_CANDIDATES


MAX_DEADLINE = 10.0
MAX_ENERGY = 100000.0
MAX_ELEVATION = 90.0
MAX_ORBITAL_TIME = 300.0
MAX_DATA_BYTES = 500_000_000.0
MAX_BANDWIDTH = 100_000_000.0


def build_dqn_state(
        server_metrics,
        d_cpu,
        deadline,
        data_bytes_global,
        bw_Bps_global
):
    """
    convert COSMOS server metrics into a fixed size state
    that can be supplied to the DQN

    the function also creates an action mask identifying
    which candidate SEN positions are currently valid
    """

    # prevent more candidates being supplied than the neural network was designed to handle

    if len(server_metrics) > MAX_CANDIDATES:
        raise ValueError(
            f"Received {len(server_metrics)} candidates, "
            f"but MAX_CANDIDATES is {MAX_CANDIDATES}."
        )

    # preserve the order of the candidate SENs DQN action 0 corresponds to candidate 0  etc

    valid_metrics = []

    for metric in server_metrics:
        candidate_values = [
            metric["waiting_cpu"],
            metric["waiting_net"],
            metric["transfer_time"],
            metric["orbitalSunset"],
            metric["elevation_angle"],
            metric["expected_completion_time"],
            metric["energy"]
        ]

        if all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in candidate_values
        ):
            valid_metrics.append(metric)

    if not valid_metrics:
        raise ValueError("No finite DQN candidate SENs available.")

    ordered_metrics = valid_metrics

    state = []

    # initially assume that none of the 50 actions are valid
    mask = [False] * MAX_CANDIDATES

    # add the features for each real candidate SEN divided by max values for normalization
    for i, metric in enumerate(ordered_metrics):


        # candidate features:
        # waiting_cpu - CPU queue delay
        # waiting_net - network queue delay
        # transfer_time - task transfer time
        # orbitalSunset - remaining visibility time
        # elevation_angle - satellite elevation
        # expected_completion_time - predicted finish time
        # energy - remaining SEN energy
        # 1.0 - candidate exists


        state.extend([
            metric["waiting_cpu"]/MAX_DEADLINE,
            metric["waiting_net"]/MAX_DEADLINE,
            metric["transfer_time"]/MAX_DEADLINE,
            metric["orbitalSunset"]/MAX_ORBITAL_TIME,
            metric["elevation_angle"]/MAX_ELEVATION,
            metric["expected_completion_time"]/MAX_DEADLINE,
            metric["energy"]/MAX_ENERGY,
            1.0
        ])

        # allow the DQN to select this candidate
        mask[i] = True

    # fill unused candidate positions with zeros so that
    # every state always has the same size
    missing_candidates = (
        MAX_CANDIDATES - len(ordered_metrics)
    )

    state.extend(
        [0.0] * (
            missing_candidates
            * FEATURES_PER_CANDIDATE
        )
    )

    # add information describing the task itself divided by max values for normalization
    # task features:
    # d_cpu - CPU demand
    # deadline - task deadline
    # data_bytes_global - task data size
    # bw_Bps_global - available bandwidth
    state.extend([
        d_cpu/MAX_DEADLINE,
        deadline/MAX_DEADLINE,
        data_bytes_global/MAX_DATA_BYTES,
        bw_Bps_global/MAX_BANDWIDTH
    ])

    for i, value in enumerate(state):
        if not math.isfinite(value):
            raise ValueError(
                f"Invalid DQN state value at index {i}: {value}"
            )

    return state, mask, ordered_metrics