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

    # prevent more candidates being supplied than the
    # neural network was designed to handle
    if len(server_metrics) > MAX_CANDIDATES:
        raise ValueError(
            f"Received {len(server_metrics)} candidates, "
            f"but MAX_CANDIDATES is {MAX_CANDIDATES}."
        )

    # preserve the order of the candidate SENs
    # DQN action 0 corresponds to candidate 0  etc
    ordered_metrics = list(server_metrics)

    state = []

    # initially assume that none of the 50 actions are valid
    mask = [False] * MAX_CANDIDATES

    # add the features for each real candidate SEN
    for i, metric in enumerate(ordered_metrics):

        state.extend([
            metric["waiting_cpu"],
            metric["waiting_net"],
            metric["transfer_time"],
            metric["orbitalSunset"],
            metric["elevation_angle"],
            metric["expected_completion_time"],
            metric["energy"],

            # 1 means that this candidate position contains a real SEN
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

    # add information describing the task itself
    state.extend([
        d_cpu,
        deadline,
        data_bytes_global,
        bw_Bps_global
    ])

    return state, mask, ordered_metrics