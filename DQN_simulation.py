"""
This module acts as the bridge between COSMOS and the DQN.
It takes the current task and candidate satellite information from COSMOS,
sends that data through the translator, and asks the DQN which SEN to choose

after the task is executed, it checks the result, calculates the reward, manages any pending transitions
and sends the completed experience into replay memory for training
 """




from DQN_data_Translator import build_dqn_state, STATE_SIZE, ACTION_SIZE
from DQN_agent import select_action, memory, replay

# set penalty for task failure in training
FAILURE_PENALTY = -1.0


# stores DQN decisions that are waiting for a reward or next state
pending_transitions = {}

# tracks the most recent DQN scheduling decision
last_decision_task_id = None


def _try_store_transition(task_id):
    transition = pending_transitions.get(task_id)

    if transition is None:
        return

    if transition["reward"] is None or transition["next_state"] is None:
        print(
            f"[DQN PENDING] Task {task_id} waiting | "
            f"Reward known: {transition['reward'] is not None} | "
            f"Next state known: {transition['next_state'] is not None}"
        )

    if transition["reward"] is None:
        return

    if transition["next_state"] is None:
        return

    memory.push(
        transition["state"],
        transition["action"],
        transition["next_state"],
        transition["reward"],
        transition["next_valid_action_mask"],
        transition["done"]
    )

    loss = replay()

    if loss is None:
        print(
            f"[DQN TRAIN] Task {task_id} stored | "
            f"Replay memory: {len(memory)}"
        )
    else:
        print(
            f"[DQN TRAIN] Task {task_id} stored | "
            f"Replay memory: {len(memory)} | "
            f"Loss: {loss:.6f}"
        )

    del pending_transitions[task_id]


def attach_next_state(state, valid_action_mask):
    global last_decision_task_id

    if last_decision_task_id is None:
        return

    previous_transition = pending_transitions.get(
        last_decision_task_id
    )

    if previous_transition is None:
        return

    if previous_transition["next_state"] is not None:
        return

    previous_transition["next_state"] = list(state)

    previous_transition["next_valid_action_mask"] = list(
        valid_action_mask
    )

    try_store_transition(last_decision_task_id)


def record_reward(task_id, reward):
    transition = pending_transitions.get(task_id)

    if transition is None:
        return

    transition["reward"] = float(reward)

    try_store_transition(task_id)



def calculate_reward(completed, rejected, deadline):

    # Task completed
    if completed is not None:
        service_time = float(completed[7])

        # COSMOS stores the after-sunset flag at index 15
        exec_after_sunset = bool(completed[15])

        # Completing after sunset is a hard failure
        if exec_after_sunset:
            return FAILURE_PENALTY, "sunset_failure"

        # Small latency penalty.
        # Always between 0 and 0.25.
        latency_penalty = 0.25 * (
            service_time / (1.0 + service_time)
        )

        # A deadline violation is still a successful completion
        # in COSMOS, but receives a lower reward.
        if service_time > float(deadline):
            reward = 0.5 - latency_penalty
            return reward, "deadline_violation"

        # Successful completion before deadline and sunset
        reward = 1.0 - latency_penalty
        return reward, "success"

    # Task could not be successfully allocated/executed
    if rejected is not None:
        return FAILURE_PENALTY, "rejected"

    # Outcome is not known yet
    return None, "pending"


def SearchNode_DQN_v1(
    env,
    server,
    task_id,
    required_ram,
    required_disk,
    image_size,
    Volume_size,
    arrival_time_system,
    initial_server_counter,
    different_server_counter,
    other_server_counter,
    task_type,
    max_energy,
    d_cpu,
    deadline
):
    global last_decision_task_id

    # Imported here so simulation.py can import DQN_simulation.py
    # without creating a circular import at startup
    import simulation

    print(
        f"--- [{env.now:.2f}] Task {task_id} "
        f"using SearchNode Policy: DQN ---"
    )

    # Current SEN + its neighbours
    neighbors_at_distance_one = list(server.get_neighbors())
    neighbors_at_distance_one.append(server)

    # Current network/task information
    bw_Bps_global, data_bytes_global = simulation.network_metrics(
        image_size,
        Volume_size_MB=Volume_size
    )

    d_net_predicted = (
        data_bytes_global / bw_Bps_global
        if bw_Bps_global > 0
        else float("inf")
    )

    # Calculate live information for each candidate SEN
    server_metrics = simulation._calculate_heuristic_metrics(
        neighbors_at_distance_one,
        server,
        image_size,
        Volume_size,
        task_type,
        d_cpu,
        deadline,
        max_energy,
        d_net_predicted,
        data_bytes_global,
        bw_Bps_global
    )

    # Remove candidates without valid orbital sunset information
    server_metrics = [
        metric
        for metric in server_metrics
        if metric["orbitalSunset"] not in (None, 0)
    ]

    if not server_metrics:
        raise ValueError(
            "No valid DQN candidate SENs available."
        )

    # Build the current DQN state
    state, valid_action_mask, ordered_metrics = build_dqn_state(
        server_metrics,
        d_cpu,
        deadline,
        data_bytes_global,
        bw_Bps_global
    )

    # This state is the next_state of the previous DQN decision
    attach_next_state(
        state,
        valid_action_mask
    )

    # Ask the DQN to select a candidate
    action = select_action(
        state,
        valid_action_mask
    )

    # Convert action number back into an actual SEN
    chosen_metric = ordered_metrics[action]
    chosen_server = chosen_metric["server"]

    transfer_time = chosen_metric.get(
        "transfer_time",
        0.0
    )

    print(
        f"--- [{env.now:.2f}] Task {task_id} DQN selected "
        f"{chosen_server.name} (action {action}) ---"
    )

    # Save this decision while we wait for reward and next_state
    pending_transitions[task_id] = {
        "state": list(state),
        "action": int(action),
        "reward": None,
        "next_state": None,
        "next_valid_action_mask": None,
        "done": False
    }

    last_decision_task_id = task_id

    # Continue through normal COSMOS execution
    yield from simulation._finalize_and_assign_task(
        env,
        server,
        chosen_server,
        task_id,
        image_size,
        Volume_size,
        arrival_time_system,
        transfer_time,
        task_type,
        d_cpu,
        deadline,
        required_ram,
        required_disk,
        initial_server_counter,
        different_server_counter,
        other_server_counter
    )

    # Check whether the task completed
    completed = next(
        (
            task
            for task in chosen_server.completed_tasks
            if task[0] == task_id
        ),
        None
    )

    # Check whether the task was rejected
    rejected = next(
        (
            task
            for task in chosen_server.rejected_tasks
            if task[0] == task_id
        ),
        None
    )

    # Some routing failures may be recorded on the original SEN
    if rejected is None and server is not chosen_server:
        rejected = next(
            (
                task
                for task in server.rejected_tasks
                if task[0] == task_id
            ),
            None
        )

    reward, outcome = calculate_reward(
        completed,
        rejected,
        deadline
    )

    if outcome == "success":
        service_time = completed[7]

        print(
            f"[DQN] Task {task_id} completed | "
            f"Service time: {service_time:.3f} | "
            f"Reward: {reward:.3f}"
        )

        record_reward(
            task_id,
            reward
        )

    elif outcome == "deadline_violation":
        service_time = completed[7]

        print(
            f"[DQN] Task {task_id} completed with deadline violation | "
            f"Service time: {service_time:.3f} | "
            f"Deadline: {deadline:.3f} | "
            f"Reward: {reward:.3f}"
        )

        record_reward(
            task_id,
            reward
        )

    elif outcome == "sunset_failure":
        print(
            f"[DQN] Task {task_id} failed after sunset | "
            f"Reward: {reward:.3f}"
        )

        record_reward(
            task_id,
            reward
        )

    elif outcome == "rejected":
        print(
            f"[DQN] Task {task_id} rejected | "
            f"Reward: {reward:.3f}"
        )

        record_reward(
            task_id,
            reward
        )


def finish_episode():
    global last_decision_task_id

    # Any task still pending when the simulation ends is treated
    # as a failure because it did not complete within the
    # simulation horizon.
    for task_id in list(pending_transitions.keys()):

        transition = pending_transitions.get(task_id)

        if transition is None:
            continue

        # Task never produced a completed/rejected outcome
        if transition["reward"] is None:
            transition["reward"] = FAILURE_PENALTY

            print(
                f"[DQN] Task {task_id} unfinished at simulation end | "
                f"Reward: {FAILURE_PENALTY:.3f}"
            )

        # If this was the final scheduling decision, there is
        # no later state, so make it a terminal transition.
        if transition["next_state"] is None:
            transition["next_state"] = (
                [0.0] * STATE_SIZE
            )

            transition["next_valid_action_mask"] = (
                [False] * ACTION_SIZE
            )

            transition["done"] = True

        # The transition now has everything required
        # for replay memory.
        try_store_transition(task_id)

    pending_transitions.clear()
    last_decision_task_id = None