"""
A development test for the DQN pipeline that uses sample task and candidate data to check that state creation,
action masking, action selection, replay memory, and training all work correctly.

It is not a COSMOS performance experiment and its rewards/loss values should not be treated as dissertation results.
"""



from DQN_data_Translator import build_dqn_state, STATE_SIZE, ACTION_SIZE


class FakeServer:
    def __init__(self, name):
        self.name = name


server_metrics = [
    {
        "server": FakeServer("SEN_A"),
        "waiting_cpu": 0.5,
        "waiting_net": 0.2,
        "transfer_time": 0.1,
        "orbitalSunset": 20.0,
        "elevation_angle": 45.0,
        "expected_completion_time": 1.5,
        "energy": 95000.0
    },
    {
        "server": FakeServer("SEN_B"),
        "waiting_cpu": 1.0,
        "waiting_net": 0.4,
        "transfer_time": 0.3,
        "orbitalSunset": 15.0,
        "elevation_angle": 30.0,
        "expected_completion_time": 2.2,
        "energy": 80000.0
    },
    {
        "server": FakeServer("SEN_C"),
        "waiting_cpu": 0.2,
        "waiting_net": 0.1,
        "transfer_time": 0.2,
        "orbitalSunset": 25.0,
        "elevation_angle": 55.0,
        "expected_completion_time": 1.0,
        "energy": 99000.0
    }
]


state, valid_action_mask, ordered_metrics = build_dqn_state(
    server_metrics=server_metrics,
    d_cpu=0.5,
    deadline=10,
    data_bytes_global=500_000_000,
    bw_Bps_global=35_000_000
)


print("Expected state size:", STATE_SIZE)
print("Actual state size:", len(state))

print("Expected action size:", ACTION_SIZE)
print("Mask size:", len(valid_action_mask))

print("Number of valid actions:", sum(valid_action_mask))

print("First 10 mask values:")
print(valid_action_mask[:10])

print("Candidate order:")
for i, metric in enumerate(ordered_metrics):
    print(i, metric["server"].name)

print("First candidate state values:")
print(state[:8])


from DQN_agent import select_action, replay, memory

#build training conditions
#this is the state and the valid action mask
print("\ntesting DQN training...")

for i in range(150):

    action = select_action(
        state,
        valid_action_mask
    ).item()

    if action == 2:
        reward = 1.0
    else:
        reward = 0.0

    next_state = state.copy()
    next_valid_action_mask = valid_action_mask.copy()

    done = False

    memory.push(
        state,
        action,
        next_state,
        reward,
        next_valid_action_mask,
        done
    )


loss = replay()

print("Experiences stored:", len(memory))
print("Training loss:", loss)
print("DQN training test completed")