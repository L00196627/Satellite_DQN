"""
Contains the core reinforcement-learning logic, including the policy and target networks,
replay memory, action selection, and training updates.
The network receives the current state and outputs a Q-value for each possible candidate SEN.

The core DQN structure follows the official PyTorch reinforcement learning tutorial,
 with COSMOS specific state translation and action masking added for satellite task placement.

 https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""




import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN_data_Translator import STATE_SIZE, ACTION_SIZE

"""
 BATCH_SIZE is the number of transitions sampled from the replay buffer
 GAMMA is the discount factor as mentioned in the previous section
 EPS_START is the starting value of epsilon
 EPS_END is the final value of epsilon
 EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
 TAU is the update rate of the target network
 LR is the learning rate of the ``AdamW`` optimizer
"""


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "next_state",
        "reward",
        "next_valid_action_mask",
        "done"
    )
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)

target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(
    policy_net.parameters(),
    lr=LR,
    amsgrad=True
)

memory = ReplayMemory(10000)

loss_fn = nn.SmoothL1Loss()

steps_done = 0

# uses what the network has learned to choose a SEN
def select_action(state, valid_action_mask):

    global steps_done
    # generate a random value used for epsilon-greedy action selection
    sample = random.random()

    # calculate the current exploration rate
    # epsilon gradually decreases as more decisions are made
    eps_threshold = (
        EPS_END
        + (EPS_START - EPS_END)
        * math.exp(-1.0 * steps_done / EPS_DECAY)
    )

    steps_done += 1
    # find the action indexes that correspond to valid candidate SENs
    valid_actions = [
        i
        for i, valid in enumerate(valid_action_mask)
        if valid
    ]

    if not valid_actions:
        raise ValueError("No valid DQN actions are available.")
    # use the policy network to select the best action
    if sample > eps_threshold:

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # gradients are not needed when only selecting an action
        with torch.no_grad():
            # mask invalid or padded candidate actions
            q_values = policy_net(state)

            mask = torch.BoolTensor(
                valid_action_mask
            ).to(device)

            q_values[0, ~mask] = float("-inf")
            # select the valid action with the highest predicted Q-value
            return q_values.max(1).indices.item()

    else:

        return random.choice(valid_actions)

# train the DQN using a random batch of past experiences
# stored in replay memory
def replay():

    if len(memory) < BATCH_SIZE:
        return

    minibatch = memory.sample(BATCH_SIZE)

    states, actions, next_states, rewards, next_masks, dones = zip(
        *minibatch
    )

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    next_masks = torch.BoolTensor(next_masks).to(device)
    dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

    # Current Q values
    current_q = policy_net(states).gather(1, actions)

    # Target Q values
    next_q_values = target_net(next_states).masked_fill(
        ~next_masks,
        float("-inf")
    )

    next_q = next_q_values.max(1)[0].detach().unsqueeze(1)
    next_q[dones] = 0.0

    # Bellman equation
    target_q = rewards + (GAMMA * next_q)

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    update_target_network()

    return loss.item()

# updates target network to follow policy network
def update_target_network():

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = (
            policy_net_state_dict[key] * TAU
            + target_net_state_dict[key] * (1 - TAU)
        )

    target_net.load_state_dict(target_net_state_dict)
# saves learned information for subsequent training
def save_model(filename="dqn_model.pth"):
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps_done": steps_done
    }, filename)

    print(f"[DQN] Model saved to {filename}")


def load_model(filename="dqn_model.pth"):
    global steps_done
    # try to load a previously saved DQN checkpoint
    # map_location ensures the model is loaded onto the current device
    try:
        checkpoint = torch.load(
            filename,
            map_location=device
        )
    except FileNotFoundError:
        print("[DQN] No saved model found. Starting new model.")
        return
    # restore the learned weights of the main policy network
    policy_net.load_state_dict(
        checkpoint["policy_net"]
    )
    # restore the target network weights used during DQN training
    target_net.load_state_dict(
        checkpoint["target_net"]
    )
    # restore the optimizer state so training can continue normally
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    # restore the number of action-selection steps completed so far
    # this keeps the epsilon-decay schedule consistent across runs
    steps_done = checkpoint["steps_done"]

    print(f"[DQN] Model loaded from {filename}")