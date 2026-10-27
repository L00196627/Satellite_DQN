# COSMOS with DQN-Based Task Placement

This repository extends **COSMOS (Computing, Orbit, Satellite Modeling and Simulation)** with an experimental **Deep Q-Network (DQN)** task placement policy for Satellite Edge Computing (SEC).

COSMOS is a Python discrete event simulator for Low Earth Orbit (LEO) satellite edge computing. It models dynamic Starlink-based satellite topologies, Satellite Edge Nodes (SENs), task generation, queuing, energy consumption, inter-satellite routing and multiple task-placement strategies.

The DQN extension is intended to compare a learned task-placement policy against the simulator's existing **heuristic/ERT** and **Integer Linear Programming (ILP)** approaches under the same simulation conditions.


---

## Project Goal

The main scheduling question is:

> Given a task and the current state of the local satellite neighbourhood, which Satellite Edge Node should execute the task?

The project compares three approaches:

| Policy | Description |
|---|---|
| **ERT / Heuristic** | Existing COSMOS scheduling logic based on calculated candidate metrics and feasibility checks. |
| **ILP** | Integer Linear Programming formulation used as an optimisation-based comparison. |
| **DQN** | Deep reinforcement learning policy that learns which candidate SEN to select from simulator experience. |

Routing is treated separately from task placement. For controlled experiments, the routing algorithm should remain fixed while the placement policy changes.

---

## Main Components

### `main.py`


serves as the main driver for the LEO Satellite Edge Computing simulation.
It integrates discrete event simulation (SimPy) with orbital mechanics (Skyfield) and
optimization logic (ILP/Heuristics) to evaluate task offloading performance.


---

### `simulation.py`


contains the primary SimPy processes that drive the simulation's
operational phase. It handles the lifecycle of tasks from generation to completion.

The `SearchNode` configuration determines which placement strategy is used.


---

### `EdgeServer.py`

defines the `EdgeServer` class, which represents a Low Earth Orbit (LEO) satellite
acting as an edge computing node within the simulation environment.

The module integrates discrete event simulation (SimPy) with orbital mechanics (Skyfield)
to simulate a dynamic satellite network.

The scheduling policy chooses an `EdgeServer`. The server then participates in the normal task execution process.

---

### `ILP_simulation.py`

implements the logic for assigning and executing tasks within a satellite
network using an optimization-based approach (ILP).

It handles the lifecycle of a task from the decision phase (searching for the best node)
to the execution phase (CPU processing and Network transmission).


The underlying optimisation formulation is implemented primarily in:


---

## DQN Extension

The DQN implementation is deliberately separated into small components.

### `DQN_data_Translator.py`

Converts live COSMOS scheduling information into a fixed-size numeric state for the neural network. For each candidate SEN, it includes CPU and network waiting time, transfer time, orbital sunset, elevation angle, expected completion time, energy, and whether the candidate exists. It also includes task information such as CPU requirement, deadline, data size, and bandwidth.

The current design supports up to **50 candidate SENs**.

With 8 values per candidate and 4 task level values, 50 * 8 +4 = 404 state values. Therefore the state size is 404 and the action is is 8. 

Unused candidate positions are padded and masked so that the DQN cannot select them.

---

### `DQN_simulation.py`

Acts as the integration layer between COSMOS and the DQN. It manages the interaction between live simulation data, DQN decision-making, normal COSMOS task execution, and the reward returned to the learning system.

This module collects the current task and candidate SEN information, passes this data to `DQN_data_Translator.py`, requests an action from `DQN_agent.py` and maps the selected action to a real candidate SEN. The chosen SEN is then used by the normal COSMOS simulation logic.

After execution, the task outcome is converted into a reward. Because tasks can overlap and may complete out of order, the module maintains pending transitions until the required reward and next state information are available. Once a transition is complete, it is added to replay memory and used for DQN training.

`DQN_simulation.py` therefore provides the connection between the event driven COSMOS simulator and the standard DQN learning process.

---

### `DQN_agent.py`

Contains the core reinforcement learning logic, including the policy and target networks, replay memory, action selection and training updates. The network receives the current state and outputs a Q value for each possible candidate SEN.


---

### `DQN_test.py`

A development test for the DQN pipeline that uses sample task and candidate data to check that state creation, action masking, action selection, replay memory and training all work correctly.

It is **not** a COSMOS performance experiment and its rewards/loss values should not be treated as dissertation results.


---

### `find_max_candidates.py`

Utility used to determine the maximum candidate-set size found in the saved COSMOS topology data. This data can be obtained by running the simulator. 

The scan observed that there was a maximum candidate count of 50, with 51 including the current SEN. Therefore the max candidates variable is set at 50. 

If DQN scheduling is later allowed to begin from arbitrary forwarded SENs, this value should be reviewed.


---

## Installation


### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install COSMOS dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch for the DQN extension

If PyTorch is not already included in `requirements.txt`:

```bash
pip install torch
```

---

## Configuration

The main simulator settings are stored in config.json5. 

---

## Building the Satellite Configuration

Before running the experiments for the first time, generate the saved orbital/topology configurations.

Set:

```json5
Build_Configurations: true,
Load_Configuration: false
```

Then run:

```bash
python main.py
```

This may take considerably longer than a normal simulation run. Please allow 1 to 2 hours for the data to be generated. 

After the data files have been generated, switch to:

```json5
Build_Configurations: false,
Load_Configuration: true
```

The saved data can then be reused for later experiments.


---

## Output

Simulation results are written to the result/ directory.

COSMOS creates parameter-specific output folders containing CSV files like summary, Per task results and simulation results and energy statistics. 


---

## Routing

Task placement and routing are separate decisions.

Task placement answers:

> Which SEN should execute the task?

Routing answers:

> How should the task data reach that SEN?

COSMOS supports routing options including:

- BATMAN
- GREEDY
- DSR

For placement-policy experiments, a fixed routing algorithm should be used across all compared methods.



---

## Notes

- The DQN action mask currently distinguishes real candidate slots from padded slots.
- Feasibility constraints such as resource, deadline and orbital/sunset conditions should be handled consistently when comparing policies.
- The candidate ordering supplied to the DQN should remain deterministic during training and evaluation.
- `find_max_candidates.py` is a development utility and does not need to run during every simulation.
- `DQN_test.py` is a component test rather than an experimental benchmark.

---

## Acknowledgements

This project builds on the existing COSMOS satellite edge computing simulator written by Danilo Magliarisi and extends it with a DQN-based task-placement experiment.


