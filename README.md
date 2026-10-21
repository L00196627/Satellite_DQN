# COSMOS with DQN-Based Task Placement

This repository extends **COSMOS (Computing, Orbit, Satellite Modeling and Simulation)** with an experimental **Deep Q-Network (DQN)** task placement policy for Satellite Edge Computing (SEC).

COSMOS is a Python/SimPy discrete-event simulator for Low Earth Orbit (LEO) satellite edge computing. It models dynamic Starlink-based satellite topologies, Satellite Edge Nodes (SENs), task generation, queuing, energy consumption, inter-satellite routing, and multiple task-placement strategies.

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

## High-Level Architecture

```text
main.py
   |
   |-- loads configuration
   |-- builds/loads topology
   |-- creates the simulation environment
   |-- starts task generation
   |
   v
simulation.py
   |
   |-- generates tasks
   |-- calculates candidate SEN metrics
   |-- selects SearchNode policy
   |
   +---- ERT / Heuristic
   |
   +---- ILP ----> ILP_simulation.py
   |                  |
   |                  +----> sec_ilp_snapshot_v3.py
   |
   +---- DQN ----> DQN_data_Translator.py
                     |
                     +----> DQN_agent.py
                              |
                              +----> chosen SEN
```

After a target SEN is selected, the normal COSMOS task-assignment and execution logic continues.

---

## Main Components

### `main.py`


serves as the main driver for the LEO Satellite Edge Computing simulation.
It integrates discrete-event simulation (SimPy) with orbital mechanics (Skyfield) and
optimization logic (ILP/Heuristics) to evaluate task offloading performance.

Run the project with:

```bash
python main.py
```

---

### `simulation.py`


contains the primary SimPy processes that drive the simulation's
operational phase. It handles the lifecycle of tasks from generation to completion.

The `SearchNode` configuration determines which placement strategy is used.

Conceptually:

```python
if policy == "ILP":
    ...
elif policy == "DQN":
    ...
else:
    # ERT / heuristic
    ...
```

---

### `EdgeServer.py`

defines the `EdgeServer` class, which represents a Low Earth Orbit (LEO) satellite
acting as an edge computing node within the simulation environment.

The module integrates discrete event simulation (SimPy) with orbital mechanics (Skyfield)
to simulate a dynamic satellite network.

The scheduling policy chooses an `EdgeServer`; the server then participates in the normal task execution process.

---

### `ILP_simulation.py`

implements the logic for assigning and executing tasks within a satellite
network using an optimization-based approach (ILP).

It handles the lifecycle of a task from the decision phase (searching for the best node)
to the execution phase (CPU processing and Network transmission).


The underlying optimisation formulation is implemented primarily in:

```text
sec_ilp_snapshot_v3.py
```

---

## DQN Extension

The DQN implementation is deliberately separated into small components.

### `DQN_data_Translator.py`

Converts live COSMOS scheduling information into a fixed-size numeric state for the neural network. For each candidate SEN, it includes CPU and network waiting time, transfer time, orbital sunset, elevation angle, expected completion time, energy, and whether the candidate exists. It also includes task information such as CPU requirement, deadline, data size, and bandwidth.

The current design supports up to **50 candidate SENs**.

With 8 values per candidate and 4 task-level values:

```text
50 * 8 + 4 = 404 state values
```

Therefore:

```python
STATE_SIZE = 404
ACTION_SIZE = 50
```

Unused candidate positions are padded and masked so that the DQN cannot select them.

---

### `DQN_agent.py`

Contains the core reinforcement-learning logic, including the policy and target networks, replay memory, action selection, and training updates. The network receives the current state and outputs a Q-value for each possible candidate SEN.


The core DQN structure follows the official PyTorch reinforcement-learning tutorial, with COSMOS-specific state translation and action masking added for satellite task placement.

PyTorch tutorial:

https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

---

### `DQN_test.py`

A development test for the DQN pipeline that uses sample task and candidate data to check that state creation, action masking, action selection, replay memory, and training all work correctly.

It is **not** a COSMOS performance experiment and its rewards/loss values should not be treated as dissertation results.

Run it with:

```bash
python DQN_test.py
```

---

### `find_max_candidates.py`

Utility used to determine the maximum candidate-set size found in the saved COSMOS topology data. This data can be obtained by running the simulator. 

A scheduling candidate set consists of:

```text
current SEN + directly connected neighbours
```

The scan observed:

```text
Maximum Access Point candidate count: 50
Maximum candidate count for any SEN: 51
```

The current DQN uses:

```python
MAX_CANDIDATES = 50
```

because the current placement decision begins from an Access Point SEN.

If DQN scheduling is later allowed to begin from arbitrary forwarded SENs, this value should be reviewed.

Run with:

```bash
python find_max_candidates.py
```

---

## Repository Structure
> *It should be noted that the 'data' and 'result' folders are generated after the simulator has ran. 

```text
COSMOS/
- main.py
- simulation.py
-  EdgeServer.py
-  Task.py
-  Satellite.py
-  Observer.py
-  topology.py
-  routing_Manager.py
- simulation_OGM.py
-  Ogm.py
- packet.py
-  globals.py
-  utils.py

- ILP_simulation.py
-  sec_ilp_snapshot_v3.py

-  DQN_data_Translator.py
- DQN_agent.py
- DQN_test.py
-  find_max_candidates.py

-  config.json5
-  img_resolution.json5
-  requirements.txt

-  data/
-  result/
```



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

The main simulator settings are stored in:

```text
config.json5
```

Important options include:

```json5
simulation_duration: 300,
seed: 9973,
access_point: 5,
arrival_time_exponential: 0.5,
deadline: 10,
initial_energy: 100000,

request_distribution: {
    distribution: "OrbitAware"
},

AP_selection: "optimal",
SearchNode: "ERT",

Routing_algorithm: {
    BATMAN: false,
    GREEDY: true,
    DSR: false
}
```

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

## Running the Existing Heuristic

Set:

```json5
SearchNode: "ERT"
```

Then run:

```bash
python main.py
```

This uses the existing heuristic scheduling path in `simulation.py`.

---

## Running the ILP Policy

Set:

```json5
SearchNode: "ILP"
```

Then run:

```bash
python main.py
```

COSMOS will route SearchNode decisions through `ILP_simulation.py` and the ILP solver.

---

## Running the DQN Policy

(Not yet finished) . Once DQN integration is enabled in `simulation.py` and recognised by `main.py`, set:

```json5
SearchNode: "DQN"
```

The intended flow is:

```text
real COSMOS candidate data >  DQN_data_Translator.py > DQN_agent.py > selected action > candidate SEN > normal COSMOS task execution
```

---

## Output

Simulation results are written to the:

```text
result/
```

directory.

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


