"""
EdgeServer Module
=================

This module defines the `EdgeServer` class, which represents a Low Earth Orbit (LEO) satellite
acting as an edge computing node within the simulation environment.

The module integrates discrete event simulation (SimPy) with orbital mechanics (Skyfield)
to simulate a dynamic satellite network.

Key Functionalities:
--------------------
* **Dual-Stage Resource Model**: Manages separate queues for CPU (computation) and Network (transmission).
* **Routing Logic**: Implements decisions for various routing algorithms (BATMAN, GREEDY, DSR)
    and handles packet forwarding based on orbital topology.
* **Energy Management**: Simulates energy consumption for processing and transmission,
    enforcing energy budgets and handling battery depletion.
* **Metric Estimation**: Provides methods to estimate Waiting Times (W_cpu, W_net)
    and calculate Selection Scores for task offloading decisions.

Global Configuration:
---------------------
The module relies on a global configuration dictionary (`globals.config`) to set parameters
such as CPU capacity, initial energy, and routing flags.
"""
import globals
from math import sqrt
import simpy
from skyfield.api import EarthSatellite
from collections import OrderedDict
from user_based_topology import getSystemFromSat
from Task import Task, findAlgorithm
from utils import colorize, sendTask


# Read the JSON configuration file
config = globals.config

BATMAN = globals.config["Routing_algorithm"]["BATMAN"]
GREEDY = globals.config["Routing_algorithm"]["GREEDY"]
DSR = globals.config["Routing_algorithm"]["DSR"]

class EdgeServer:
    def __init__(self, env, name, satellite: EarthSatellite, orbitalSunset, is_acc_point, elev_angle):
        """
        Initialize an EdgeServer instance.

        :param env: Simulation environment.
        :param name: Name of the edge server.
        """

        self.env = env
        self.name = name
        self.satellite = satellite
        self.orbitalSunset = orbitalSunset
        self.is_acc_point = is_acc_point
        self.elev_angle = elev_angle
        self.neighbors = {}
        self.latency = {}
        self.bandwidth = {}
        self.completed_tasks = []
        self.cpu_busy_until = 0.0
        self.net_busy_until = 0.0

        # New CPU + NET codes for the 2-stage model
        cpu_capacity = config.get("cpu_capacity_queue", 5)
        net_capacity = config.get("net_capacity_queue", 5)
        self.cpu_dev = simpy.Resource(env, capacity=cpu_capacity)
        self.net_dev = simpy.Resource(env, capacity=net_capacity)
        self._cpu_capacity = cpu_capacity
        self._net_capacity = net_capacity

        self.energy_reserved = 0.0

        self.rejected_tasks = []  # List of Rejected Tasks
        self.energy = config.get("initial_energy", 10000.0)  # J (highest value)

        C_sen_cfg = config.get("C_sen", {"min": 1e7, "max": 1e10})
        self.C_sen_min = C_sen_cfg.get("min", 1e7)
        self.C_sen_max = C_sen_cfg.get("max", 1e10)

        self.C_sen = globals.rnd.uniform(self.C_sen_min, self.C_sen_max)

        self.tasks = []  # List of Tasks to Be Sent
        self.dead_tasks = []  # List of Dead Tasks (TTL = 0)

        self.ogm_sequence = 0
        self.OGMs = []
        self.OGMs_NP = []
        self.ogm_table = {}
        self.OGMs_History = OrderedDict()
        self.OGMs_History_dim = 2046
        self.ogm_sequence = 0           # Count of GMOs Issued
        self.OGMs = []                  # OGM to process
        self.OGMs_NP = []               # OGM received and Not-Processed
        self.ogm_table = {}             # OGMs Table {'originator': { 'neighbor': 'count'

        self.OGMs_History = OrderedDict()# List of GMOs Reviewed in the Past (FIFO)
        self.OGMs_History_dim = 2046     # History Size Limit for OGM

        self.routes = {}            # Dictionary of Completed Routes
        self.packets = []           # List of packages to be disposed of
        self.packets_seq = 0        # Shipped Packages Counter
        self.pkt_history = []

    def export_state(self, env):
        """
        Export the current state of the satellite as a dictionary, including tasks
        in the queue (.queue) and those currently in service (.users).
        """
        ENABLE_MONITORING = config.get("enable_queue_monitoring", False)

        # Helper function for collecting data from a list of SimPy requests
        def get_tasks_data(resource_list, demand_type):
            tasks_list = []
            for req_or_user in resource_list:
                # If it's a request, it could be encapsulated in a SimPy wrapper
                # In this setup, we assume that the `request` object has `task_data`
                if hasattr(req_or_user, 'task_data'):
                    task_obj = req_or_user.task_data
                    tasks_list.append({
                        "task_id": task_obj.id,
                        # Use the appropriate query type (d_cpu or d_net)
                        "demand": getattr(task_obj, demand_type, 'N/A'),
                        "deadline": getattr(task_obj, 'deadline', 'N/A'),
                        "image_size_MB": getattr(task_obj, 'image_size_MB', 'N/A'),
                    })
            return tasks_list

        tasks_in_cpu_queue = []
        tasks_in_net_queue = []
        tasks_in_cpu_service = []
        tasks_in_net_service = []

        if ENABLE_MONITORING:
            # Code (Waiting to get started)
            tasks_in_cpu_queue = get_tasks_data(self.cpu_dev.queue, 'd_cpu')
            tasks_in_net_queue = get_tasks_data(self.net_dev.queue, 'd_net')

            # Service (Running)
            # We need to consider the tasks that are currently using the resource
            tasks_in_cpu_service = get_tasks_data(self.cpu_dev.users, 'd_cpu')
            tasks_in_net_service = get_tasks_data(self.net_dev.users, 'd_net')

        # --- Neighborhood Details Collection (carried over from the previous version) ---
        neighbor_data = []
        for neighbor_obj in self.get_neighbors():
            neighbor_name = neighbor_obj.name
            neighbor_data.append({
                "name": neighbor_name,
                "latency_s": self.latency.get(neighbor_obj, 'N/A'),
                "bandwidth_MBps": self.bandwidth.get(neighbor_obj, 'N/A'),
                "elev_angle": neighbor_obj.elev_angle,
            })

        # --- Complete State Assembly ---
        state = {
            "time": env.now,
            "satellite": self.name,
            "in_listening_dome": self.elev_angle >= 40,
            "elev_angle": self.elev_angle,
            "energy_budget_J": self.energy,
            "energy_reserved_J": self.energy_reserved,
            "is_access_point": self.is_acc_point,
            "position": self.getPositionVector(globals.ist_in_conf),
            
            # Completed Tasks
            "completed_tasks_count": len(self.completed_tasks),
            # Code e Task
            "queue_cpu_len": len(tasks_in_cpu_queue),  # Conteggio solo i task in attesa
            "queue_net_len": len(tasks_in_net_queue),  # Conteggio solo i task in attesa

            # New fields for monitoring
            "cpu_service_len": len(tasks_in_cpu_service),
            "net_service_len": len(tasks_in_net_service),

            "queue_cpu_waiting": tasks_in_cpu_queue,
            "queue_net_waiting": tasks_in_net_queue,
            "queue_cpu_service": tasks_in_cpu_service,
            "queue_net_service": tasks_in_net_service,

            # Neighbors (detailed)
            "neighbors_count": len(neighbor_data),
            "neighbors_metrics": neighbor_data,
        }

        if ENABLE_MONITORING:
            return state
        return None
    def record_rejected_task(self, task_id, task_type, arrival_time_system, img_size, rejection_reason, d_cpu ):
        """
        Record a rejected task along with the reason for the rejection
        """
        self.rejected_tasks.append(
            (task_id, task_type, arrival_time_system, img_size, rejection_reason, d_cpu )
        )
        print(f"[Task {task_id}] REJECTED on {self.name} due to: {rejection_reason}")

    def task_completed(self, task_id, task_type, arrival_time_system, arrival_time_task_queue,
                       start_time, end_time, execution_time, service_time, time_in_queue,
                       selected_server, num_hops, lunghezza_coda, transfer_time, image_size,
                       DeadLine, exec_after_set,
                       eps_cpu=0.0, eps_net=0.0):
        """
        Record completed tasks, including energy metrics (CPU + NET) and task type
        """

        # Calculate the total energy at the beginning of the function to avoid the error
        total_energy = eps_cpu + eps_net

        if self.elev_angle < config["Phi_max"]:
            exec_after_set = True

        self.completed_tasks.append(
            (task_id,  task_type, arrival_time_system, arrival_time_task_queue,
             start_time, end_time, execution_time, service_time, time_in_queue,
             selected_server, num_hops, lunghezza_coda,
             transfer_time, image_size,
             DeadLine, exec_after_set, eps_cpu, eps_net, total_energy, self.energy)
        )

        print(f"[Task {task_id}] Task eseguito su {self.name} {self.elev_angle} degrees | "
              f"CPU={eps_cpu:.4f}J, NET={eps_net:.4f}J, TOTAL={total_energy:.4f}J, "
              f"Remaining={self.energy:.2f}J")

    def add_neighbor(self, neighbor_server, hop_count, latency, bandwidth):
        """
                Add a neighbor server with its hop count and latency.

                :param neighbor_server: Neighbor server to add.
                :param hop_count: Number of hops to reach the neighbor server.
                :param latency: Latency to the neighbor server.

                :return: None
                """
        self.neighbors[neighbor_server] = hop_count
        self.latency[neighbor_server] = latency
        self.bandwidth[neighbor_server] = bandwidth

    def update_neighbors(self, new_neighbors, new_latency, new_bandwidth):
        """
        Update the neighbors, latency, and bandwidth of the edge server.

        :param new_neighbors: Dictionary of new neighbors and their hop counts.
        :param new_latency: Dictionary of new latencies to the neighbors.
        :param new_bandwidth: Dictionary of new bandwidths to the neighbors.

        :return: None
        """
        self.neighbors = new_neighbors
        self.latency = new_latency
        self.bandwidth = new_bandwidth

    def get_neighbors(self):
        '''
        Get neighbor servers at a distance of one hop.

        :return: List of neighbor servers at a distance of one hop.
        '''
        neighbors_at_distance_one = []

        for neighbor, hop_count in self.neighbors.items():
            if hop_count <= 1:
                neighbors_at_distance_one.append(neighbor)

        return neighbors_at_distance_one

    def get_satellite(self):
        return self.satellite

    def get_latency(self, neighbor_server):
        '''
        Get latency to a specific neighbor server.

        :param neighbor_server: Neighbor server to get the latency for.

        :return: Latency to the neighbor server.
        '''
        return self.latency.get(neighbor_server, None)

    def get_bandwidth(self, neighbor_server):
        '''
        Get bandwidth to a specific neighbor server.

        :param neighbor_server: Neighbor server to get the bandwidth for.

        :return: Bandwidth to the neighbor server.
        '''
        return self.bandwidth.get(neighbor_server, None)

    def __str__(self):
        return f"Satellite :{self.name}\n\telev:{self.elev_angle}\n\tis_AP:{self.is_acc_point}"

    def getPositionVector(self, t):
        """
        This function returns a 3-dimensional vector,
        which represents the satellite's position at a given moment.
        """

        return getSystemFromSat(self.satellite, t, True).position.km.tolist()
    # --------------------------------------------------
    # Energy Management
    # --------------------------------------------------
    def compute_routing_energy(self, file_size, bandwidth, Ptrasm=1.0):
        """
        Routing Power (transmission) [Joule].
        file_size in Byte, bandwidth in Byte/s
        """
        if bandwidth and bandwidth > 0:
            return Ptrasm * (file_size / bandwidth)
        return 0.0

    def compute_execution_energy(self, execution_time, C_sen, e=5e-26):
        """
        Computing Power [Joule].
        execution_time ~ Service Request (s)
         C_sen ~ CPU capacity in cycles/s
        """
        d = execution_time
        return d * e * (C_sen ** 3)

    # --------------------------------------------------
    # Waiting time (W^cpu, W^net)
    # --------------------------------------------------

    def _get_min_remaining(self, resource, kind='cpu'):
        """
         Returns the minimum remaining time among the active tasks (time until the first release).
        If there are free slots, returns 0.0.
        """
        now = self.env.now
        if len(resource.users) < resource.capacity:
            return 0.0

        min_remaining = float('inf')
        for user in resource.users:
            if hasattr(user, 'task_data'):
                td = user.task_data
                if kind == 'cpu':
                    d = getattr(td, 'd_cpu', None)
                    start = getattr(td, 'start_time_cpu', None)
                    legacy_busy_until = getattr(self, 'cpu_busy_until', None)
                else:
                    d = getattr(td, 'd_net', None)
                    start = getattr(td, 'start_time_net', None)
                    legacy_busy_until = getattr(self, 'net_busy_until', None)

                if d is None:
                    continue

                if start is not None:
                    remaining = max(0.0, d - (now - start))
                else:
                    # Fall back to `busy_until` if available; otherwise, assume `d` as the upper bound
                    if legacy_busy_until is not None:
                        remaining = max(0.0, legacy_busy_until - now)
                    else:
                        remaining = d

                min_remaining = min(min_remaining, remaining)

        return min_remaining if min_remaining != float('inf') else 0.0

    def _get_running_remaining_total(self, resource, kind='cpu'):
        """
     Sum of the remaining time for tasks currently running (all servers).
        This is used to calculate the total work in progress.
        """
        now = self.env.now
        running_sum = 0.0
        for user in resource.users:
            if hasattr(user, 'task_data'):
                td = user.task_data
                if kind == 'cpu':
                    d = getattr(td, 'd_cpu', None)
                    start = getattr(td, 'start_time_cpu', None)
                    legacy_busy_until = getattr(self, 'cpu_busy_until', None)
                else:
                    d = getattr(td, 'd_net', None)
                    start = getattr(td, 'start_time_net', None)
                    legacy_busy_until = getattr(self, 'net_busy_until', None)

                if d is None:
                    continue

                if start is not None:
                    remaining = max(0.0, d - (now - start))
                else:
                    if legacy_busy_until is not None:
                        remaining = max(0.0, legacy_busy_until - now)
                    else:
                        remaining = 0.5 * d  # conservative fallback

                running_sum += remaining

        return running_sum

    def W_cpu(self):
        """
        Estimate the waiting time (remaining time in the queue) for a new task
        in the CPU queue (W_cpu). Use the principle of total
        remaining work divided by the service capacity (average service time).
        """

        # 1. Calculate the sum of the d_cpu values for all tasks currently in the queue
        queue_sum = self._get_queue_demand_sum(self.cpu_dev.queue, kind='cpu')

        # 2. Gets the minimum remaining execution time of the task currently being served.
        #    This is the time remaining until the completion of the task currently being served that has the lowest demand.
        #    If the CPU is idle, it will be 0.0.
        min_rem = self._get_min_remaining(self.cpu_dev, kind='cpu')

        # 3. Calculates the sum of the remaining time for all tasks currently running.
        #    This includes the time of the task that defines `min_rem` plus the time of the other running tasks.
        running_total = self._get_running_remaining_total(self.cpu_dev, kind='cpu')

        # 4. Determine the server's capacity (number of CPUs)
        capacity = max(1, getattr(self, '_cpu_capacity', 1))

        # 5. Calculate the total work that needs to be completed before the new task is ready.
        #    (Remaining work on running tasks + Work on pending tasks)
        work_tot = running_total + queue_sum

        # 6. Remaining work after the first interval (min_rem) is complete.
        #    During the min_rem interval, ‘capacity’ units of work (the servers)
        #    are processed, consuming (capacity * min_rem) total work.
        #    work_after_first is the remaining work that must be handled after
        #    at least one server becomes available.
        work_after_first = max(0.0, work_tot - capacity * min_rem)

        # 7. Final estimate of wait time (W_cpu):
        #    completion time of the nearest task (min_rem)
        #    + time to process the remaining work (work_after_first / capacity,
        #      since ‘capacity’ servers will now handle the rest of the work).
        return min_rem + (work_after_first / capacity)

    def W_net(self):
        """
          Estimates the waiting time (remaining time in the queue) for a new task
        in the Network queue (W_net). The logic is identical to that of W_cpu, but applied
        to the network resource (net_dev).
        """

        # 1. Calculate the sum of the d_net values for all tasks currently in the queue
        queue_sum = self._get_queue_demand_sum(self.net_dev.queue, kind='net')

        # 2. Returns the minimum remaining execution time for the task currently being served
        min_rem = self._get_min_remaining(self.net_dev, kind='net')

        # 3. Returns the sum of the remaining time for all tasks currently running
        running_total = self._get_running_remaining_total(self.net_dev, kind='net')

        # 4. Determines the network server's capacity (usually 1)
        capacity = max(1, getattr(self, '_net_capacity', 1))

        # 5. Calculate the total work that needs to be completed before the new task is ready.
        work_tot = running_total + queue_sum

        # 6. Remaining work after the first interval is complete (min_rem).
        #    work_tot - (capacity * min_rem) is the amount of remaining work
        #    that the ‘capacity’ servers still need to process.
        work_after_first = max(0.0, work_tot - capacity * min_rem)

        # 7. Final estimated waiting time (W_net):
        #    (completion time of the nearest task) + (time to process the remaining work)
        return min_rem + (work_after_first / capacity)

    def _get_queue_demand_sum(self, simpy_resource_queue, kind='cpu'):
        total_demand = 0.0
        for req in simpy_resource_queue:
            if hasattr(req, 'task_data'):
                task_obj = req.task_data
                if kind == 'cpu':
                    total_demand += getattr(task_obj, 'd_cpu', 0.0)
                else:
                    total_demand += getattr(task_obj, 'd_net', 0.0)
        return total_demand

    # --------------------------------------------------
    # Pipeline CPU -> NET
    # --------------------------------------------------

    def greedy_approach(self, env, task):
        dest_pos = globals.observer.getPositionVector(globals.ist_in_conf)
        ranker_neighbors = []

        for server in self.neighbors:
            neighbor_distance = get_pos_proximity(dest_pos, server.getPositionVector(globals.ist_in_conf))
            t = (server, neighbor_distance, server.is_acc_point)
            ranker_neighbors.append(t)

        ranker_neighbors.sort(key=lambda x: (not x[2], x[1]))

        best_server = None
        for neighbor_tuple in ranker_neighbors:
            if neighbor_tuple[0].name not in task.visited:
                best_server = neighbor_tuple[0]
                break

        if best_server:
            yield from sendTask(env, task, self, best_server, 'GREEDY')

    def deliver_to_Observer(self, env, mode, task):
        print(f"[MODE: {mode}]")
        task.routingEndTime = env.now
        algo = findAlgorithm()
        yield from sendTask(env, task, self, globals.observer, algo)

    def forward_packet(self, env):
        if len(self.neighbors) > 0:
            for task in self.tasks:

                if not task.arrived:

                    if config["AP_routing_bidirectional"]:
                        # Bidirectional: we send the task to the access points
                        if self.is_acc_point:
                            yield from self.deliver_to_Observer(env, 'BIDIRECTIONAL', task)
                            continue
                    else:
                        # Check that the satellite is in the dome
                        if self.elev_angle >= 40:
                            yield from self.deliver_to_Observer(env, 'MONODIRECTIONAL', task)
                            continue

                    # ! Algorithm
                    max_neighbor = None
                    if BATMAN:
                        if task.dest_node in self.ogm_table:
                            max_neighbor, max_value = find_OGM_intersection(
                                self.ogm_table[task.dest_node], self.neighbors, task
                            )
                        else:
                            print(f"[{task.id}] {task.dest_node} Route Temporarily Unknown")
                    if BATMAN and GREEDY:
                        if max_neighbor:
                            yield from sendTask(env, task, self, max_neighbor, 'BATMAN')
                        else:
                            yield from self.greedy_approach(env, task)

                    elif BATMAN:
                        if max_neighbor:
                            yield from sendTask(env, task, self, max_neighbor, 'BATMAN')

                    elif GREEDY:
                        yield from self.greedy_approach(env, task)

        # else:
        # print(f“{self.name} NO LONGER HAS NEIGHBORS TO WHICH TO TRANSMIT elev: {self.elev_angle}°”)
        # print(“Task IDs:”, [task.id for task in self.tasks])

    def get_selection_score(self, task_type, d_cpu, d_net, D_r,
                            energy_budget_max=1.0,
                            file_size_bytes=None, bandwidth_Bps=None,
                            w_e=None, w_R=None):
        """
        Calculate the score.
        - d_cpu: CPU time required (s)
        - d_net: estimated network time (s) (used in R_pred). If you want eps_net to be correct, pass file_size_bytes and bandwidth_Bps.
        - D_r: task deadline (s)
        - file_size_bytes, bandwidth_Bps: optional, required for compute_routing_energy
        - w_e, w_R: energy/delay weights (if None, they take values from the configuration or the default 0.5/0.5)
        """

        # weights (fallback)
        if w_e is None or w_R is None:
            w_e = config.get("weight_energy", 0.5)
            w_R = config.get("weight_response", 0.5)
        # 1) time prediction
        Wc = self.W_cpu()
        Wn = self.W_net()
        R_predicted = Wc + d_cpu + Wn + d_net

        # 2) CPU power estimate (requires C_sen)
        C_sen = getattr(self, 'C_sen', globals.rnd.uniform(self.C_sen_min, self.C_sen_max))


        eps_cpu = self.compute_execution_energy(d_cpu, C_sen, e=config.get("energy_coefficient", 5e-26))

        # 3) NET energy estimate: requires file_size_bytes and bandwidth_Bps
        if file_size_bytes is None or bandwidth_Bps is None:
            # If they are not provided, try to derive them (or set eps_net=0 as a fallback)
            eps_net = 0.0
        else:
            eps_net = self.compute_routing_energy(file_size_bytes, bandwidth_Bps, config.get("Ptrasm", 1.0))

        # 4) Check constraints (deadlines and available energy)
        if R_predicted > D_r or (eps_cpu + eps_net) > self.energy:
            return -float('inf')  # unacceptable

        # 5)normalization
        #         # Let's normalize B_i with respect to energy_budget_max
        if energy_budget_max > 0:
            B_normalized = self.energy / energy_budget_max
        else:
            # If the maximum available budget in the network is 0,
            # it means that everyone is out of funds. B_normalized is 0.
            B_normalized = 0.0

        # Let's normalize W with respect to the deadline D_r (to avoid division by zero)
        denom = D_r if D_r > 0 else 1.0
        if task_type in ("Generic_Service", "CPU_Intensive"):
            W_norm = Wc / denom
        elif task_type == "Batch":
            W_norm = Wn / denom
        elif task_type == "CPU_and_Data_Intensive":
            W_norm = (Wc + Wn) / denom
        else:
            W_norm = 0.0

        # 6) Weighted score: We want to maximize benefit (B_normalized) and minimize delay (W_norm)
        score = w_e * B_normalized - w_R * W_norm
        return score

def find_OGM_intersection(ogm_table, neighbors, task):
    """
    Finds the intersection between the actual neighbors and those in the OGM table,
    excluding satellites already visited by the task, and returns the neighbor with the highest OGM value.

    Args:
        ogm_table (dict): Dictionary that maps neighbor names to OGM values.
        neighbors (dict): Dictionary of neighbors.
        task (Task): Task object containing the set of satellites already visited.

    Returns:
        tuple: (max_neighbor, max_value)
            - max_neighbor: The neighbor with the highest OGM value (neighbor object).
            - max_value: The OGM value associated with max_neighbor.
            If there are no valid neighbors, both are None.
    """
    # Builds a dictionary of neighbors that are both in the OGM table and among the actual neighbors,
    # excluding those already visited by the task (to avoid loops).
    intersection = {
        neighbor: ogm_table[neighbor.name]
        for neighbor in neighbors
        if (
            neighbor.name in ogm_table
            and neighbor.name not in task.visited  # anti-loop filter
        )
    }

    # Find the neighbor with the highest OGM value in the intersection.
    max_neighbor, max_value = None, None
    if intersection:  # Prevents a ValueError if `intersection` is empty
        max_neighbor, max_value = max(
            intersection.items(), key=lambda item: item[1])

    return max_neighbor, max_value

def get_pos_proximity(pos1, pos2):
    """
    Calculate the distance between two points in three-dimensional space
    Args:
        pos1: Position vector of obj1
        pos2: Position vector of obj2
    :return: Length of the line segment from obj1 to obj2
    """
    # Extraction of coordinate components
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2

    # Calculate the Euclidean distance
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def build_task_csv_path(folder, at, cpu):
    csv_routing_task = ""
    if BATMAN and GREEDY:
        csv_routing_task = f"{folder}/[DINAMICO]_AT_{at}_CPU_{cpu}.csv"
    elif BATMAN:
        csv_routing_task = f"{folder}/[BATMAN]_AT_{at}_CPU_{cpu}.csv"
    elif GREEDY:
        csv_routing_task = f"{folder}/[GREEDY]_AT_{at}_CPU_{cpu}.csv"
    elif DSR:
        csv_routing_task = f"{folder}/[DSR]_AT_{at}_CPU_{cpu}.csv"
    return csv_routing_task
