"""
Satellite Network Simulation Entry Point
========================================

This script serves as the main driver for the LEO Satellite Edge Computing simulation.
It integrates discrete-event simulation (SimPy) with orbital mechanics (Skyfield) and
optimization logic (ILP/Heuristics) to evaluate task offloading performance.

Workflow:
---------
1.  **Initialization**: Sets up the SimPy environment and loads global configurations.
2.  **Topology Setup**: Loads satellite TLE data, calculates orbits, and establishes
    inter-satellite links (ISLs) or ground connections based on the specified mode
    (Access Point selection, Routing Algorithm).
3.  **Task Scheduling**:
    * Initializes background loads (Batch tasks) on satellites.
    * Starts the dynamic task generator (`generate_tasks`).
4.  **Simulation Execution**: Runs the simulation for the configured duration, handling
    task processing, energy consumption, and network routing.
5.  **Data Collection & Logging**:
    * Periodically snapshots network state via `data_collector`.
    * Exports detailed CSV logs for Task Lifecycle, Migration, and Energy Statistics.
    * Saves final datasets in JSON5 format.

Output:
-------
Results are saved in a structured directory tree under `result/`, organized by:
Solver Type -> Weights -> Deadline -> Energy Budget -> Image Resolution Profile.
"""
import csv
import json5
import os
import sys
import simpy
import json
from EdgeServer import build_task_csv_path
from enums import AccPointMode
import ILP_simulation, simulation
from Task import findAlgorithm, generate_Tasks_Status, convert_task_list_in_dict
from simulation import generate_tasks
from topology import build_configurations, get_global_mode, load_saved_configuration, loadConfiguration, \
    periodic_recall_Topology_monitor, string_to_skyfield_time, loadConfiguration_simple
from user_based_topology import get_current_time, getObserverObj
from SaveCurrentSATOnFile import saveTLEOnFile
from routing_Manager import load_saved_OGM, periodic_recall_Routing_monitor
from Observer import Observer
from simulation_OGM import process_OGM_enviroment_simulation, remove_first_30_configurations
import globals
import time

from utils import colorize

simulation_dataset = []


def _win_longpath(p: str) -> str:
    """This makes the path compatible with long Windows paths by using the \\?\\ prefix."""
    if os.name != "nt":
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):  # UNC path
        return "\\\\?\\UNC" + p[1:]
    return "\\\\?\\" + p


# Add the new data collection process
def data_collector(env, interval, start_time, end_time):
    """
   Process for periodically collecting network status data
    within a specified time interval.
    """
    global simulation_dataset
    simulation_dataset = {
        "metadata": {
            "battery_life": globals.config.get("initial_energy", 0.0)
        },
        "snapshots": []
    }
    # Wait until the start of the collection period
    yield env.timeout(start_time)

    while env.now <= end_time:
        # Create a “snapshot” of the current state
        snapshot = {
            "time": env.now,
            "satellites": []
        }

        # Scan all the satellites and collect the data
        for sat_obj in globals.edge_servers:
            if sat_obj:
                state = sat_obj.export_state(env)
                if state.get("neighbors_count") > 0:
                    snapshot["satellites"].append(state)

        # Add the snapshot to the global list
        simulation_dataset["snapshots"].append(snapshot)

        # Wait for the specified amount of time before the next collection
        yield env.timeout(interval)


if __name__ == "__main__":
    # Set seed and environment
    start_time_simulation_real = time.time()
    env = simpy.Environment()
    config, resolution_config = globals.config, globals.resolution_config
    print(
        f"[INFO] Using config file: {globals.config_file_path if hasattr(globals, 'config_file_path') else 'unknown'}")

    MaxTry = config.get("max_try", 10)
    r_algo = None
    # 1) Configuration Construction
    if config.get("Build_Configurations", False):

        configurations_base, configuration_optimal = None, None

        if not config.get("load_saved_configuration"):
            tle_data = saveTLEOnFile()
            t0 = get_current_time()
            configurations_base = build_configurations(t0, tle_data, AccPointMode.BASE)
            configuration_optimal = build_configurations(t0, tle_data, AccPointMode.OPTIMAL)
        else:
            # TEST PHASE: If we have already created the configurations, we load them
            configurations_base = load_saved_configuration(AccPointMode.BASE)
            configuration_optimal = load_saved_configuration(AccPointMode.OPTIMAL)

        if config["redistribuite_OGM"]:
            process_OGM_enviroment_simulation(configurations_base)
            remove_first_30_configurations(AccPointMode.BASE)
            remove_first_30_configurations(AccPointMode.OPTIMAL)

        sys.exit("File of configurations created")

    # 2) Loading or Creating a Topology
    if config.get("Load_Configuration", False):
        mode = get_global_mode()  # sets the simulation mode
        r_algo = findAlgorithm()
        simple_exec = True

        globals.data_configurations = load_saved_configuration(mode)

        if r_algo == "BATMAN" or r_algo == "DINAMICO":
            # If the algorithm requires TABLE GMOs
            globals.OGMs_tables = load_saved_OGM()
            globals.edge_servers, globals.global_access_point = loadConfiguration(env, globals.data_configurations,
                                                                                  globals.OGMs_tables)
        else:
            globals.edge_servers, globals.global_access_point = loadConfiguration_simple(env,
                                                                                         globals.data_configurations)

        env.process(periodic_recall_Topology_monitor(env, globals.data_configurations, globals.OGMs_tables))
    else:
        sys.exit("No configuration required!")

    globals.observer = Observer(env, getObserverObj())  # Singleton Observer
    env.process(periodic_recall_Routing_monitor(env))

    if not globals.data_configurations:
        sys.exit("Error: The configuration file is empty.")
    skyfield_time = string_to_skyfield_time(globals.data_configurations["t0"])

    globals.initial_server_counter = {
        server.name: 0 for server in globals.edge_servers}
    globals.different_server_counter = {
        server.name: 0 for server in globals.edge_servers}
    globals.other_server_counter = {
        server.name: 0 for server in globals.edge_servers}

    # 3 batches online
    batch_task_id = 1000  # base id per batch
    for server in globals.edge_servers:
        n_batches = globals.rnd.randint(0, 3)  # one or more batches per server (range 0–3)
        for _ in range(n_batches):
            batch_task_id += 1000

            # system requirements (RAM/disk) if you want to keep them
            required_ram = globals.rnd.randint(config["required_ram"]["min"], config["required_ram"]["max"])
            required_disk = globals.rnd.randint(config["required_disk"]["min"], config["required_disk"]["max"])

            # Choose `image_size` based on the BATCH rules in your `resolution_config`
            ranges = resolution_config["size_ranges_MB"]
            batch_params = ranges["BATCH_TASK"]
            r = globals.rnd.random()
            if r < batch_params["gamma_H_weight"]:
                image_size = globals.rnd.uniform(*batch_params["H_range"])
            else:
                image_size = globals.rnd.uniform(*batch_params["VH_range"])

            # d_cpu = 0 per batch
            d_cpu_batch = 0.0
            bw_Bps, data_bytes = simulation.network_metrics(image_size)
            d_net_batch = data_bytes / bw_Bps if bw_Bps > 0 else float('inf')
            delta_D = config.get("delta_D", 0.2)
            deadline_batch = (1.0 + delta_D) * (d_cpu_batch + d_net_batch)

            # --- SELECTING BATCH MODE (ILP vs. ERT) ---

            if str(config.get("SearchNode", "")).upper() == "ILP":
                # Note: We pass `batch_task_id` as an INT (removed `str()`) to avoid a TypeError in the final sort
                env.process(ILP_simulation.TaskAssignment_ILP(
                    env,
                    server,
                    batch_task_id,  # <--- FIX: str() REMOVED; it is now INT
                    image_size,
                    env.now,
                    0,
                    0.0,
                    "Batch",
                    0.0,
                    deadline_batch,
                    net_bw_override_Bps=None,
                    result_sink={},
                    allow_retry=False
                ))
            elif str(config.get("SearchNode", "")).upper() == "ERT":
                # ERT Mode: Use simulation.TaskAssignment
                env.process(simulation.TaskAssignment(
                    env,
                    server,
                    batch_task_id,  # INT
                    image_size,
                    env.now,
                    0,
                    0.0,
                    "Batch",
                    0.0,
                    deadline_batch
                ))
            else:
                # ERT/Standard Mode: Let's use the function in simulation.py
                env.process(simulation.enqueue_batch_in_net(
                    env,
                    server,
                    batch_task_id,  # <--- already INT
                    image_size,
                    env.now,
                    deadline_batch  # We'll miss the absolute or relative deadline, depending on how the department handles it
                ))
            # -----------------------------------------------------------------------------------------------

    print("Batch tasks for all servers scheduled (0..3 per server).")

    # 4) Start generating tasks
    env.process(generate_tasks(
        env,
        globals.initial_server_counter,
        globals.different_server_counter,
        globals.other_server_counter
    ))

    # 5) Naming Conventions for Folders and Files
    ap = config.get("access_point", 0)
    seed_val = config["seed"]
    gen_dist = config["generate_tasks"]["distribution"]
    req_dist = config["request_distribution"]["distribution"]
    if req_dist == "0_0_0":
        req_dist = "RR"
    atime = config["arrival_time_exponential"]
    cpu_mean = config["CPU_timeout"]["gen"]["mean"]
    solver = config["SearchNode"]
    ap_dir_bidir = config["AP_routing_bidirectional"]  # (Booleano) AP_Routing
    ap_selection = config["AP_selection"]
    energy_budget = config["initial_energy"]
    deadline = config["deadline"]
    ilp_w_e, ilp_w_R = config["ilp_weights"]["w_e"], config["ilp_weights"]["w_R"]
    complete_sim_solver = None

    if req_dist == "DTS-base" and ap_selection == "base" and solver == "ERT":
        complete_sim_solver = "DTS-base"
    elif req_dist == "DTS-base" and ap_selection == "optimal" and solver == "ERT":
        complete_sim_solver = "DTS-APopt"
    elif req_dist == "OrbitAware" and ap_selection == "optimal" and solver == "ERT":
        complete_sim_solver = "OrbitAware"
    elif req_dist == "DTS-base" and ap_selection == "optimal" and solver == "ILP":
        complete_sim_solver = "ILP"
    else:
        sys.exit(
            f"Complete_sim_solver not right! CHECK: req_dist:{req_dist} ap_selection:{ap_selection} solver:{solver}")

    # BETA, ALPHA, GAMMA
    beta = resolution_config["beta_probabilities"]
    bg, bcpui, bcpudi = beta["Generic_Service"], beta["CPU_Intensive"], beta["CPU_and_Data_Intensive"]
    alpha = resolution_config["size_ranges_MB"]["CPU_DATA_INTENSIVE"]
    am, ah, avh = alpha["alpha_M_weight"], alpha["alpha_H_weight"], alpha["alpha_VH_weight"]
    gamma = resolution_config["size_ranges_MB"]["BATCH_TASK"]
    gh, gvh = gamma["gamma_H_weight"], gamma["gamma_VH_weight"]
    img_res_dir = f"IMG_RES_bg_{bg}_bcpui_{bcpui}_bcpudi_{bcpudi}_am_{am}ah_{ah}_avh_{avh}_gh_{gh}_gvh_{gvh}"

    # Base folder: includes modes, APs, and seeds
    base_dir = ""
    if complete_sim_solver == "ILP":
        base_dir = f"result/{complete_sim_solver}_sim_SystemAP{ap}/w_e{ilp_w_e}_wR_{ilp_w_R}/deadline_{deadline}/Energy_budget_{energy_budget}/{img_res_dir}/Routing_bidirectional_{ap_dir_bidir}/seed_{seed_val}/r_algo_{r_algo}/we_{ilp_w_e}_wR_{ilp_w_R}/"
    else:
        base_dir = f"result/{complete_sim_solver}_sim_SystemAP{ap}/deadline_{deadline}/Energy_budget_{energy_budget}/{img_res_dir}/Routing_bidirectional_{ap_dir_bidir}/seed_{seed_val}/r_algo_{r_algo}/"
    os.makedirs(base_dir, exist_ok=True)

    # CSV files and logs with complete nomenclature
    csv_task = (
        f"{base_dir}/results_"
        f"{gen_dist}_REQ-{complete_sim_solver}_"
        f"AT_{atime}_CPU_{cpu_mean}.csv"
    )
    csv_mig = (
        f"{base_dir}/migration_"
        f"{gen_dist}_REQ-{complete_sim_solver}_"
        f"AT_{atime}_CPU_{cpu_mean}.csv"
    )
    csv_routing_task = build_task_csv_path(base_dir, atime, cpu_mean)


    def _ensure_parent_dir(path_str: str) -> str:
        parent = os.path.dirname(path_str) or "."
        parent = os.path.abspath(parent)
        os.makedirs(_win_longpath(parent), exist_ok=True)
        return os.path.join(parent, os.path.basename(path_str))


    csv_task = _ensure_parent_dir(csv_task)
    csv_mig = _ensure_parent_dir(csv_mig)
    csv_routing_task = _ensure_parent_dir(csv_routing_task)

    # Save the CSV name in the config for any external modules
    config["csv_name"] = {"name": csv_task}
    # writing the *used* configuration to the simulation's output folder
    used_config_path = os.path.join(base_dir, "used_config.json5")
    os.makedirs(os.path.dirname(used_config_path), exist_ok=True)
    with open(used_config_path, 'w') as wf:
        json5.dump(config, wf, indent=2)
    print(f"[INFO] Configuration saved in: {used_config_path}")

    print(f"--- Start Simulation ---")
    print(f"Folder: {base_dir}")
    print(f"CSV tasks: {csv_task}")
    print(f"CSV migrations: {csv_mig}")

    # Example: every 2 seconds, from second 100 to 200
    env.process(data_collector(env, interval=2, start_time=100, end_time=200))

    # 6) Running the Simulation
    env.run(config['simulation_duration'])

    # Specify the name of the output file
    output_folder = "Generated_datasets"

    if config.get("save_generated_tasks_dataset", True) and globals.gbl_generated_tasks_data:
        file_name = "generated_tasks_counter.json5"
        # save it to the simulation's specific folder
        output_path = os.path.join(base_dir, file_name)

        print(f"\nSave the generated dataset to: {output_path}")

        try:
            with open(output_path, 'w') as f:
                json5.dump(globals.gbl_generated_tasks_data, f, indent=4)
            print("The backup was successfully completed.")
        except Exception as e:
            print(f"ERROR saving the dataset:{e}")

    # Save the dataset at the end of the simulation
    ENABLE_MONITORING = config.get("enable_queue_monitoring", False)
    if ENABLE_MONITORING:
        monitor_file_name = "simulation_dataset.json"
        output_path_monitor = os.path.join(base_dir, monitor_file_name)
        with open(output_path_monitor, "w") as f:
            json.dump(simulation_dataset, f, indent=4)
        print("\nSimulation state dataset saved in 'simulation_dataset.json'")

    # Print the task summary using the Observer function
    globals.observer.print_task_summary()
    print("-" * 10)
    generate_Tasks_Status(csv_routing_task, base_dir)
    task_dict = convert_task_list_in_dict(globals.gbl_tasks)

    # ---------------------------------------------------------------------
    # 7) Write results to a CSV file (Completed/Rejected/Pending Tasks)
    # ---------------------------------------------------------------------

    # Gather all the servers into a single list for iteration
    all_servers_by_name = {}
    for s in (globals.edge_servers or []) + (globals.global_access_point or []):
        if s is None:
            continue
        all_servers_by_name[s.name] = s
    all_servers = list(all_servers_by_name.values())

    with open(csv_task, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        # Header updated to 27 columns
        writer.writerow([
            "Task ID", "Task Type", "Status", "Arrival Time (System)",
            "Arrival Time (Queue)", "Start Time", "End Time", "Execution time",
            "Service Time", "Time in system", "Time in queue", "Server Name", "Num Hops",
            "Num Hops Routing",  # <-- New Column
            "Queue length", "transfer_time", "Image_Size_MB", "Exec_after_set",
            "Energy_CPU [J]", "Energy_NET [J]", "Energy_TOTAL [J]", "Remaining_energy [J]", "Remaining_energy [%]",
            "Rejection Reason",
            "Routing Init Time",  # <-- New Column
            "Routing End Time",  # <-- New Column
            "Routing Duration"  # <-- New Column
        ])

        initial_energy_for_percent = config.get("initial_energy", 0.0)

        for srv in all_servers:
            # completed tasks
            for entry in getattr(srv, 'completed_tasks', []):

                (tid, task_type, arr_sys, arr_q, start_t, end_t, ex_t, service_t,
                 time_q, sel_srv, hops, qlen, tranfer_t, image_size,
                 DeadLine, exec_set, eps_cpu, eps_net, eps_tot, srv_rem_energy) = entry

                time_in_system = (end_t - arr_sys) if (
                        isinstance(end_t, (int, float)) and isinstance(arr_sys, (int, float))) else "N/A"

                remaining_percent = "N/A"
                if isinstance(srv_rem_energy, (int, float)) and initial_energy_for_percent > 0:
                    remaining_percent = (srv_rem_energy / initial_energy_for_percent) * 100

                # Retrieve the routing data from the dictionary
                r_hops, r_init_time, r_end_time, r_duration = "N/A", "N/A", "N/A", "N/A"  # Default
                if tid in task_dict:
                    try:
                        r_hops, r_init_time, r_end_time, r_duration = task_dict[tid].get_stat_csv()
                    except Exception as e:
                        print(f"Warning: could not get routing stats for task {tid}: {e}")

                writer.writerow([
                    tid, task_type, "Completed", arr_sys, arr_q, start_t, end_t,
                    ex_t, service_t, time_in_system, time_q, sel_srv, hops,
                    r_hops,  # <-- Valore Routing Hops
                    qlen, tranfer_t, image_size, exec_set,
                    eps_cpu, eps_net, eps_tot, srv_rem_energy, remaining_percent, "N/A",
                    r_init_time,  # <-- Valore Routing Init Time
                    r_end_time,  # <-- Valore Routing End Time
                    r_duration  # <-- Valore Routing Duration
                ])

            # rejected tasks
            # Calculate the percentage of remaining energy for rejected tasks
            remaining_percent = "N/A"
            if isinstance(srv.energy, (int, float)) and initial_energy_for_percent > 0:
                remaining_percent = (srv.energy / initial_energy_for_percent) * 100

            for (tid, task_type, arr_sys, img_size, reason, d_cpu) in getattr(srv, 'rejected_tasks', []):
                #  --- START OF CORRECT BLOCK FOR REJECTED ---
                writer.writerow([
                    tid,  # Task ID
                    task_type,  # Task Type
                    "Rejected",  # Status
                    arr_sys,  # Arrival Time (System)
                    "N/A",  # Arrival Time (Queue)
                    "N/A",  # Start Time
                    "N/A",  # End Time
                    d_cpu,  # Execution time (Corretto)
                    0.0,  # Service Time (Corretto)
                    0.0,  # Time in system (Corretto)
                    0.0,  # Time in queue (Corretto)
                    srv.name,  # Server Name
                    0,  # Num Hops (Corretto)
                    "N/A",  # Num Hops Routing
                    "N/A",  # Queue length
                    0.0,  # transfer_time (Corretto)
                    img_size,  # Image_Size_MB
                    False,  # Exec_after_set (Corretto)
                    0.0,  # Energy_CPU [J] (Corretto)
                    0.0,  # Energy_NET [J] (Corretto)
                    0.0,  # Energy_TOTAL [J] (Corretto)
                    srv.energy,  # Remaining_energy [J]
                    remaining_percent,  # Remaining_energy [%]
                    reason,  # Rejection Reason
                    "N/A",  # Routing Init Time
                    "N/A",  # Routing End Time
                    "N/A"  # Routing Duration
                ])

            # residual tasks: CPU queue & NET queue
            residual_tasks = []
            if ENABLE_MONITORING:  # Check only if monitoring is active
                for req in getattr(srv, 'cpu_dev').queue:
                    if hasattr(req, 'task_data'):
                        td = req.task_data
                        residual_tasks.append(
                            {"tid": td.id, "type": "CPU_Waiting", "demand": getattr(td, 'd_cpu', 'N/A')})
                for req in getattr(srv, 'net_dev').queue:
                    if hasattr(req, 'task_data'):
                        td = req.task_data
                        residual_tasks.append(
                            {"tid": td.id, "type": "NET_Waiting", "demand": getattr(td, 'd_net', 'N/A')})

            total_residual_count = len(residual_tasks)
            # Calculate the remaining percentage for the remaining tasks as well (use the same percentage as for the rejected tasks)
            remaining_percent = "N/A"
            if isinstance(srv.energy, (int, float)) and initial_energy_for_percent > 0:
                remaining_percent = (srv.energy / initial_energy_for_percent) * 100

            for task_data in residual_tasks:
                tid = task_data["tid"]
                task_type_label = f"Residual ({task_data['type']})"
                demand = task_data["demand"]
                # If the question is known, use it; otherwise, use 0.0
                execution_time = demand if task_data["type"] == "CPU_Waiting" and isinstance(demand,
                                                                                             (int, float)) else 0.0
                service_time = demand if task_data["type"] == "NET_Waiting" and isinstance(demand,
                                                                                           (int, float)) else 0.0

                # --- START OF CORRECT BLOCK FOR RESIDUAL ---
                writer.writerow([
                    tid,  # Task ID
                    task_type_label,  # Task Type
                    "In Queue",  # Status
                    "N/A",  # Arrival Time (System)
                    "N/A",  # Arrival Time (Queue)
                    "N/A",  # Start Time
                    "N/A",  # End Time
                    execution_time,  # Execution time (dal demand)
                    service_time,  # Service Time (dal demand)
                    "N/A",  # Time in system
                    "N/A",  # Time in queue
                    srv.name,  # Server Name
                    "N/A",  # Num Hops
                    "N/A",  # Num Hops Routing
                    total_residual_count,  # Queue length
                    "N/A",  # transfer_time (Corretto)
                    "N/A",  # Image_Size_MB (Corretto)
                    False,  # Exec_after_set (Corretto)
                    0.0,  # Energy_CPU [J] (Corretto)
                    0.0,  # Energy_NET [J] (Corretto)
                    0.0,  # Energy_TOTAL [J] (Corretto)
                    srv.energy,  # Remaining_energy [J]
                    remaining_percent,  # Remaining_energy [%]
                    "In Queue at End",  # Rejection Reason
                    "N/A",  # Routing Init Time
                    "N/A",  # Routing End Time
                    "N/A"  # Routing Duration
                ])

        # Also dump global batch completions (if any)
        # NOTE: Now that we are using TaskAssignment_ILP, batches are also logged in srv.completed_tasks.
        # However, TaskAssignment_ILP also populates globals.gbl_tasks.
        # If TaskAssignment_ILP does not populate globals.gbl_batch_completed, this section will be empty,
        # but the data will be correct above (under srv.completed_tasks).
        for entry in getattr(globals, 'gbl_batch_completed', []):
            (tid, task_type, arr_sys, arr_q, start_t, end_t, ex_t, service_t,
             tq, sel_srv, hops, qlen, trf, image_size,
             exec_set, eps_cpu, eps_net, eps_tot, srv_rem_energy) = entry

            time_in_system = (end_t - arr_sys) if (
                    isinstance(end_t, (int, float)) and isinstance(arr_sys, (int, float))) else "N/A"

            remaining_percent = "N/A"
            if isinstance(srv_rem_energy, (int, float)) and initial_energy_for_percent > 0:
                remaining_percent = (srv_rem_energy / initial_energy_for_percent) * 100

            writer.writerow([
                tid, task_type, "Completed", arr_sys, arr_q, start_t, end_t,
                ex_t, service_t, time_in_system, tq, sel_srv, hops,
                "N/A",  # Num Hops Routing
                qlen, trf, image_size,
                exec_set, eps_cpu, eps_net, eps_tot, srv_rem_energy, remaining_percent, "N/A",
                "N/A",  # Routing Init Time
                "N/A",  # Routing End Time
                "N/A"  # Routing Duration
            ])

    print(f"Simulation results saved to: {csv_task}")

    # --------------------------------------------------
    # 8) Writing Energy Statistics and Task Counts
    # --------------------------------------------------

    csv_energy_stats = (
        f"{base_dir}/energy_stats_"
        f"{gen_dist}_REQ-{complete_sim_solver}_"
        f"AT_{atime}_CPU_{cpu_mean}.csv"
    )

    print(f"\nSalvataggio statistiche energetiche e conteggi task su: {csv_energy_stats}")

    try:
        # Get the initial energy from the config to calculate the percentage
        initial_energy = config.get("initial_energy", 0.0)

        # 1. Initialize a dictionary to collect statistics
        stats_per_server = {}
        for srv in all_servers:
            stats_per_server[srv.name] = {
                "server_obj": srv,
                "energy_consumptions": [],
                "task_counts": {
                    "Generic_Service": 0,
                    "CPU_Intensive": 0,
                    "CPU_and_Data_Intensive": 0,
                    "Batch": 0
                }
            }

        # 2. Populate the statistics from NON-BATCH tasks (from srv.completed_tasks)
        for srv in all_servers:
            for entry in getattr(srv, 'completed_tasks', []):
                task_type = entry[1]  # Indice 1 per task_type
                energy = entry[18]  # Indice 18 per eps_tot

                if task_type in stats_per_server[srv.name]["task_counts"]:
                    stats_per_server[srv.name]["task_counts"][task_type] += 1

                if isinstance(energy, (int, float)) and energy > 0.0:
                    stats_per_server[srv.name]["energy_consumptions"].append(energy)

        # 3. Populate the statistics from BATCH tasks (from globals.gbl_batch_completed)
        for entry in getattr(globals, 'gbl_batch_completed', []):
            server_name = entry[9]  # Indice 9 per sel_srv
            task_type = entry[1]  # Indice 1 per task_type
            energy = entry[18]  # Indice 18 per eps_tot

            if server_name in stats_per_server and task_type == "Batch":
                stats_per_server[server_name]["task_counts"]["Batch"] += 1
                if isinstance(energy, (int, float)) and energy > 0.0:
                    stats_per_server[server_name]["energy_consumptions"].append(energy)

        # 4. Write the CSV file
        with open(csv_energy_stats, mode='w', newline='') as f_stats:
            writer_stats = csv.writer(f_stats)

            # Write the header
            writer_stats.writerow([
                "Server_Name",
                "Total_Tasks_Completed",
                "Generic_Service_Count",
                "CPU_Intensive_Count",
                "CPU_Data_Intensive_Count",
                "Batch_Count",
                "Generic_Service_%",
                "CPU_Intensive_%",
                "CPU_Data_Intensive_%",
                "Batch_%",
                "Total_Energy_Consumed_J",
                "Total_Energy_Consumed_%",
                "Min_Energy_Consumed_J",
                "Max_Energy_Consumed_J",
                "Avg_Energy_Consumed_J",
                "Final_Remaining_Energy_J"
            ])

            # 5. Calculate the final statistics and write the lines
            for server_name, stats in stats_per_server.items():

                task_counts = stats["task_counts"]
                energy_consumptions = stats["energy_consumptions"]

                total_tasks_completed = sum(task_counts.values())

                # Calculating Percentages by Task Type
                if total_tasks_completed > 0:
                    percent_generic = (task_counts["Generic_Service"] / total_tasks_completed) * 100
                    percent_cpu = (task_counts["CPU_Intensive"] / total_tasks_completed) * 100
                    percent_cpu_data = (task_counts["CPU_and_Data_Intensive"] / total_tasks_completed) * 100
                    percent_batch = (task_counts["Batch"] / total_tasks_completed) * 100
                else:
                    percent_generic = 0.0
                    percent_cpu = 0.0
                    percent_cpu_data = 0.0
                    percent_batch = 0.0

                # Calculation of Energy Statistics
                if energy_consumptions:
                    total_e = sum(energy_consumptions)
                    min_e = min(energy_consumptions)
                    max_e = max(energy_consumptions)
                    avg_e = total_e / len(energy_consumptions)
                    total_e_percent = (total_e / initial_energy) * 100 if initial_energy > 0 else 0.0
                else:
                    total_e = 0.0
                    min_e = 0.0
                    max_e = 0.0
                    avg_e = 0.0
                    total_e_percent = 0.0

                # Write the line
                writer_stats.writerow([
                    server_name,
                    total_tasks_completed,
                    task_counts["Generic_Service"],
                    task_counts["CPU_Intensive"],
                    task_counts["CPU_and_Data_Intensive"],
                    task_counts["Batch"],
                    percent_generic,
                    percent_cpu,
                    percent_cpu_data,
                    percent_batch,
                    total_e,
                    total_e_percent,
                    min_e,
                    max_e,
                    avg_e,
                    stats["server_obj"].energy  # Energia finale rimasta
                ])

    except Exception as e:
        print(f"ERRORE during writing the energy stats file: {e}")

    # real-time execution time calculation
    end_time_simulation_real = time.time()
    duration_simulation = end_time_simulation_real - start_time_simulation_real
    print(f"Tempo di esecuzione REALE della simulazione {duration_simulation:.4f} secondi")
