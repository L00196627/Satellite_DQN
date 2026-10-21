"""
ILP-Based Task Scheduling Module
================================

This module implements the logic for assigning and executing tasks within a satellite
network using an optimization-based approach (ILP).

It handles the lifecycle of a task from the decision phase (searching for the best node)
to the execution phase (CPU processing and Network transmission).

Key Components:
---------------
* **TaskAssignment_ILP**: A SimPy process simulating the physical execution of a task
  (queuing, CPU processing, energy consumption, and downlink/network transfer).
* **SearchNode_ILP_Hybrid_v2**: The decision-making entry point. It constructs a local
  network snapshot ($E_k$) and invokes an ILP solver to find the optimal node.
  It includes fallback mechanisms for robustness.
* **Snapshot Construction**: Helpers (`_build_snapshot_Ek`, `_merge_queues_for_ilp`)
  to format simulation state into data structures compatible with the solver.

Dependencies:
-------------
* `simpy`: For discrete-event simulation.
* `sec_ilp_snapshot_v3`: Custom module containing the solver logic and data structures.
* `globals`: Access to global configuration and simulation state.
"""
import globals
from Task import Task
from sec_ilp_snapshot_v3 import solve_on_Ek, Snapshot, SENState, QueueTask, alpha_from_physics, solve_on_Ek_hierarchical

config = globals.config
resolution_config = globals.resolution_config

c_sen_config_global = config.get("C_sen", 1e9)
if isinstance(c_sen_config_global, dict):
    c_min = c_sen_config_global.get("min", 1e9)
    c_max = c_sen_config_global.get("max", 1e9)
    C_sen = globals.rnd.uniform(c_min, c_max)
else:
    C_sen = float(c_sen_config_global)

bw_MBps = float(config.get("available_bandwidth", {}).get("min", 2150.0))
bw_Bps = bw_MBps * (1024 ** 2) if bw_MBps is not None else 0.0
e_coeff = config.get("energy_coefficient", 5e-26)  # energy coefficient (numerical example)


def network_metrics(image_size_MB, Volume_size_MB=0.0):
    """
    Calculate the available bandwidth (in Bps) and the total data size (in bytes).

    Args:
        image_size_MB (float): Image size in MB.
        Volume_size_MB (float, optional): Additional volume size in MB. Default is 0.0.

    Returns:
        tuple: (bw_Bps, data_bytes)
    """

    # Calculate the total data size in MB and then in bytes
    total_data_MB = image_size_MB + Volume_size_MB
    data_bytes = total_data_MB * (1024 ** 2)

    return bw_Bps, data_bytes


def TaskAssignment_ILP(env, selected_server, task_id, image_size,
                   arrival_time_system, num_hops, transfer_time,
                   task_type, d_cpu, D_r, net_bw_override_Bps=None,
                   result_sink=None, allow_retry=False, routing_already_charged=False, routing_energy_from=None):
    """
    SimPy process that assigns a task to the selected server and simulates:
      - waiting in the CPU queue (cpu_dev)
      - CPU execution (yield timeout)
      - possible transfer to the NET queue (net_dev)
      - downlink to the Ground Unit for certain tasks (delay_to_transfer)

    Main parameters:
      - env: SimPy environment
      - selected_server: server object that exposes methods and resources (W_cpu, W_net, cpu_dev, net_dev, etc.)
      - image_size: image size in MB
      - transfer_time: transfer time from the source node to this server (already calculated)
    """
    if result_sink is None:
        result_sink = {}
    result_sink["ok"] = False
    result_sink["reason"] = None
    result_sink["fatal"] = False

    # Waste disposal assistant
    def reject(reason, fatal=False):
        # Save info in the sink
        result_sink["ok"] = False
        result_sink["reason"] = reason
        result_sink["fatal"] = fatal
        # If we aren't using a fallback, or if the failure is fatal,
        # let's actually log the rejection
        if (not allow_retry) or fatal:
            selected_server.record_rejected_task(task_id, task_type,
                                                 arrival_time_system, image_size, reason, d_cpu)

        return

    ENABLE_MONITORING = config.get("enable_queue_monitoring", False)

    task_OBS = Task(task_id, selected_server.name, 'OBS', env.now, task_type, image_size)
    bw_Bps, data_bytes = network_metrics(image_size)

    # Initialize the attributes only if monitoring is active
    if ENABLE_MONITORING:
        task_OBS.d_cpu = d_cpu
        task_OBS.d_net = data_bytes / bw_Bps if bw_Bps > 0 else 0.0
        task_OBS.deadline = arrival_time_system + D_r
        task_OBS.image_size_MB = image_size

    # Wait for the initial transfer to the selected_server
    yield env.timeout(transfer_time)
    T_deadline_abs = arrival_time_system + D_r
    D_rem = T_deadline_abs - env.now
    if D_rem <= 0:
        reject("Deadline Exceeded", fatal=True)
        result_sink["ok"] = False
        return
    arrival_time_task_queue = env.now

    # Helper for counting users + queue. Calculates the total number of tasks associated with a resource at a specific point in time by adding both the tasks currently running and those in the queue
    def _res_len_with_users(res):
        # number of tasks currently using the resource (res.users)
        users_len = len(getattr(res, "users", []))
        # Number of tasks currently waiting in the queue (res.queue)
        queue_len = len(getattr(res, "queue", []))
        return users_len + queue_len

    # Initialize queue variables for reporting
    qlen_on_enqueue_cpu = None
    qlen_on_enqueue_net = None

    eps_cpu, eps_net, time_in_queue = 0.0, 0.0, 0.0  # Estimated CPU/NET energy that we will use for reserves and subtractions
    start_time = env.now

    _, data_bytes = network_metrics(image_size)  # only the bytes
    eff_Bps = net_bw_override_Bps if (net_bw_override_Bps and net_bw_override_Bps > 0) else bw_Bps

    d_net = data_bytes / eff_Bps if eff_Bps > 0 else 0.0
    eps_net = selected_server.compute_routing_energy(data_bytes, eff_Bps if eff_Bps > 0 else 1.0,
                                                     config.get("Ptrasm", 1.0))
    net_time = d_net

    # Estimated wait times (server methods; if none exist, fall back to 0). Wn and Wc are estimated wait latencies in the queues (server methods)
    try:
        Wn = selected_server.W_net()
    except Exception:
        Wn = getattr(selected_server, "W_net", lambda: 0.0)()

    try:
        Wc = selected_server.W_cpu()
    except Exception:
        Wc = getattr(selected_server, "W_cpu", lambda: 0.0)()

    # ---------------------------
    # Branch: Generic / CPU_Intensive (solo CPU)
    # ---------------------------
    if task_type in ("Generic_Service", "CPU_Intensive"):
        # energy required to run the task on the server
        eps_cpu = selected_server.compute_execution_energy(d_cpu, C_sen, e=e_coeff)
        # Check whether the server has available power (taking reserves into account)
        if selected_server.energy - selected_server.energy_reserved < eps_cpu:
            reject("Insufficient Energy for CPU", fatal=False)
        # Check whether the server has available power (taking reserves into account)
        R = Wc + d_cpu + d_net
        if R > D_r:
            # If the estimate exceeds the configured deadline, reject the task
            selected_server.record_rejected_task(task_id, task_type, arrival_time_system, image_size,
                                                 "Deadline Exceeded", d_cpu)

            result_sink["ok"] = False
            return
            # I reserve energy to avoid race conditions with other tasks
        selected_server.energy_reserved += eps_cpu

        # I request the CPU resource (SimPy Resource) — the server maintains an internal queue
        req_cpu = selected_server.cpu_dev.request()
        qlen_on_enqueue_cpu = _res_len_with_users(selected_server.cpu_dev)
        if ENABLE_MONITORING:
            req_cpu.task_data = task_OBS
            print(
                f"[{env.now:.3f}] Task {task_id} enqueued on CPU {selected_server.name} qlen_enqueue={qlen_on_enqueue_cpu}")

        # I conserve energy
        selected_server.energy_reserved += eps_cpu
        req_cpu = selected_server.cpu_dev.request()
        # Waiting for CPU service
        yield req_cpu
        # When I get the CPU, I log the time spent in the queue
        # Now we have the resource, but is the energy still there?
        if selected_server.energy < eps_cpu:
            selected_server.energy_reserved -= eps_cpu  # Canceling the reservation
            selected_server.cpu_dev.release(req_cpu)  # Releasing the resource
            reject("Insufficient Energy for CPU", fatal=True)
            return
        time_in_queue = env.now - arrival_time_task_queue
        selected_server.cpu_busy_until = env.now + d_cpu

        # CPU execution
        yield env.timeout(d_cpu)
        selected_server.cpu_busy_until = env.now  # reset when the task finishes
        # subtract the CPU power actually consumed
        selected_server.energy -= eps_cpu
        selected_server.cpu_dev.release(req_cpu)

        # -------------------------
        # Downlink to Ground User: It is assumed that the output has a size equal to image_size
        # -------------------------
        BANDWIDTH_TO_GU_BPS = config.get("Bandwidth_to_GU_Bps", 100000)  # NOTE: Units must be B/s
        file_size_bytes = image_size * (1024 ** 2)  # MB to byte conversion
        delay_to_transfer = file_size_bytes / BANDWIDTH_TO_GU_BPS  # transmission time to GU

        # Release the memory allocation after all CPU-dependent operations
        yield env.timeout(delay_to_transfer)
        selected_server.energy_reserved -= eps_cpu

    # ---------------------------
    # Branch: CPU_and_Data_Intensive (CPU + NET)
    # ---------------------------
    elif task_type in ("CPU_and_Data_Intensive"):
        eps_cpu = selected_server.compute_execution_energy(d_cpu, C_sen, e=e_coeff)

        if selected_server.energy - selected_server.energy_reserved < (eps_cpu + eps_net):
            reject("Insufficient Energy for CPU+NET", fatal=False)
            return
        # Here, d_cpu and d_net are the service times for task R
        R = Wc + d_cpu + Wn + d_net
        if R > D_r:
            selected_server.record_rejected_task(task_id, task_type, arrival_time_system, image_size,
                                                 "Deadline Exceeded", d_cpu)

            result_sink["ok"] = False
            return

        # Total energy reserve (CPU + NET)
        selected_server.energy_reserved += (eps_cpu + eps_net)

        # enqueue on the CPU
        req_cpu = selected_server.cpu_dev.request()
        qlen_on_enqueue_cpu = _res_len_with_users(selected_server.cpu_dev)
        if ENABLE_MONITORING:
            req_cpu.task_data = task_OBS
        print(
            f"[{env.now:.3f}] Task {task_id} enqueued on CPU {selected_server.name} qlen_enqueue={qlen_on_enqueue_cpu}")

        # Wait for the CPU
        yield req_cpu

        #  === FIX: CPU Power Check After Queue ===
        if selected_server.energy < eps_cpu:
            selected_server.energy_reserved -= (eps_cpu + eps_net)
            selected_server.cpu_dev.release(req_cpu)
            reject("Insufficient Energy for CPU", fatal=True)
            return
        # ========================================

        Wc = env.now - arrival_time_task_queue
        selected_server.cpu_busy_until = env.now + d_cpu

        # CPU execution
        yield env.timeout(d_cpu)
        selected_server.cpu_busy_until = env.now
        selected_server.energy -= eps_cpu
        selected_server.cpu_dev.release(req_cpu)

        cpu_service_end = env.now

        # Now enqueue on the NET (if required)
        req_net = selected_server.net_dev.request()
        qlen_on_enqueue_net = _res_len_with_users(selected_server.net_dev)
        if ENABLE_MONITORING:
            req_net.task_data = task_OBS
        print(
            f"[{env.now:.3f}] Task {task_id} enqueued on NET {selected_server.name} qlen_enqueue_net={qlen_on_enqueue_net}")

        yield req_net

        # === FIX: Check NET Energy POST-Queue ===
        # Note: The CPU energy has already been used. Let's check if there is any NET energy left.
        if selected_server.energy < eps_net:
            selected_server.energy_reserved -= eps_net  # releasing the reserved balance
            selected_server.net_dev.release(req_net)
            # Technically, the task failed halfway through; we'll mark it as “rejected” or “fail.”
            reject("Insufficient Energy for NET", fatal=True)
            return
        # ========================================

        Wn = env.now - cpu_service_end
        if Wn < 0:
            Wn = 0.0

        selected_server.net_busy_until = env.now + net_time

        yield env.timeout(net_time)
        selected_server.net_busy_until = env.now
        if not routing_already_charged:
            selected_server.energy -= eps_net
        else:
            # Optional: If you want, note that the routing energy has already been taken from `routing_energy_from`.
            pass
        selected_server.net_dev.release(req_net)

        time_in_queue = Wc + Wn
        selected_server.energy_reserved -= (eps_cpu + eps_net)

    # ---------------------------
    # Branch: Batch (only NET)
    # ---------------------------
    elif task_type == "Batch":
        # Solo coda Network
        if selected_server.energy - selected_server.energy_reserved < eps_net:
            reject("Insufficient Energy for NET", fatal=False)
            return

        R = Wn + d_net
        if R > D_r:
            selected_server.record_rejected_task(task_id, task_type, arrival_time_system, image_size,
                                                 "Deadline Exceeded", d_cpu)

            result_sink["ok"] = False
            return

        selected_server.energy_reserved += eps_net

        req_net = selected_server.net_dev.request()
        qlen_on_enqueue_net = _res_len_with_users(selected_server.net_dev)
        if ENABLE_MONITORING:
            req_net.task_data = task_OBS
        print(
            f"[{env.now:.3f}] Batch {task_id} enqueued on NET {selected_server.name} qlen_enqueue_net={qlen_on_enqueue_net}")

        yield req_net

        # === FIX: Check NET Energy POST-Queue ===
        if selected_server.energy < eps_net:
            selected_server.energy_reserved -= eps_net
            selected_server.net_dev.release(req_net)
            reject("Insufficient Energy for NET", fatal=True)
            return
        # ========================================

        time_in_queue = env.now - arrival_time_task_queue
        selected_server.net_busy_until = env.now + net_time

        yield env.timeout(net_time)
        selected_server.net_busy_until = env.now
        if not routing_already_charged:
            selected_server.energy -= eps_net
        else:
            # Optional: If you want, note that the routing energy has already been taken from `routing_energy_from`.
            pass
        selected_server.energy_reserved -= eps_net
        selected_server.net_dev.release(req_net)

    # End of branches: Calculate final metrics and record completion
    end_time = env.now
    execution_time = end_time - start_time
    service_time = execution_time + transfer_time + time_in_queue

    selected_server.tasks.append(task_OBS)
    globals.gbl_tasks.append(task_OBS)

    if selected_server.elev_angle < config["Phi_max"] - config["Phi_buffer"]:
        task_OBS.label = 'SEN_OUT_OF_BUFF'
        if ENABLE_MONITORING:
            print(f"{selected_server} {selected_server.elev_angle}° {task_OBS.id} set as {task_OBS.label}")

    # Fallback for qlen: use the captured enqueue values; otherwise, measure the current state
    cpu_q = qlen_on_enqueue_cpu if qlen_on_enqueue_cpu is not None else _res_len_with_users(selected_server.cpu_dev)
    net_q = qlen_on_enqueue_net if qlen_on_enqueue_net is not None else _res_len_with_users(selected_server.net_dev)
    qlen = cpu_q + net_q

    # Pass the task type and energy values
    selected_server.task_completed(
        task_id, task_type, arrival_time_system, arrival_time_task_queue,
        start_time, end_time, execution_time, service_time, time_in_queue,
        selected_server.name, num_hops, qlen, transfer_time,
        image_size, DeadLine=False, exec_after_set=False,
        eps_cpu=eps_cpu, eps_net=eps_net
    )
    result_sink["ok"] = True
    result_sink["reason"] = None
    result_sink["fatal"] = False

    if ENABLE_MONITORING:
        print(
            f"[{env.now:.3f}] Task {task_id} served on {selected_server.name} qlen_enqueue_cpu={qlen_on_enqueue_cpu} qlen_enqueue_net={qlen_on_enqueue_net} time_in_queue={time_in_queue:.3f}")


def _merge_queues_for_ilp(sat_state: dict):
    """
    Combines waiting+service for CPU/NET and maps tasks to the format expected by the ILP.
    - CPU: uses ‘demand’ -> d_cpu (s)
    - NET: uses image_size_MB as d_net (MB) when available; uses ‘demand’ (MB) as a fallback
    Returns: (cpu_queue, net_queue)
    """
    cpu_q = []
    net_q = []

    # CPU
    for key in ("queue_cpu_waiting", "queue_cpu_service"):
        for t in sat_state.get(key, []):
            cpu_q.append(QueueTask(
                task_id=str(t.get("task_id")),
                d_cpu=float(t.get("demand", 0.0) or 0.0),
                d_net=0.0,  # Not used for the CPU queue
                D=float(t.get("deadline", 300.0) or 300.0)
            ))

    # NET
    for key in ("queue_net_waiting", "queue_net_service"):
        for t in sat_state.get(key, []):
            d_net_MB = t.get("image_size_MB", None)
            if d_net_MB is None or d_net_MB == 'N/A':
                # fallback: Think of “demand” as MB
                d_net_MB = float(t.get("demand", 0.0) or 0.0)
            net_q.append(QueueTask(
                task_id=str(t.get("task_id")),
                d_cpu=0.0,
                d_net=float(d_net_MB),
                D=float(t.get("deadline", 300.0) or 300.0)
            ))

    return cpu_q, net_q


def _build_snapshot_Ek(env, candidate_servers, current_task_id, d_cpu_req, d_net_MB_req, deadline_req,
                       config):
    """
    Creates a minimal Snapshot object for `solve_on_Ek` on `Ek={k}+neighbors[k]`.
    - `d_net` of the *current task* is expressed in MB (the ILP converts it to seconds using the bandwidth).
    - B = current energy; B_max = initial_energy (fallback from the configuration).
    """
    sen_map = {}
    neighbors_map = {}
    listening = []

    # Create SEN and neighborhood maps
    for srv in candidate_servers:
        st = srv.export_state(env)  # requires enable_queue_monitoring=true
        B = float(st.get("energy_budget_J", getattr(srv, "energy", 0.0)))
        B_max = float(config.get("initial_energy", B if B > 0 else 1.0))

        cpu_q, net_q = _merge_queues_for_ilp(st)

        # NOTE: ILP uses `default_net_bw_MBps`, so it is not required to set `s.net_bw_MBps`
        # but if you want, you can pass an average value per node as an add-on attribute:
        s_state = SENState(
            B=B,
            B_max=B_max,
            Rmax_norm=1.0,
            cpu_queue=cpu_q,
            net_queue=net_q,
            in_service_cpu=None,
            in_service_net=None,
            net_bw_bps=None,  # Leave it as None -> it will use default_net_bw_MBps
        )
        sen_map[srv.name] = s_state

        # neighbors: just the list of names
        neigh_names = []
        for n in srv.get_neighbors():
            neigh_names.append(n.name)
        neighbors_map[srv.name] = neigh_names

        # listening dome (if required by policy): we'll get it from the state
        if st.get("in_listening_dome", False):
            listening.append(srv.name)

    # Requests: ONLY the current task
    req = QueueTask(
        task_id=str(current_task_id),
        d_cpu=float(d_cpu_req),
        d_net=float(d_net_MB_req),  # MB!
        D=float(deadline_req)
    )

    snap = Snapshot(
        time=float(env.now),
        listening_dome=listening,
        neighbors=neighbors_map,
        sen=sen_map,
        requests=[req],
        pre_R={},  # no precomputation
        pre_E={}
    )
    return snap


# Make sure you've initialized the global dictionary at the top (only once):
# globals.gbl_task_hops = {}  # initialized during the import above
# globals.gbl_task_final_hops = {}  # initialized during the import above

def _finalize_and_assign_task(env, server_selected, server, task_id,
                              image_size, Volume_size, arrival_time_system,
                              transfer_time, task_type, d_cpu, deadline,
                              required_ram, required_disk,
                              initial_server_counter, different_server_counter, other_server_counter,
                              net_bw_override_Bps=None,
                              result_sink=None,
                              allow_retry=False):
    """
    Final block: update counters, calculate routing energy,
    record global data, and call TaskAssignment.
    """
    # Initialize `result_sink` if necessary
    if result_sink is None:
        result_sink = {}

    # Increase in the “observed applications” counter
    initial_server_counter[server_selected.name] += 1

    # Make sure the counter for this task exists
    globals.gbl_task_hops.setdefault(str(task_id), 0)

    data_bytes_global = (image_size + Volume_size) * (1024 ** 2)
    bw_Bps_global = bw_Bps

    # Let's calculate whether a physical transfer occurs, and if so,
    # apply the energy and increment the per-task counter.
    num_hops_local = globals.gbl_task_hops.get(str(task_id), 0)

    if server != server_selected:
        # Here we are about to perform a transfer from server_selected to server
        # Calculate routing energy
        bw_MBps_link = server_selected.get_bandwidth(server)

        if bw_MBps_link is not None and bw_MBps_link > 0:
            bw_Bps_for_energy = bw_MBps_link * (1024 ** 2)
            data_bytes_for_energy = data_bytes_global
        else:
            bw_Bps_for_energy = bw_Bps_global if bw_Bps_global > 0 else 0
            data_bytes_for_energy = data_bytes_global

        if bw_Bps_for_energy > 0:
            Ptrasm = config.get("Ptrasm", 1.0)
            eps_net = server_selected.compute_routing_energy(data_bytes_for_energy, bw_Bps_for_energy, Ptrasm)
        else:
            eps_net = 0.0

        # >>> FIX: ENERGY ROUTING CONTROL <<<
        if server_selected.energy < eps_net:
            print(
                f"[{env.now:.2f}] [Task {task_id}] Routing FAIL {server_selected.name}->{server.name}: No Energy for TX")
            if result_sink is not None:
                result_sink["ok"] = False
                result_sink["reason"] = "Routing Energy Exhausted"
                result_sink["fatal"] = True  # There's no point in trying again if the source is dead.

            # Record the rejection for statistical purposes
            server_selected.record_rejected_task(task_id, task_type, arrival_time_system, image_size,
                                                 "Routing Energy Exhausted", d_cpu)
            return
        # >>> FINE FIX <<<

        # DRAW ENERGY: It's now safe to do so
        server_selected.energy -= eps_net
        print(f"[{env.now:.2f}] [Task {task_id}] Routed {server_selected.name} -> {server.name} | "
              f"E_NET={eps_net:.6f} J | Remaining={server_selected.energy:.2f} J")

        # ---- Here we decide on the semantics: counting hops as “performed transfers” ----
        globals.gbl_task_hops[str(task_id)] = globals.gbl_task_hops.get(str(task_id), 0) + 1
        num_hops_local = globals.gbl_task_hops[str(task_id)]

        # Update to fallback/other counters (they remain unchanged)
        different_server_counter[server_selected.name] += 1
        other_server_counter[server.name] += 1

    # Prepare the log data (we'll use the current num_hops_local)
    task_data = {
        "task_id": task_id,
        "arrival_time": arrival_time_system,
        "type": task_type,
        "ram": required_ram,
        "disk": required_disk,
        "image_size": image_size,
        "exec_time": d_cpu,
        "transfer_time": transfer_time,
        "performed_transfers": globals.gbl_task_hops.get(str(task_id), 0),
        "final_hops": None,
        "execution_server": server.name
    }

    # Call TaskAssignment_ILP, passing num_hops_local
    local_sink = result_sink if result_sink is not None else {}
    yield from TaskAssignment_ILP(env, server, task_id, image_size,
                                  arrival_time_system, num_hops_local, transfer_time,
                                  task_type, d_cpu, deadline,
                                  net_bw_override_Bps=net_bw_override_Bps,
                                  result_sink=local_sink,
                                  allow_retry=allow_retry, routing_already_charged=True, routing_energy_from=server_selected.name)

    # Record global data: Save the final count ONLY if it was successfully assigned
    ok_assigned = bool(local_sink.get("ok", False))
    performed = globals.gbl_task_hops.get(str(task_id), 0)

    if ok_assigned:
        globals.gbl_task_final_hops[str(task_id)] = performed
        task_data["final_hops"] = performed
        globals.gbl_generated_tasks_data.append(task_data)
    else:
        reason = local_sink.get("reason")
        print(f"[{env.now:.2f}] [Task {task_id}] assignment failed on {server.name} reason={reason}")

    return



def SearchNode_ILP_Hybrid_v2(env, server_selected, task_id, required_ram, required_disk, image_size, Volume_size,
                             arrival_time_system,
                             initial_server_counter, different_server_counter, other_server_counter,
                             task_type, max_energy, d_cpu, deadline):
    """
    Wrapper: Attempts ILP if configured in config[“SearchNode”].
    Otherwise, or if that fails, it executes SearchNode_Heuristic_v1.
    """

    if str(config.get("SearchNode", "")).upper() == "ILP":

        # --- Start of ILP block  ---
        print(f"[{env.now:.2f}] [Task {task_id}] Entering ILP branch (v2)")

        # 1. Get the neighbors (required for ILP)
        neighbors_at_distance_one = list(server_selected.get_neighbors())
        if server_selected not in neighbors_at_distance_one:
            neighbors_at_distance_one.append(server_selected)

        d_net_MB_req = float(image_size + (Volume_size or 0.0))

        # Local Snapshot and Resolution
        snap = _build_snapshot_Ek(
            env=env,
            candidate_servers=neighbors_at_distance_one,
            current_task_id=task_id,
            d_cpu_req=float(d_cpu),
            d_net_MB_req=float(d_net_MB_req),
            deadline_req=float(deadline),
            config=config
        )

        # --- Physical Coefficients ---
        P_net = float(config.get("Ptrasm", 1.0))
        alpha_cpu, alpha_net_J_per_byte = alpha_from_physics(C_sen, e_coeff, P_net, bw_Bps)
        alpha_net = alpha_net_J_per_byte * (1024 ** 2)  # J/MB
        w_e = float(config.get("ilp_weights", {}).get("w_e", 0.5))
        w_R = float(config.get("ilp_weights", {}).get("w_R", 0.5))

        # --- Built-in function to extract results ---
        def _extract_chosen_server(ilp_res, task_id):
            # ... (your extraction logic is correct) ...
            tid = str(task_id)
            if isinstance(ilp_res, dict):
                a = ilp_res.get("assignments")
                if isinstance(a, list):
                    for item in a:
                        if str(item.get("task") or item.get("task_id") or item.get("id")) == tid:
                            sen = item.get("sen") or item.get("server")
                            if sen: return str(sen)
                    if len(a) == 1 and isinstance(a[0], dict):
                        sen = a[0].get("sen") or a[0].get("server")
                        if sen: return str(sen)
                if isinstance(a, dict):
                    v = a.get(tid)
                    if v is not None: return str(v)
                for key in ("solution", "assignments_list", "result", "x"):
                    if key in ilp_res:
                        v = _extract_chosen_server(ilp_res[key], task_id)
                        if v: return v
            if isinstance(ilp_res, list):
                for item in ilp_res:
                    if isinstance(item, (tuple, list)) and len(item) >= 2 and str(item[0]) == tid:
                        return str(item[1])
                    if isinstance(item, dict):
                        itid = str(item.get("task_id") or item.get("task") or item.get("id") or "")
                        if itid == tid:
                            sen = item.get("sen") or item.get("server") or item.get("assignment") or item.get("value")
                            if sen: return str(sen)
                if len(ilp_res) == 1 and isinstance(ilp_res[0], str):
                    return ilp_res[0]
            if isinstance(ilp_res, str):
                return ilp_res
            return None

        #====== CALL TO THE SOLVER ======
        objective_mode = str(config.get("ilp_objective", "weighted")).lower()
        if objective_mode == "hierarchical":
            ilp_res = solve_on_Ek_hierarchical(
                snapshot=snap, k=server_selected.name, picked_tasks=[str(task_id)],
                primary=str(config.get("lexi_primary", "energy")).lower(),
                tol=float(config.get("lexi_tol", 0.10)),
                alpha_cpu=alpha_cpu, alpha_net=alpha_net,
                solver_name=str(config.get("ilp_solver", "CBC")),
                default_net_bw_MBps=bw_MBps,
                debug=bool(config.get("ilp_debug", False)),
                tasks_from_prof=False
            )
        else:
            ilp_res = solve_on_Ek(
                snapshot=snap, k=server_selected.name, picked_tasks=[str(task_id)],
                w_energy=w_e, w_time=w_R,
                alpha_cpu=alpha_cpu, alpha_net=alpha_net,
                solver_name="CBC", use_node_Rmax_norm=False,
                default_net_bw_MBps=bw_MBps,
                debug=False, tasks_from_prof=False
            )

        print(f"[ILP] res_type={type(ilp_res).__name__} value_preview={str(ilp_res)[:160]}")
        chosen_server_name = _extract_chosen_server(ilp_res, task_id)

        # --- ILP Finalization (if successful) ---
        cfg_fb = config.get("fallback_retry", {"enabled": True, "max_attempts": 3, "deadline_slack": 0.05})
        fallback_enabled = bool(cfg_fb.get("enabled", True))
        fallback_attempts = int(cfg_fb.get("max_attempts", 3))
        deadline_slack = float(cfg_fb.get("deadline_slack", 0.0))

        neighbors = list(neighbors_at_distance_one)  # Also include server_selected

        def _try_assign_once(exec_server,
                             force_transfer_time=None,
                             force_bw_Bps=None):
            lat = server_selected.get_latency(exec_server)
            bw_MBps_link = server_selected.get_bandwidth(exec_server)

            if force_transfer_time is not None:
                transfer_time_local = force_transfer_time
                net_bw_override_Bps = force_bw_Bps
            else:
                if (lat is not None) and (bw_MBps_link is not None) and bw_MBps_link > 0:
                    transfer_time_local = ((image_size + Volume_size) * (1024 ** 2)
                                           / (bw_MBps_link * (1024 ** 2))) + lat
                    net_bw_override_Bps = bw_MBps_link * (1024 ** 2)
                else:
                    transfer_time_local = 0.0
                    net_bw_override_Bps = None

            rs = {}
            yield from _finalize_and_assign_task(
                env, server_selected, exec_server, task_id,
                image_size, Volume_size, arrival_time_system,
                transfer_time_local, task_type, d_cpu, deadline,
                required_ram, required_disk,
                initial_server_counter, different_server_counter, other_server_counter,
                net_bw_override_Bps=net_bw_override_Bps,
                result_sink=rs,
                allow_retry=True  # <--- Here's the key
            )
            return rs.get("ok", False), rs.get("reason"), rs.get("fatal", False)

        # ====== CASE 1: ILP Chose a Server ======
        if chosen_server_name:
            print(f"[{env.now:.2f}] [Task {task_id}] ILP chose server: {chosen_server_name}")
            server = next((s for s in neighbors if s.name == chosen_server_name), server_selected)

            ok, reason, fatal = (yield from _try_assign_once(server))
            if ok:
                return

            # If the failure is fatal (e.g., “Deadline Exceeded”),
            # there's no point in retrying on other SENs
            if fatal:
                return

            # Otherwise, I can try again on other random SENs (fallback CASE 1)
            if not fallback_enabled or fallback_attempts <= 0:
                return

            excluded = {server.name}
            attempts = 0
            assigned = False
            last_reason = reason

            while attempts < fallback_attempts and not assigned:
                attempts += 1
                available = [s for s in neighbors if s.name not in excluded]
                if not available:
                    break

                sen_alt = globals.rnd.choice(available)
                excluded.add(sen_alt.name)

                print(f"[FALLBACK-ILP-C1][{env.now:.2f}] Task {task_id} tenta su {sen_alt.name} (tentativo {attempts})")
                ok_alt, reason_alt, fatal_alt = (yield from _try_assign_once(sen_alt))
                if ok_alt:
                    assigned = True
                else:
                    last_reason = reason_alt or last_reason
                    if fatal_alt:
                        break  # The next one would be pointless

            # If it still hasn't been assigned even after the fallback, you already have the internal rejections/logs:
            return

        # ====== CASE 2: ILP infeasible/none ======
        print(f"[{env.now:.2f}] [Task {task_id}] ILP infeasible/none → fallback random (ILP-only).")
        if not fallback_enabled or fallback_attempts <= 0:
            return

        excluded = set()
        attempts = 0
        assigned = False

        while attempts < fallback_attempts and not assigned:
            attempts += 1

            # I'll take the list of candidates who haven't been tested yet
            available = [s for s in neighbors if s.name not in excluded]
            if not available:
                print(f"[FALLBACK-ILP][{env.now:.2f}] Task {task_id}: nessun neighbor disponibile per il fallback.")
                break

            # I'll pick a SEN at random
            sen_alt = globals.rnd.choice(available)
            excluded.add(sen_alt.name)

            # Calculate transfer_time and bandwidth as you normally would
            lat = server_selected.get_latency(sen_alt)
            bw_MBps_link = server_selected.get_bandwidth(sen_alt)

            if (lat is not None) and (bw_MBps_link is not None) and bw_MBps_link > 0:
                bw_Bps_link_alt = bw_MBps_link * (1024 ** 2)
                total_data_bytes = (image_size + Volume_size) * (1024 ** 2)
                transfer_time_alt = (total_data_bytes / bw_Bps_link_alt) + lat
            else:
                bw_Bps_link_alt = None
                transfer_time_alt = 0.0

            print(f"[FALLBACK-ILP][{env.now:.2f}] Task {task_id} tenta su {sen_alt.name} (tentativo {attempts})")

            # I'll try to actually assign the task to this SEN
            result_sink = {}
            yield from _finalize_and_assign_task(
                env, server_selected, sen_alt, task_id,
                image_size, Volume_size, arrival_time_system,
                transfer_time_alt, task_type, d_cpu, deadline,
                required_ram, required_disk,
                initial_server_counter, different_server_counter, other_server_counter,
                net_bw_override_Bps=bw_Bps_link_alt,
                result_sink=result_sink
            )

            assigned = bool(result_sink.get("ok", False))

        # If, even after the fallback attempts, I haven't assigned the task, I'll let the rejections
        # recorded in TaskAssignment remain as the “final verdict”
        return

    # =========================
    # BRANCH: EURISTIC (called in v1)
    # Reasons for being here:
    # 1. config[“SearchNode”] was not ‘ILP’
    # 2. config[“SearchNode”] was “ILP” but the solver failed
    # =========================

    print(f"[{env.now:.2f}] [Task {task_id}] Entering HEURISTIC branch (v2) -> calling v1")

    # We pass all the original arguments to v1, which is the standard heuristic function.
    '''yield from simulation.SearchNode_Heuristic_v1(
        env, server_selected, task_id, required_ram, required_disk, image_size, Volume_size,
        arrival_time_system,
        initial_server_counter, different_server_counter, other_server_counter,
        task_type, max_energy, d_cpu, deadline
    )'''
