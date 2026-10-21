"""
Global State and Configuration Management
=========================================

This module acts as the central repository for the simulation's shared state, configuration data,
and global objects. It is responsible for bootstrapping the simulation environment by:

1.  **Resolving Configuration Paths**: Determining which configuration files to load based on
    environment variables, command-line arguments, or default values.
2.  **Loading Configurations**: Parsing ``json5`` files for simulation parameters and task profiles.
3.  **Initializing Randomness**: Setting up seeded random number generators (Python `random` and NumPy)
    to ensure simulation reproducibility.
4.  **Managing Global State**: Declaring and initializing global counters, lists (e.g., `edge_servers`),
    and synchronization primitives (locks) used across different modules.

Usage
-----
Importing this module immediately triggers the configuration loading process.
It should be imported early in the `main.py` execution flow.

>>> import globals
>>> print(globals.config["simulation_duration"])
>>> rng_val = globals.rnd.random()
"""
import threading
import os
import sys
import json5
import random as _random
import numpy as np
from utils import colorize

# Default Names
DEFAULT_CONFIG = 'config.json5'
DEFAULT_IMG = 'img_resolution.json5'
DEFAULT_OGMS_TABLES = 'data/OGMs_table.json'

# ---------------------------------------------------------------------
# Determine which files to use:
# 1) Preferred: SIM_CONFIG and SIM_IMG environment variables (used by the launcher)
# 2) Fallback: sys.argv[1], sys.argv[2] (if main.py is called with arguments)
# 3) Final fallback: DEFAULT_CONFIG, DEFAULT_IMG
# ---------------------------------------------------------------------
def _resolve_config_paths():
    print(f"[DEBUG GLOBALS] sys.argv: {sys.argv}")
    # 1) env vars
    cfg_env = os.environ.get('SIM_CONFIG')
    img_env = os.environ.get('SIM_IMG')

    if cfg_env or img_env:
        config_file = cfg_env if cfg_env else DEFAULT_CONFIG
        img_file = img_env if img_env else DEFAULT_IMG
        return config_file, img_file

    # 2) argv fallback (main.py <config> <img>)
    if len(sys.argv) >= 3:
        config_file, img_file = sys.argv[1], sys.argv[2]
        return config_file, img_file

    if len(sys.argv) == 2:
        # only config passed via argv
        return sys.argv[1], DEFAULT_IMG

    # 3) default
    return DEFAULT_CONFIG, DEFAULT_IMG

# Resolve the paths
config_file_path, img_resolution = _resolve_config_paths()

# Read the main configuration file
try:
    with open(config_file_path, 'r') as config_file:
        config = json5.load(config_file)
        print(colorize(f"[INFO] Configuration loaded from: {config_file_path}", "green"))
except FileNotFoundError:
    config = None
    sys.exit(colorize(
        f"Unable to find the configuration file {config_file_path}. Make sure the file exists.",
        "red"))
except Exception as e:
    config = None
    sys.exit(colorize(f"[ERROR] Error loading {config_file_path}: {e}", "red"))

# Read the image configuration (TASK_GENERATOR_PARAMS)
try:
    with open(img_resolution, 'r') as res_file:
        _img_cfg = json5.load(res_file)
        # Compatibility: We're keeping the same behavior you were used to
        if "TASK_GENERATOR_PARAMS" in _img_cfg:
            resolution_config = _img_cfg["TASK_GENERATOR_PARAMS"]
        else:
            # fallback if the file is already the direct dictionary
            resolution_config = _img_cfg
        print(colorize(f"[INFO] Image resolution settings uploaded by: {img_resolution}", "green"))
except FileNotFoundError:
    resolution_config = None
    sys.exit(colorize(
        f"[ERROR] Unable to find the image file {img_resolution}. Make sure the file exists.",
        "red"))
except Exception as e:
    resolution_config = None
    sys.exit(colorize(f"[ERROR] Error loading {img_resolution}: {e}", "red"))

# ---------------------------------------------------------------------
# Seed and RNG Initialization
# ---------------------------------------------------------------------
_seed = int(config.get("seed", 42))

rnd = _random.Random(_seed)
np.random.seed(_seed)
rnd_np = np.random.default_rng(_seed)


initial_server_counter = {}
different_server_counter = {}
other_server_counter = {}
gbl_generated_tasks_data = []

global_access_point = []
next_server_index = 0
gbl_batch_completed = []
config_index = 0
gbl_task_hops = {}
gbl_task_final_hops = {}
edge_servers = []
edge_servers_topology = []

observer = None
ist_in_conf = None

gbl_tasks = []
gbl_packet = []

lock_access_edge_servers_topology = threading.Lock()

OGMs_tables = None
data_configurations = None

# end of globals.py
