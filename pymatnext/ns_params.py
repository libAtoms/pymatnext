"""dict with template of parameters for [ns] section
"""

import numpy as np

param_defaults = {
    "n_walkers": ["_REQ_", 1],
    "walk_length": ["_REQ_", 1],
    "configs_module": ["_REQ_", "ns_config_mod"],
    "exit_conditions": { "_IGNORE_": True }
}
