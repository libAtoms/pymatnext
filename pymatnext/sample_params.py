"""dict with template of parameters for [global] section"
"""

import numpy as np

param_defaults = {
    "global": {
        "output_filename_prefix": "NS",
        "random_seed": -1,
        "max_iter": -1,
        "stdout_report_interval_s": 60,
        "sample_interval": 1,
        "traj_interval": 100,
        "snapshot_interval": 10000,
        "step_size_tune": {
            "interval": 1000,
            "n_configs": 1,
            "min_accept_rate": 0.25,
            "max_accept_rate": 0.5,
            "adjust_factor": 1.25
        }
    },
    "ns": ["_REQ_", "_IGNORE_"],
    "configs": ["_REQ_", "_IGNORE_"]
}
