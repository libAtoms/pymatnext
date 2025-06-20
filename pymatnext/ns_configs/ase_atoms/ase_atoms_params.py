"""dict with template of parameters for [configs] and [configs.walk] sections
"""

param_defaults_ase_atoms = {
    "full_composition": "",
    "composition": ["_REQ_", "AbCdE2"],
    "n_atoms": ["_REQ_", 1],
    "dims": 3,
    "pbc": [True, True, True],
    "initial_rand_vol_per_atom": ["_REQ_", 1.0],
    "initial_rand_min_dist": ["_REQ_", 1.0],
    "initial_rand_n_tries": 10,
    "calculator": {
        "type": ["_REQ_", "calc_type"],
        "args": { "_IGNORE_": True }
    },
    "walk": ["_REQ_", "_IGNORE_"]
}

param_defaults_walk = {
    "gmc_traj_len": 8,
    "cell_traj_len": 8,
    "type_traj_len": 8,

    "gmc_proportion": 0.0,
    "cell_proportion": 0.0,
    "type_proportion": 0.0,

    "max_step_size": {
        "pos_gmc_each_atom": -0.1,
        "cell_volume_per_atom": -0.05,
        "cell_shear_per_rt3_atom": -1.0,
        "cell_stretch": 0.2
    },

    "step_size": {
        "pos_gmc_each_atom": -1.0,
        "cell_volume_per_atom": -1.0,
        "cell_shear_per_rt3_atom": -1.0,
        "cell_stretch": -1.0
    },

    "cell": {
        "min_aspect_ratio": 0.8,
        "flat_V_prior": True,
        "pressure_GPa": 0.0,
        "pressure": None,
        "submove_probabilities": {
            "volume": 0.7,
            "shear": 0.15,
            "stretch": 0.15
        }
    },

    "type": {
        "sGC": False,
        "mu": { "_IGNORE_": True }
    }
}
