# `pymatnext` input TOML parameters

This reference documents every field emitted by the current `pymatnext
--defaults` command. The template below is TOML-compatible: descriptions and
type annotations are TOML comments. Commented assignments are required fields
with no default; uncomment them and provide a value. The TOML string
`"_NONE_"` is used to set a Python `None` value.

The itemized list shows the nesting of the TOML tables and the fields that can
be set in each table. Each field includes its TOML default and description;
fields without a default must be specified by the user.

- `[general]`
  - `output_filename_prefix = "NS"` #  prefix for all output files
  - `output_filename_prefix_extra = ""` #  extra string to add to output_filename_prefix, designed for easy per-realization overriding
  - `random_seed` *(no default; must be specified by the user)* #  seed for random number generator
  - `max_iter = "_NONE_"` #  maximum NS iteration
  - `stdout_report_interval_s = 60` #  interval in seconds between reports to stdout, < 0 for no reports
  - `sample_interval = 1` #  interval in NS iterations between saved NS samples, <= 0 to disable
  - `traj_interval = 100` #  interval in NS iterations between saved NS configurations, <= 0 to disable
  - `snapshot_interval = 10000` #  interval in NS iterations between restart snapshots, <= 0 to disable
  - `snapshot_save_old = 2` #  how many old snapshots so save
  - `clone_history = false` #  save a full history of which config was cloned at each iteration
  - `[general.step_size_tune]`
    - `interval = 1000` #  NS iteration interval between step-size tuning
    - `n_configs = 1` #  number of configs to run when computing step size related statistics
    - `min_accept_rate = 0.25` #  minimum accept rate for tuning step size
    - `max_accept_rate = 0.5` #  maximum accept rate for tuning step size
    - `adjust_factor = 1.25` #  factor by which step size is multiplied/divided at each tuning iteration
  - `[general.walk_traj_info]`
    - `iter_min = "_NONE_"` #  first iteration at which walk trajectory is saved
    - `iter_max = "_NONE_"` #  last iteration (inclusive) at which walk trajectory is saved, negative for no maximum
    - `interval = "_NONE_"` #  interval at which walk trajectory is saved
    - `avg_times = []` #  save walk trajectory with time averaging over these time scales
- `[ns]`
  - `n_walkers` *(no default; must be specified by the user)* #  Number of NS walkers (live points)
  - `walk_length` *(no default; must be specified by the user)* #  Length of NS walk to produce a new, decorrelated config, in number of energy/force evaluations
  - `step_size_tune_walk_length = "_NONE_"` #  NS walk length used to tune step size
  - `configs_module` *(no default; must be specified by the user)* #  module that defines configurations
  - `initial_config_file = "_NONE_"` #  file with initial configurations, if not generated randomly
  - `initial_max_val = "_NONE_"` #  value of overriding NS energy initial maxmimum
  - `[ns.exit_conditions]`
    - `module = "_NONE_"` #  module that defines an ExitLoop class that checks for an exit condition, with __call__ method that takes current iteration and max value and returns a bool
    - `module_kwargs = {}` #  arbitrary kwargs for ExitLoop constructor in addition to the NS object itself
- `[configs]`
  - `full_composition = ""` #  composition that spans all possible elements that could appear, if different from initial
  - `composition` *(no default; must be specified by the user)* #  initial composition of configurations
  - `n_atoms` *(no default; must be specified by the user)* #  number of atoms in each configuration
  - `dims = 3` #  number of dimensions (2 or 3)
  - `pbc = [true, true, true]` #  periodicity of system along each cell vector
  - `initial_rand_vol_per_atom` *(no default; must be specified by the user)* #  random initial configuration volume per atom
  - `initial_rand_min_dist` *(no default; must be specified by the user)* #  random initial configuration minimum distance between atoms
  - `initial_rand_n_tries = 10` #  number of tries to get initial configuration that obeys minimum distances
  - `[configs.calculator]`
    - `type` *(no default; must be specified by the user)* #  calculator type
    - `args = {}` `#  arbitrary args for calculator constructor. If 'ASE', 'module' with name of importable module defining a `calc` Calculator object. If 'LAMMPS', 'cmds': list of `pair_style ...` etc. lammps commands, 'types': dict with atomic numbers or chemical symbols as keys and lammps types as values, 'header': optional list of header commands, 'cmd_args': optional command args, 'log_file': optional file for lammps log, 'name': optional lammps shared lib name, 'activate_mliappy_kokkos': optional bool for mliappy kokkos, 'boundary': optional args for lammps boundary command`
  - `[configs.walk]`
    - `gmc_traj_len = 8` #  length of GMC walks
    - `cell_traj_len = 8` #  length of cell move walks
    - `type_traj_len = 8` #  length of type (sGC) move walks
    - `gmc_proportion = 0.0` #  proportion of steps to do GMC walks with
    - `cell_proportion = 0.0` #  proportion of steps to do cell walks with
    - `type_proportion = 0.0` #  proportion of steps to do type (sGC) walks with
    - `combined = false` #  use NS walk function that combines all move types
    - `[configs.walk.max_step_size]`
      - `pos_gmc_each_atom = -0.1` #  maximum GMC position move (for each atom), if negative multiplied by cube root of atomic volume
      - `cell_volume_per_atom = -0.05` #  maximum cell volume step (per atom), if negative multiplied by atomic volume
      - `cell_shear_per_rt3_atom = -1.0` #  maximum cell shear step (per natoms^1/3), if negative multiplied by cube root of atomic volume
      - `cell_stretch = 0.2` #  maximum cell stretch (unitless strain)
    - `[configs.walk.step_size]`
      - `pos_gmc_each_atom = -1.0` #  initial GMC positions move (for each atom)
      - `cell_volume_per_atom = -1.0` #  initial cell volume step (per atom)
      - `cell_shear_per_rt3_atom = -1.0` #  initial cell shear step (per natoms^1/3)
      - `cell_stretch = -1.0` #  initial cell stretch step
    - `[configs.walk.cell]`
      - `min_aspect_ratio = 0.8` #  minimum cell aspect ratio to accept
      - `flat_V_prior = true` #  use a prior independent of volume, rather than ensemble-correct V^natoms
      - `pressure = 0.0` #  applied pressure in eV/A^3 (or pressure_GPa in GPa)
      - `[configs.walk.cell.submove_probabilities]`
        - `volume = 0.7` #  probability to try a cell volume move
        - `shear = 0.15` #  probability to try a cell shear move
        - `stretch = 0.15` #  probability to do a cell stretch move
    - `[configs.walk.type]`
      - `sGC = false` #  semi-grand-canonical (species change) moves.
      - `mu = {}` #  dict with atomic numbers as keys and chemical potentials as values for semi-grand-canonical moves
