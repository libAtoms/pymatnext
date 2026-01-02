# Usage

```
pymatnext [ --random_seed / s <seed> ] [ --output_file_postfix / -p <postfix> ] [ --max_iter / -i <max iter> ] <params_file>
```

Do a nested sampling run based on the parameters in `<params_file>` in [toml format](https://toml.io/en/).


## Command line arguments

 - `--random_seed / s <seed>`: set a random seed (overriding parameter file)
 - `--max_iter / -i <max_iter>`: maximum NS iteration (overriding arameter file)
 - `--output_file_postfix / -p <postfix>`: a suffix to all output files that is added to the parameter file value

By setting different random seeds and output postfix strings, multiple independent runs can be started (for better
sampling) without having to modify the parameter file.

## Output

### Nested sampling quantities

 - sampled quantities in `<global.output_file_prefix><output_file_postfix>.NS_samples`
   - JSON format header line, prefixed by `#`,  describing NS run parameters and quantities in file, for analysis
   - one line every `<global.sample_interval>` NS iterations, with iteration number, global index of
     configuration selected, NS quantity, and configuration-specific quantities specified in header `extra`
     dict item.

 - sampled configurations in `<global.output_file_prefix><output_file_postfix>.traj.<filename_suffix>` 
   - One configuration in a type-specific format (`extxyz` for atomic configurations) every
    `<global.traj_interval>` NS iterations

 - snapshots
   - NS state in `<global.output_file_prefix><output_file_postfix>.iter_<iter>.state.json>`
   - NS configurations in `<global.output_file_prefix><output_file_postfix>.iter_<iter>.configs.<filename_suffix>>`

## Example

Full featured example of a small system with variable cell, semi-grand-canonical run using LAMMPS internal propagators.
```
[global]

    output_filename_prefix = "EAM_LAMMPS_sGC"
    random_seed = 5

    max_iter = 50000

    stdout_report_interval = 60

    snapshot_interval = 50000

    [global.step_size_tune]

        interval = 1000

[ns]

    n_walkers = 280
    walk_length = 400
    configs_module = "pymatnext.ns_configs.ase_atoms"

    [ns.exit_conditions]

        module = "pymatnext.loop_exit.Z_of_T"

        [ns.exit_conditions.module_kwargs]
            T_to_ns_quant = "kB"
            T = 5000.0

[configs]

    composition = "AlCu"
    n_atoms = 16
    initial_rand_vol_per_atom = 800.0
    initial_rand_min_dist = 0.5

    [configs.calculator]

        type = "LAMMPS"

        [configs.calculator.args]

            cmds = ["pair_style eam/alloy", "pair_coeff * * <PATH_TO_POTENTIALS>/AlCu_Zhou04.eam.alloy Al Cu"]

            types.13 = 1
            types.29 = 2

    [configs.walk]

        gmc_traj_len = 8
        cell_traj_len = 4
        type_traj_len = 8

        gmc_proportion = 1.0
        cell_proportion = 1.5
        type_proportion = 1.0

        [configs.walk.cell]

            [configs.walk.cell.submove_probabilities]
                volume = 0.33
                shear = 0.33
                stretch = 0.33

        [configs.walk.type]
            sGC = true
            mu.13 = 0.1
            mu.29 = 0.2
```

Hopefully section and key names are self explanatory.

### Additional notes on run parameters

  - All intervals are in NS iterations, except `stdout_report_interval_s` which is in seconds
  - `configs.calculator.type` can be `"ASE"` or `"LAMMPS"`
    - if `"LAMMPS"`, `args` consists of `cmds`, with LAMMPS commands, and `types` dict (one key for each species)
    - if `"ASE"`, `args` consists of `module` key with module that defines a `calc` symbol containing an `ase.calculators.Calculator` object
  - `configs.walk.*_traj_len` controls the number of steps in a walk block of that type
  - `configs.walk.*_proportion` controls the fraction of steps overall that are used for that type of move
  - If `config.walk.type.sGC = true`, a dict of `mu` values, one per species, is required.
  - There are position, cell, and atom-type walks, with associated step size parameters that are auto-tuned. Default values depends on
    an overall volume scale given by `initial_rand_vol_per_atom` and corresponding length scale given by its cube root.
    - Maxima are in `[configs.walk.max_step_size]` section. Defaults for first three are negative.
      - `pos_gmc_each_atom`: distance (typically A) that each atom should typically make in GMC step. If negative, used as multiplier for
        length scale.
      - `cell_volume_per_atom`: change in volume (typically A^3), will also be scaled by number of atoms. If negative, used as multiplier
        for volume scale.
      - `cell_shear_per_rt3_atom`: cell shear magnitude (typically A) which multiplies _normalized_ cell vectors, will also be scaled by 
        cube root of number of atoms.  If negative used as multiplier for length scale.
      - `cell_stretch`: cell stretch, fractional (i.e. strain)
    - Initial values with same key names are in `[configs.walk.step_size]` section. Any negative values are replaced with half the corresponding maximum.
