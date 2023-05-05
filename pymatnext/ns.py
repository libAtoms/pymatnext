from copy import deepcopy
import importlib

import re
from pathlib import Path

import json

import numpy as np

from .ns_utils import rngs as new_rngs

from pymatnext.params import check_fill_defaults
from .ns_params import param_defaults


class NS:
    """Nested sampling object

    Parameters
    ----------
    params_ns: dict
        setup parameters
    comm: Communicator
        communicator for parallelism
    MPI: mpi4py.MPI or equivalent namespace
        namespace needed for symbols require to call MPI functions
    random_seed: int or None
        random seed for RNGs
    params_config: dict
        dict for [configs] toml section
    output_filename_prefix: str / Path
        prefix of all output filenames
    different_n_rng_local: bool, default False
        allow restart from snapshot even if number of local rngs stored does not match number this time
    extra_config: bool, default False
        allocate storage for an extra config, e.g. to use as a buffer
    """
    def __init__(self, params_ns, comm, MPI, random_seed, params_configs, output_filename_prefix, different_n_rng_local=False,
                 extra_config=False):
        check_fill_defaults(params_ns, param_defaults, label="ns")

        self.comm = comm
        self.MPI = MPI
        self.n_configs_global = params_ns["n_walkers"]
        self.global_walk_length = params_ns["walk_length"]

        # get configuration constructor from module that defines exactly one class whose name starts with NSConfig_
        nsconfig_mod = importlib.import_module(params_ns["configs_module"])
        nsconfig_classes = [symb for symb in dir(nsconfig_mod) if symb.startswith("NSConfig_")]
        assert len(nsconfig_classes) == 1
        self.NSConfig = getattr(nsconfig_mod, nsconfig_classes[0])
        self.NSConfig.initialize(params_configs)

        # local walk is shortened by factor of number of parallel processes
        self.local_walk_length = int(self.global_walk_length / comm.size)

        # allocate correct numbers of configs to each node
        if self.n_configs_global % self.comm.size == 0:
            self.n_configs_local = self.n_configs_global // self.comm.size
            self.max_n_configs_local = self.n_configs_global // self.comm.size
            # self.n_configs_global_offset = self.n_configs_local * self.comm.rank
        else:
            raise ValueError(f"Number of configurations {self.n_configs_global} must be divisible by number of processes {self.comm.size}")

        # number of quantities depends on the type of config, so these must want for call to init_configs()
        self._allgatherv_counts = None
        self._allgatherv_displt = None
        self._ns_quants_global = None

        # set when find_max() is called
        self.rank_of_max = None
        self.local_ind_of_max = None
        self.max_val = None

        self.local_configs = []

        old_state_files = NS._old_state_files(output_filename_prefix)
        if len(old_state_files) > 0:
            # found a snapshot
            snapshot_state_file = old_state_files[-1]
            self.snapshot_iter = NS._iter_from_state_file(snapshot_state_file)

            initial_config_file = snapshot_state_file.replace(".state.json", f".configs{self.NSConfig.filename_suffix}")
            with open(snapshot_state_file) as fin:
                snapshot_state = json.load(fin)
        else:
            # no snapshot, generate from scratch
            snapshot_state = {}
            if params_ns["initial_config_file"] != "_NONE_":
                initial_config_file = params_ns["initial_config_file"]
            else:
                initial_config_file = None
            self.snapshot_iter = -1

        self.init_rngs(random_seed, snapshot_state.get("rngs", None), different_nlocal=different_n_rng_local)
        self.init_configs(params_configs, initial_config_file, extra=extra_config)


    def report_store(self, loop_iter):
        """Store quantities needed for NSConfig-specific report on progress of NS iteration

        Parameters
        ----------
        loop_iter: int
            current NS loop iteration number
        """
        return self.NSConfig.report_store(loop_iter, self.max_val)


    def report(self):
        """Write report on progress of NS iteration

        Returns
        -------
        report_str: text line with report info
        """
        return self.NSConfig.report()

    @staticmethod
    def _iter_from_state_file(filename):
        """extract iteration number from snapshot filename

        Parameters
        ----------
        filename: str
            filename

        Returns
        -------
        iter_i: int iteration number
        """
        return int(re.sub('.state.json', '', re.sub(r'.*iter_', '', filename)))

    @staticmethod
    def _old_state_files(output_filename_prefix):
        """get sorted list of old state snapshot files

        Parameters
        ----------
        output_filename_prefix: str
            prefix of output filenames

        Returns
        -------
        old_state_files: list(str) filenames, in order of increasing iteration
        """
        if output_filename_prefix is None:
            return []

        prefix_path = Path(output_filename_prefix)

        old_state_files = [str(f) for f in prefix_path.parent.glob(prefix_path.name + ".iter_*.state.json")]
        old_state_files = sorted(old_state_files, key = lambda filename: NS._iter_from_state_file(filename))

        return old_state_files


    def init_rngs(self, random_seed=None, bit_generator_states=None, different_nlocal=False):
        """Initialize rngs from previous state in a dict, or from a seed

        Parameters
        ----------
        random_seed: int, default None
            seed to use (if state is not provided)
        bit_generator_states: dict, default None
            dict with saved rng states ``{"global": dict,  "locals": [dict, ...]}``,
            overrides ``random_seed``
        different_nlocal: bool, default False
            if bit_generator_states is provided, use it as much as possible even if
            it provides a mismatched number of local bit_generators
        """
        # construct rng objects
        self.rng_global, self.rng_local = new_rngs(self.comm.rank, random_seed)

        # override state if provided
        if bit_generator_states is not None:
            self.read_rngs(bit_generator_states, different_nlocal=different_nlocal)


    def init_configs(self, params_configs, configs_file=None, extra=False):
        """initialize all configurations by reading or constructing objects on head node and
        sending them to each compute node

        Sets self.snapshot_iter > 0 if a snapshot was read

        Parameters
        ----------
        params_configs: dict
            dictionary of parameters for configurations created by NSConfig method
        extra: bool, default False
            construct an extra config to use for pilot walks
        """

        if configs_file is None:
            assert self.snapshot_iter < 0

        # define generators for new configs from file or randomly generated
        new_configs_generator = self.NSConfig.new_configs_generator(self.n_configs_global,
                params_configs, self.rng_global, configs_file)

        # generate on root, send to each node
        self.local_configs = []
        if self.comm.rank == 0:
            for config_i, new_config in enumerate(new_configs_generator()):
                if config_i >= self.n_configs_global:
                    raise RuntimeError(f"Got too many configs (expected {self.n_configs_global}) from new config generator {new_configs_generator}")

                target_rank = config_i // self.max_n_configs_local
                if target_rank == self.comm.rank:
                    self.local_configs.append(new_config)
                else:
                    self.comm.send(new_config, target_rank, tag=15 + config_i % self.max_n_configs_local)

            if config_i + 1 != self.n_configs_global:
                raise RuntimeError(f"Got not enough configs ({config_i + 1}) from config generator, expected {self.n_configs_global}")
        else:
            for config_i in range(self.n_configs_local):
                self.local_configs.append(self.comm.recv(source=0, tag=15 + config_i))

        # prepare all configs for NS simulation
        for local_config in self.local_configs:
            local_config.prepare()
            local_config.init_calculator()

        # NOTE: this really belongs with the class, not the individual config
        # need to move initialization of the Zs to a classmethod
        n_quantities = self.local_configs[0].n_quantities

        self._allgatherv_counts = [self.max_n_configs_local * n_quantities] * self.comm.size
        self._allgatherv_displs = [i * (self.max_n_configs_local * n_quantities) for i in range(self.comm.size)]

        self._ns_quants_global = np.finfo(np.float64).min * np.ones((self.comm.size, self.max_n_configs_local, n_quantities))

        if extra:
            # construct extra config to use as a buffer, contents don't matter
            self.extra_config = self.NSConfig(params_configs, rng=None,
                                              compression=self.n_configs_global / (self.n_configs_global + 1),
                                              allocate_only=True)

        # re-sync after root process used rng_global to generate configs
        self.rng_global.bit_generator.state = self.comm.bcast(self.rng_global.bit_generator.state, root = 0)


    def find_max(self):
        """Find maximum of nested sampling quantity, as well as its location (parallel process rank and
        local index), and other (system specific) quantities 

        Stores in self.rank_of_max, self.local_ind_of_max, self.max_val
        """
        # gather all quantities so all nodes can do the same maximization
        # NOTE: if bandwidth is limiting, rather than latency (which seems unlikely),
        # might be faster to gather only NS quantity, find max location, then bcast all
        # other quantities from that location
        self._ns_quants_global[self.comm.rank][:self.n_configs_local][:] = [config.ns_quantities() for config in self.local_configs]
        self.comm.Allgatherv(self.MPI.IN_PLACE, [self._ns_quants_global, self._allgatherv_counts, self._allgatherv_displs, self.MPI.DOUBLE])

        # find max of NS quantity (index 0 in each config's quantities) with pure numpy ops

        # local index of max value on each proc
        local_ind_of_max_in_each_proc = np.argmax(self._ns_quants_global[:, :, 0], axis=1)
        # max value on each proc
        val_of_max_in_each_proc = np.max(self._ns_quants_global[:, :, 0], axis=1)

        # rank of proc that has global max
        self.rank_of_max = np.argmax(val_of_max_in_each_proc)
        # index in proc that has global max
        self.local_ind_of_max = local_ind_of_max_in_each_proc[self.rank_of_max]
        # global max value
        self.max_val = val_of_max_in_each_proc[self.rank_of_max]
        # other quantities of max value config
        self.max_quants = self._ns_quants_global[self.rank_of_max, self.local_ind_of_max, 1:]


    def global_ind(self, rank, local_ind):
        """global index corresponding to a rank and local index

        Parameters
        ----------
        rank: int
            rank of process
        local_ind: int
            local index in rank

        Returns
        -------
        global_ind: global index
        """
        return rank * self.max_n_configs_local + local_ind


    def local_ind(self, global_ind):
        """rank and local index corresponding to a global index

        Parameters
        ----------
        global_ind: int
            global index

        Returns
        --------
        rank, local_ind: rank and local index
        """
        
        return global_ind // self.max_n_configs_local, global_ind % self.max_n_configs_local


    def step_size_tune(self, n_configs=1, min_accept_rate=0.25, max_accept_rate=0.5, adjust_factor=1.25):
        """tune step sizes with pilot walks

        Parameters
        ----------
        n_configs: int, default 1
            number of configs to walk
        min_accept_rate: float, default 0.25
            minimum allowed acceptance rate
        max_accept_rate: float, default 0.5
            maximum allowed acceptance rate
        adjust_factor: float, default 1.25
            factor to adjust step size by
        """
        max_step_size = list(self.local_configs[0].max_step_size.values())
        step_size = [v / m for v, m in zip(self.local_configs[0].step_size.values(), max_step_size)]
        step_size_names = list(self.local_configs[0].step_size.keys())

        n_params = len(step_size)

        last_too_small = [False] * n_params
        last_too_big = [False] * n_params

        # save data from local_configs[0], which will be used for all pilot walks
        local_configs_0_data = self.local_configs[0].backup()

        first_iter = True

        while True:
            accept_freq = np.zeros((n_params, 2))
            for ns_config_i in range(n_configs):
                ns_config = self.local_configs[ns_config_i]
                if ns_config_i == 0:
                    self.local_configs[0].restore(local_configs_0_data)
                else:
                    self.local_configs[0].copy_contents(ns_config)

                self.local_configs[0].reset_walk_counters()
                accept_freq_contribution = self.local_configs[0].walk(self.max_val, self.local_walk_length, self.rng_local)
                accept_freq += np.asarray(accept_freq_contribution)

            accept_freq = self.comm.allreduce(accept_freq, self.MPI.SUM)

            if first_iter and self.comm.rank == 0:
                for name, size, max_size, freq in zip(step_size_names, self.local_configs[0].step_size.values(), max_step_size, accept_freq):
                    print("step_size_tune initial", name, "size", size, "max", max_size, "freq", freq)
                first_iter = False

            done = []
            for param_i in range(n_params):
                if accept_freq[param_i][0] > 0:
                    accept_rate_i = accept_freq[param_i, 1] / accept_freq[param_i, 0]
                    # only adjust if some steps were attempted
                    step_size[param_i], done_i, last_too_small[param_i], last_too_big[param_i] = self._tune_from_accept_rate(
                            step_size[param_i], last_too_small[param_i], last_too_big[param_i], accept_rate_i,
                            min_accept_rate, max_accept_rate, adjust_factor)
                else:
                    done_i = True

                done.append(done_i)

            # set actual step sizes by rescaling by maximum
            new_step_size = {k: v * m for k, v, m in zip(step_size_names, step_size, max_step_size)}
            for ns_config in self.local_configs:
                ns_config.step_size = new_step_size

            # if self.comm.rank == 0:
                # print("step_size_tune done", list(zip(done, accept_freq, step_size)))

            if all(done):
                break

            if any(np.asarray(step_size) < 1.0e-12):
                raise RuntimeError(f"Stepsize got too small with automatic tuning {step_size} {step_size_names}")

        if self.comm.rank == 0:
            for name, size in zip(step_size_names, self.local_configs[0].step_size.values()):
                print("Final step_size_tune", name, size)

        # restore to original config
        self.local_configs[0].restore(local_configs_0_data)


    def _tune_from_accept_rate(self, step_size, last_too_small, last_too_big, accept_rate,
                               min_accept_rate, max_accept_rate, adjust_factor):
        """Adjust the step size based on the acceptance rate of the current step, and
        what happened in previous steps

        Parameters
        ----------
        step_size: float
            current step size
        last_too_small: bool
            last time step was too small
        last_too_big: bool
            last time step was too big
        accept_rate: float
            acceptance rate for most recent walk
        min_accept_rate: float
            minimum acceptance rate to aim for
        max_accept_rate: float
            maximum acceptance rate to aim for
        adjust_factor: float
            factor to multiply/divide by

        Returns
        -------
        step_size, done, too_small, too_big: int, bool, bool, bool
            new step size, whether adjustment is done, and whether it was too small or too big this time
        """
        if accept_rate < min_accept_rate:
            # print("BOB accept_rate too low")
            # accepting too few, step_size must be too big
            if last_too_small:
                # print("BOB last step was too small, interpolating")
                # oscillating, interpolate and declare done
                step_size = 0.5 * (step_size + step_size / adjust_factor)
                # print("BOB returning", step_size)
                return step_size, True, False, False
            step_size /= adjust_factor
            last_too_big = True
            # print("BOB returning", step_size)
        elif accept_rate > max_accept_rate:
            # print("BOB accept_rate too high")
            # accepting too many, step_size must be too small
            if step_size >= 1.0:
                # print("BOB already maxed out, clipping")
                # already too big, clip and declare done
                step_size = 1.0
                return step_size, True, False, False
            if last_too_big:
                # print("BOB last step was too big, interpolating")
                # oscillating, interpolate and declare done
                step_size = 0.5 * (step_size + step_size * adjust_factor)
                # print("BOB returning", step_size)
                return step_size, True, False, False
            step_size *= adjust_factor
            if step_size > 1.0:
                # print("BOB larger step maxed out, clipping")
                step_size = 1.0
            last_too_small = True
            # print("BOB returning", step_size)
        else:
            # print("BOB rate OK, returning")
            # rate is OK, declare done
            return step_size, True, False, False

        # print("BOB post adjust returning")
        return step_size, False, last_too_small, last_too_big


    def snapshot(self, loop_iter, output_filename_prefix, save_old=2):
        """write a snapshot of the NS system

        Parameters
        ----------
        loop_iter: int
            iteration of NS loop
        output_filename_prefix: str
            initial part of filenames that will be written to
        save_old: int, default 2
            number of old snapshots to save
        """
        if self.comm.rank == 0:
            old_state_files = NS._old_state_files(output_filename_prefix)
            if len(old_state_files) > save_old - 1:
                old_state_files = old_state_files[:-(save_old-1)]
            else:
                old_state_files = []

        output_filename_prefix_iter = f"{output_filename_prefix}.iter_{loop_iter}"

        config_suffix = self.NSConfig.filename_suffix

        # save snapshots
        if self.comm.rank == 0:
            with open(output_filename_prefix_iter + f".configs{config_suffix}", "w")  as fout:
                # write own data
                for local_config in self.local_configs:
                    local_config.write(fout, extra_info={"from_rank": 0}, full_state=True)

                # receive from each
                for remote_config_i in range(self.n_configs_local, self.n_configs_global):
                    source_rank = remote_config_i // self.max_n_configs_local
                    self.extra_config.recv(source_rank, self.comm, self.MPI)
                    self.extra_config.write(fout, extra_info={"from_rank": source_rank}, full_state=True)
        else:
            for local_config in self.local_configs:
                local_config.send(0, self.comm, self.MPI)

        state = {}
        state["rngs"] = self.write_rngs()
        with open(output_filename_prefix_iter + ".state.json", "w") as fout:
            json.dump(state, fout)

        # make sure writes did something
        if self.comm.rank == 0:
            if Path(output_filename_prefix_iter + f".configs{config_suffix}").stat().st_size <= 0:
                raise RuntimeError(f"Failed to write to snapshot {output_filename_prefix_iter}.configs{config_suffix} file, refusing to continue")
            if Path(output_filename_prefix_iter + ".state.json").stat().st_size <= 0:
                raise RuntimeError("Failed to write to snapshot {output_filename_prefix_iter}.state.json file, refusing to continue")

        # wipe older snapshots
        if self.comm.rank == 0 and len(old_state_files) > 0:
            for old_file in old_state_files:
                print(f"Wiping old snapshot file {old_file}")
                Path(old_file).unlink()
                Path(old_file.replace(".state.json", f".configs{config_suffix}")).unlink()


    def read_rngs(self, bit_generator_states, different_nlocal=False):
        """read rngs from a dict

        Parameters
        ----------
        bit_generator_states: dict
            dict with "global" rng state and "locals" list of rng states
        different_nlocal: bool, default False
            if True, allow rngs from a different number of parallel processes than saved
            in dict (dropping extras or generating new ones, as needed)
        """
        if self.comm.rank == 0:
            if len(bit_generator_states["locals"]) != self.comm.size and not different_nlocal:
                raise RuntimeError(f"Got local rngs states for {len(bit_generator_states['locals'])} procs, but current number is {self.comm.size}")
        else:
            bit_generator_states = {"global": None, "locals": []}

        # bcast globals
        new_state = self.comm.bcast(bit_generator_states["global"], root=0)
        self.rng_global.bit_generator.state = new_state

        # scatter or generate locals
        n_locals = self.comm.bcast(len(bit_generator_states["locals"]), root = 0)
        if n_locals == self.comm.size:
            self.rng_local.bit_generator.state = self.comm.scatter(bit_generator_states["locals"][:self.comm.size], root=0)
        else:
            # number of read-in generators doesn't match number needed, start over by generating all from global rng
            self.rng_global, self.rng_local = new_rngs(self.comm.rank, None, self.rng_global)


    def write_rngs(self):
        """Write rngs to a file
        """
        if self.comm.rank == 0:
            bit_generator_states = {"global": self.rng_global.bit_generator.state}
            bit_generator_states["locals"] = self.comm.gather(self.rng_local.bit_generator.state, root=0)
        else:
            bit_generator_states = None
            _ = self.comm.gather(self.rng_local.bit_generator.state, root=0)

        return bit_generator_states
