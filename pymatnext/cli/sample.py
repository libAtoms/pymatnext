#!/usr/bin/env python3

import sys
import os
import warnings

import time
import importlib
import pprint
import itertools
import json
import traceback

from argparse import ArgumentParser

import toml
import numpy as np

from pymatnext.ns import NS
from pymatnext.params import check_fill_defaults
from pymatnext.sample_params import param_defaults

from pymatnext.loop_exit import NSLoopExit


def init_MPI():
    """initialize MPI for NS run

    Returns
    -------
    MPI, NS_comm, walker_comm: mpi4py.MPI, communicator across all NS processes, communicator
        for processors assigned to this NS walker (for parallel calculator)
    """

    # initialize MPI
    try:
        if "PYMATNEXT_NO_MPI" in os.environ:
            raise Exception("Got PYMATNEXT_NO_MPI")

        from mpi4py import MPI
        warnings.warn(f"{MPI.COMM_WORLD.rank} Using real MPI size={MPI.COMM_WORLD.size}")
        # from https://stackoverflow.com/questions/49868333/fail-fast-with-mpi4py
        def mpiabort_excepthook(type, value, traceback_obj):
            sys.stderr.write(f"{MPI.COMM_WORLD.rank} Aborting because of exception {value}\n")
            for l in traceback.format_tb(traceback_obj):
                for ll in l.splitlines():
                    sys.stderr.write(f"{MPI.COMM_WORLD.rank} {ll.rstrip()}\n")
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort()
            sys.__excepthook__(type, value, traceback_obj)
        sys.excepthook = mpiabort_excepthook
    except Exception as exc:
        warnings.warn(f"0 No MPI ({exc}), using sample_utils.MPI")
        from pymatnext.sample_utils import MPI
    NS_comm = MPI.COMM_WORLD
    walker_comm = MPI.COMM_SELF

    return MPI, NS_comm, walker_comm


def parse_args(args_list=None):
    """parse command line arguments

    Parameters
    ----------
    args_list: list(str), default None
        optional list of arguments to use instead of command line args

    Returns
    -------
    args: parsed arguments NameSpec
    """

    parser = ArgumentParser()
    parser.add_argument("--random_seed", "-s", type=int, help="random seed overriding params.global.random_seed file")
    parser.add_argument("--output_filename_postfix", "-p", help="string to append to params.global.output_filename_prefix", default="")
    parser.add_argument("--max_iter", "-i", type=int, help="max number of NS iterations, overriding params")
    parser.add_argument("--restart_diff_nproc", "-d", action="store_true", help="allow restarts to use a different number of processors that previous partial run")
    parser.add_argument("input", help="input json/yaml file")
    args = parser.parse_args(args_list)

    return args


def sample(args, MPI, NS_comm, walker_comm):
    """Do full NS sampling loop

    Parameters
    ----------
    args: argparse NameSpace
        parsed command line arguments

    MPI: mpi4py.MPI or compatible
        MPI object for MPI-call constants

    NS_comm: mpi4py.Communicator or compatible
        communicator among NS processes

    walker_comm: mpi4py.Communicator or compatible
        communicator among processes of this NS walker
    """
    # read params
    if NS_comm.rank == 0:
        with open(args.input) as fin:
            params = toml.load(fin)
    else:
        params = None
    params = NS_comm.bcast(params, root=0)
    check_fill_defaults(params, param_defaults)

    params_global = params["global"]

    # override params from CLI
    if args.random_seed is not None:
        params_global["random_seed"] = args.random_seed
    if args.max_iter is not None:
        params_global["max_iter"] = args.max_iter

    # output file prefix
    output_filename_prefix = params_global["output_filename_prefix"] + args.output_filename_postfix

    # create outer nested sampling
    ns = NS(params["ns"], NS_comm, MPI, params_global["random_seed"], params["configs"], output_filename_prefix,
            different_n_rng_local=args.restart_diff_nproc, extra_config=NS_comm.rank == 0)
    print(f"{NS_comm.rank}/{NS_comm.size} Got n_configs_local {ns.n_configs_local}")

    start_iter = ns.snapshot_iter + 1 if ns.snapshot_iter >= 0 else 0
    print(f"SNAPSHOT RESTART {start_iter}")

    # get exit conditions
    exit_cond = NSLoopExit(params["ns"]["exit_conditions"], ns)

    params_step_size_tune = params_global["step_size_tune"]

    ####################################################################################################
    # prepare for loop
    ns.find_max()

    config_suffix = ns.local_configs[0].filename_suffix

    traj_interval = params_global["traj_interval"]
    sample_interval = params_global["sample_interval"]
    snapshot_interval = params_global["snapshot_interval"]
    stdout_report_interval_s = params_global["stdout_report_interval_s"]
    step_size_tune_interval = params_step_size_tune["interval"]

    ns_file_name = f"{output_filename_prefix}.NS_samples"
    traj_file_name = f"{output_filename_prefix}.traj{config_suffix}"

    if NS_comm.rank == 0:
        if ns.snapshot_iter >= 0:
            # snapshot, truncate existing NS_samples and .traj.suffix files

            # NOTE: does this code belong here?  Maybe refactor to a function, maybe
            # move trajectory truncation into NSConfig or something?

            # truncate .NS_samples file
            f_samples = open(ns_file_name, "r+")
            # skip header
            l = f_samples.readline()
            line_i = None
            while True:
                line = f_samples.readline()
                if not line:
                    raise RuntimeError(f"Failed to find enough lines in .NS_samples file (last line {line_i}) to reach snapshot iter {ns.snapshot_iter}")

                line_i = int(line.split()[0])
                if line_i + sample_interval > ns.snapshot_iter:
                    cur_pos = f_samples.tell()
                    f_samples.truncate(cur_pos)
                    break

            f_samples.close()

            # truncate .traj.suffix file
            f_configs = open(traj_file_name, "r+")
            while True:
                try:
                    config_i = ns.NSConfig.skip(f_configs)
                except EOFError:
                    raise RuntimeError(f"Failed to find enough lines in .traj{config_suffix} file (last config {config_i}) to reach snapshot iter {ns.snapshot_iter}")

                if config_i + traj_interval > ns.snapshot_iter:
                    cur_pos = f_configs.tell()
                    f_configs.truncate(cur_pos)
                    break

            f_configs.close()

            ns_file = open(ns_file_name, "a")
            traj_file = open(traj_file_name, "a")

        else:
            # run from start, open new .NS_samples and .traj.suffix files

            ns_file = open(ns_file_name, "w")
            header_dict = { "n_walkers": ns.n_configs_global, "n_cull": 1 }
            header_dict.update(ns.local_configs[0].header_dict())
            ns_file.write("# " + " ".join(json.dumps(header_dict, indent=0).splitlines()) + "\n")

            traj_file = open(traj_file_name,  "w")

    max_iter = params_global["max_iter"]
    if max_iter > 0:
        loop_iterable = range(start_iter, max_iter)
    else:
        loop_iterable = itertools.count(start=start_iter)

    if NS_comm.rank == 0:
        print("params = ", end="")
        pprint.pprint(params, sort_dicts=False)

    time_prev_stdout_report = time.time()
    for loop_iter in loop_iterable:
        if exit_cond(ns, loop_iter):
            break

        # max info should already be set to: ns.rank_of_max, ns.local_ind_of_max, ns.max_val, ns.max_quants

        # write quantities for max config which will be culled below
        if NS_comm.rank == 0 and sample_interval > 0 and loop_iter % sample_interval == 0:
            ns_file.write(f"{loop_iter} {ns.max_val:.10f} " + " ".join([f"{quant:.10f}" for quant in ns.max_quants]) + "\n")
            ns_file.flush()

        # tune step sizes at some iteration interval
        if step_size_tune_interval > 0 and loop_iter % step_size_tune_interval == 0:
            ns.step_size_tune(n_configs=params_step_size_tune["n_configs"],
                              min_accept_rate=params_step_size_tune["min_accept_rate"],
                              max_accept_rate=params_step_size_tune["max_accept_rate"],
                              adjust_factor=params_step_size_tune["adjust_factor"])

        # pick random config as source for clone.
        global_ind_of_max = ns.global_ind(ns.rank_of_max, ns.local_ind_of_max)
        global_ind_of_clone_source = (global_ind_of_max + 1 + ns.rng_global.integers(0, ns.n_configs_global - 1)) % ns.n_configs_global
        rank_of_clone_source, local_ind_of_clone_source = ns.local_ind(global_ind_of_clone_source)

        # write max to traj file
        if traj_interval > 0 and loop_iter % traj_interval == 0:
            if NS_comm.rank == 0:
                # only head node writes
                if NS_comm.rank == ns.rank_of_max:
                    # already local
                    max_config_write = ns.local_configs[ns.local_ind_of_max]
                else:
                    # receive from correct rank
                    ns.extra_config.recv(ns.rank_of_max, ns.comm, MPI)
                    max_config_write = ns.extra_config

                max_config_write.write(traj_file, extra_info={"NS_iter": loop_iter})
                traj_file.flush()

            elif NS_comm.rank == ns.rank_of_max:
                # send to rank 0
                ns.local_configs[ns.local_ind_of_max].send(0, ns.comm, MPI)

        # do cloning locally or by send/recv pair
        if rank_of_clone_source == ns.rank_of_max and rank_of_clone_source == NS_comm.rank:
            # local copy
            ns.local_configs[ns.local_ind_of_max].copy_contents(ns.local_configs[local_ind_of_clone_source])
        elif NS_comm.rank == rank_of_clone_source:
            # send
            ns.local_configs[local_ind_of_clone_source].send(ns.rank_of_max, ns.comm, MPI)
        elif NS_comm.rank == ns.rank_of_max:
            # recv
            ns.local_configs[ns.local_ind_of_max].recv(rank_of_clone_source, ns.comm, MPI)

        # walk one per proc
        if NS_comm.rank == ns.rank_of_max:
            # always walk cloned config which is in location of old max
            i_walk = ns.local_ind_of_max
        else:
            # walk a random config
            i_walk = ns.rng_local.integers(0, ns.n_configs_local)

        ns.local_configs[i_walk].walk(ns.max_val, ns.local_walk_length, ns.rng_local)

        # find new maximum
        ns.find_max()

        # report to stdout every so often
        if NS_comm.rank == 0:
            ns.report_store(loop_iter)

            time_cur = time.time()
            if (stdout_report_interval_s > 0 and (time_cur - time_prev_stdout_report >= stdout_report_interval_s)) or loop_iter == 0:
                print(f"NS loop {loop_iter} time {time_cur-time_prev_stdout_report:4.1f} max {ns.max_val:.6f} {ns.report()}")
                time_prev_stdout_report = time_cur
                sys.stdout.flush()

        # NOTE: should this be a time rather than iteration interval?  That'd basically be straightforward,
        # except it would require an additional communication so all processes agree that it's time for a snapshot
        if loop_iter > 0 and snapshot_interval > 0 and loop_iter % snapshot_interval == 0:
            ns.snapshot(loop_iter, output_filename_prefix)

        loop_iter += 1


def main(args_list=None, mpi_finalize=True):
    """do NS sampling

    Parameters
    ----------
    args_list: list(str), default None
        list of arguments overriding command line args

    mpi_finalize: bool, default True
        call MPI.Finalize(), pass False if additional MPI things will be done afterwards
    """
    MPI, NS_comm, walker_comm = init_MPI()

    if MPI.COMM_WORLD.rank == 0:
        args = parse_args(args_list=args_list)
    else:
        args = None

    args = MPI.COMM_WORLD.bcast(args, root=0)

    sample(args, MPI, NS_comm, walker_comm)

    if mpi_finalize:
        MPI.Finalize()


if __name__ == "__main__":
    main()
