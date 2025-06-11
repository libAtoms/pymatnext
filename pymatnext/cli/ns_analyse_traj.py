#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from matplotlib.figure import Figure
except ModuleNotFoundError:
    Figure = None

import sys
import math
import numpy as np
import json
from importlib import import_module
from tqdm import tqdm
import re

import ase.io
from pymatnext.analysis import utils

def main():

    analysis_modules = " ".join([mod for mod in dir(import_module('pymatnext.analysis.tools')) if not mod.startswith("__")])

    p = argparse.ArgumentParser(description="""Analyze NS trajectory
    by running arbitrary python modules that define an `analysis()`
    function.  Function must accept a (first) argument of `atoms`
    (`ase.atoms.Atoms`) and return an `np.ndarray`, which will be
    summed over the NS trajectory with weights from the NS process.
    For built-in plotting (`--plot`), it is assumed that there are two
    rows containing the x and y values of the analysis.  In this case
    the function must accept a `header` argument, and when it is `True`
    it must return a list of two strings with labels for each row.""")

    gT = p.add_mutually_exclusive_group()
    gT.add_argument('--temperature', '-T',  nargs='+', help="""temperature""",type=float)
    gTgrid = gT.add_argument_group()
    gTgrid.add_argument('--Tmin', '-M',  help="""Minimum temperature""",type=float)
    gTgrid.add_argument('--dT', '-D',  help="""Temperature step""",type=float)
    gTgrid.add_argument('--nT', '-n',  help="""Number of temperatures""",type=int)

    # run params, from energies file or manually)
    p.add_argument('--samples_file', '-E',  help="""samples file (only for # walkers, # cull, and flat V prior from header)""")
    p.add_argument('--walkers', type=int, help="""number of walkers to calculate compression factors""")
    p.add_argument('--cull', type=int, help="""number culled to calculate compression factors""")
    p.add_argument('--flat_V_prior', type=bool, help="""flat_V_prior""", default=True)

    p.add_argument('--delta_P', '-P',  help="""delta pressure to use for reweighting with flat V prior""", type=float)

    p.add_argument('--analysis', '-a', action='append', help=
                       """analysis module.  Single string consistent of module name (which defines a function """
                       """'analysis(atoms, arg, arg ...)', followed by JSON string defining args or kwargs (not both). """
                       f"""Predefined modules (in pymatnext.analysis.tools): {analysis_modules}""", required=True)
    p.add_argument('--plot', '-p', action='store_true', help="""quick and dirty plot of analysis results""")

    p.add_argument('--kB', '-k',  help="""Boltzmann constant (defaults to eV/K)""", type=float, default=8.6173324e-5)
    p.add_argument('--accurate_sum', action='store_true', help="""use more accurate sum (math.fsum)""")
    p.add_argument('--verbose', '-v', action='store_true', help="""Verbose output (for debugging)""")
    p.add_argument('--quiet', '-q', action='store_true', help="""No progress output""")
    p.add_argument('--no_cache', dest='cache', action='store_false', help="""do not read or write analysis results to cache files""")

    p.add_argument('--ns_iters', '-i',  help="""range expression for iterations to use""")
    p.add_argument('--ns_iter_field', help="""info field for NS iter #""", default="NS_iter")
    p.add_argument('--ns_E_field', help="""info field for NS energy/enthalpy""")

    p.add_argument('--output', '-o', help="""filename base for values and figures of each analysis""", required=True)
    p.add_argument('trajfile', nargs='+', help="""input trajectory files""")

    args = p.parse_args()

    if args.samples_file is None:
        if args.walkers is None or args.cull is None:
            raise RuntimeError("Need --samples_file OR --walkers and --cull")
    else:
        if args.walkers is not None or args.cull is not None:
            raise RuntimeError("Need --samples_file OR --walkers and --cull")

    if args.accurate_sum:
        sum_f = math.fsum
    else:
        sum_f = np.sum

    def traj_configs(trajfiles, ns_iters):
        ns_iters = ns_iters.split(":")
        ns_iters += [''] * (3 - len(ns_iters))
        if ns_iters[0] == '':
            min_iter = 0
        else:
            min_iter = int(ns_iters[0])
            if min_iter < 0:
                raise ValueError(f"ns_iters range expression min {min_iter} must be >= 0")
        if ns_iters[1] == '':
            max_iter = None
        else:
            max_iter = int(ns_iters[1])
            if max_iter <= 0:
                raise ValueError(f"ns_iters range expression max {max_iter} must be > 0")
        if ns_iters[2] == '':
            step_iter = 1
        else:
            step_iter = int(ns_iters[2])
            if step_iter <= 0:
                raise ValueError(f"ns_iters range expression step {step_iter} must be > 0")

        trajfile_iters = []
        next_config = []
        for trajfile in trajfiles:
            trajfile_iters.append(ase.io.iread(trajfile, ':', parallel=False))
            try:
                next_config.append(next(trajfile_iters[-1]))
            except StopIteration:
                next_config.append(None)

        last_iter = -1
        while any([c is not None for c in next_config]):
            lowest_iter_i = np.argmin([c.info[args.ns_iter_field] if c is not None else np.infty    for c in next_config])
            assert next_config[lowest_iter_i].info[args.ns_iter_field] > last_iter
            last_iter = next_config[lowest_iter_i].info[args.ns_iter_field]

            # check for configs in selected range
            if max_iter is not None and last_iter >= max_iter:
                break
            if last_iter >= min_iter and last_iter % step_iter == 0:
                yield next_config[lowest_iter_i]

            try:
                next_config[lowest_iter_i] = next(trajfile_iters[lowest_iter_i])
            except StopIteration:
                next_config[lowest_iter_i] = None

    analysis_funcs = []
    for a_i in range(len(args.analysis)):
        analysis_str = args.analysis[a_i]
        analysis_mod_json = analysis_str.split(maxsplit=1)
        try:
            analysis_func = import_module(analysis_mod_json[0]).analysis
        except ModuleNotFoundError:
            analysis_func = import_module('pymatnext.analysis.tools.' + analysis_mod_json[0]).analysis

        analysis_args = []
        analysis_kwargs = {}
        if len(analysis_mod_json) > 1:
            analysis_args = json.loads(analysis_mod_json[1])
            if isinstance(analysis_args, dict):
                analysis_kwargs = analysis_args
                analysis_args = []
        assert isinstance(analysis_args, list) and isinstance(analysis_kwargs, dict)

        analysis_funcs.append([analysis_func, analysis_args, analysis_kwargs])
        args.analysis[a_i] = analysis_mod_json[0]

    iters = []
    Es = []
    Vs = []
    natoms = []
    extra_vals = [ [] for _ in range(len(args.analysis)) ]

    # check for cached
    found_cached = []
    if args.cache:
        not_all = False
        for i in range(len(args.trajfile[0])):
            if not all([f[i] == args.trajfile[0][i] for f in args.trajfile]):
                not_all = True
                break
        if not_all:
            output_base = Path(args.trajfile[0][:i])
            output_base_dir = output_base.parent
            output_base_file = output_base.name
        else:
            output_base = Path(args.trajfile[0])
            output_base_dir = output_base.parent
            output_base_file = output_base.stem + '.'
        for analysis_i, analysis_f in enumerate(analysis_funcs):
            analysis_str = args.analysis[analysis_i]
            if len(analysis_f[1]) > 0:
                analysis_str += '_' + '_'.join([str(v) for v in analysis_f[1]])
            if len(analysis_f[2]) > 0:
                analysis_str += '_' + '_'.join([k+'_'+str(v) for k, v in analysis_f[2].items()])
            analysis_str = re.sub(r'[ ,:]+', '_', re.sub(r'[\[\]\{\}]', '', analysis_str))
            analysis_cache_file = output_base_dir / (str(output_base_file) + analysis_str + '.npy')
            analysis_f.append(analysis_cache_file)
            if analysis_cache_file.is_file():
                extra_vals[analysis_i] = np.load(str(analysis_cache_file))
                found_cached.append(analysis_i)
                sys.stderr.write(f'Loaded analysis {analysis_str} from cached file, got shape {extra_vals[analysis_i].shape}\n')

    if args.quiet:
        iterator = traj_configs(args.trajfile, args.ns_iters)
    else:
        iterator = tqdm(traj_configs(args.trajfile, args.ns_iters))
        iterator.set_description("iter <skipping>")
    for at in iterator:
        if all(at.numbers == 0):
            at.numbers = at.arrays['type']
        iters.append(at.info[args.ns_iter_field])
        if isinstance(iterator, tqdm):
            iterator.set_description(f"iter {iters[-1]}")
        if args.ns_E_field is None:
            Es.append(at.info["NS_quantities"][0])
        else:
            Es.append(at.info[args.ns_E_field])
        Vs.append(at.get_volume())
        natoms.append(len(at))
        for analysis_i, analysis_f in enumerate(analysis_funcs):
            if analysis_i in found_cached:
                continue
            v = analysis_f[0](at, *analysis_f[1], **analysis_f[2])
            extra_vals[analysis_i].append(v.tolist())
    iters = np.asarray(iters)
    Es = np.asarray(Es)
    Vs = np.asarray(Vs)
    natoms = np.asarray(natoms)
    # massage extra_vals to correct shape and order of axes
    for v_i in range(len(extra_vals)):
        extra_vals[v_i] = np.asarray(extra_vals[v_i])
        ndims = len(extra_vals[v_i].shape)
        if v_i not in found_cached:
            extra_vals[v_i] = np.transpose(extra_vals[v_i], axes=list(range(1, ndims)) + [0])
        if args.cache and not analysis_funcs[v_i][3].is_file():
            sys.stderr.write(f'Saving analysis {args.analysis[v_i]} shape {extra_vals[v_i].shape} to cached file\n')
            np.save(analysis_funcs[v_i][3], extra_vals[v_i])


    E_min = Es[-1]
    Es -= E_min

    if args.samples_file is not None:
        with open(args.samples_file) as fin:
            header = json.loads(fin.readline())
        args.walkers = header['n_walkers']
        args.cull = header['n_cull']
        args.flat_V_prior = header['flat_V_prior']

    log_a = utils.calc_log_a(iters, args.walkers, args.cull)

    if args.temperature is None:
        args.temperature = args.Tmin + args.dT * np.arange(args.nT)

    if args.plot:
        fig = {}
        ax = {}
        for a_i, a_name in enumerate(args.analysis):
            fig[a_name] = Figure()
            ax[a_name] = fig[a_name].add_subplot()
            ax[a_name].set_title(a_name)
            axis_labels = analysis_funcs[a_i][0](header=True)
            ax[a_name].set_xlabel(axis_labels[0])
            ax[a_name].set_ylabel(axis_labels[1])

    linetypes = ['-', '--', '-.']
    outfiles = {}
    for analysis in args.analysis:
        outfiles[analysis] = open(args.output + '.' + analysis + '.data', 'w')
    for T_i, T in enumerate(args.temperature):
        results_dict = utils.analyse_T(T, Es, E_min, Vs, extra_vals, log_a, args.flat_V_prior, natoms,
                                       args.kB, 0, args.delta_P is not None and args.delta_P != 0.0, sum_f=sum_f)
        for analysis, res in zip(args.analysis, results_dict['extra_vals']):
            outfiles[analysis].write(f'# T {T} analysis {analysis}\n')
            for v in res.T:
                outfiles[analysis].write(' '.join([str(vv) for vv in v]) + '\n')
            outfiles[analysis].write('\n\n')
            if args.plot:
                ax[analysis].plot(res[0], res[1], linetypes[T_i // 10], color=f'C{T_i}', label=f'T = {T} K')
    for outfile in outfiles.values():
        outfile.close()

    if args.plot:
        for fig_name, fig_obj in fig.items():
            fig_obj.legend()
            fig_obj.savefig(args.output + '.' + fig_name + '.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
