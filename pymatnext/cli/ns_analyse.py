#!/usr/bin/env python3

import argparse
import logging

import sys
import math
import re
import numpy as np
import json
from tqdm import tqdm

from pymatnext.analysis import utils

try:
    import ase.units
    GPa = ase.units.GPa
except ModuleNotFoundError:
    GPa = None

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None

try:
    from matplotlib.figure import Figure
except ModuleNotFoundError:
    Figure = None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--Tmin', '-M',  help="""Minimum temperature""",type=float,required=True)
    p.add_argument('--dT', '-D',  help="""Temperature step""",type=float,required=True)
    p.add_argument('--nT', '-n',  help="""Number of temperatures""",type=int,required=True)
    p.add_argument('--kB', '-k',  help="""Boltzmann constant (defaults to eV/K)""", type=float, default=8.6173324e-5)
    p.add_argument('--accurate_sum', action='store_true', help="""use more accurate sum (math.fsum)""")
    p.add_argument('--verbose', '-v', action='store_true', help="""Verbose output (for debugging)""")
    p.add_argument('--line_skip', '-s',  help="""number of lines to skip""", type=int, default=0)
    p.add_argument('--line_end', '-l',  help="""line to ened analysis (python std. zero based last line + 1)""", type=int, default=None)
    p.add_argument('--interval', '-i',  help="""interval between lines to use""", type=int, default=1)
    pressure_g = p.add_mutually_exclusive_group()
    pressure_g.add_argument('--delta_P_GPa', '-P',  help="""delta pressure to use for reweighting (works best with flat V prior) in GPa""", type=float)
    pressure_g.add_argument('--delta_P',  help="""delta pressure to use for reweighting (works best with flat V prior)""", type=float)
    p.add_argument('--entropy', '-S', action='store_true', help="""compute and print entropy (relative to entropy of lowest T structure""")
    p.add_argument('--probability_entropy_minimum', type=float, help="""probability entropy mininum that indicates a problem with sampling""", default=5.0)
    p.add_argument('--plot', '-p', nargs='*', help="""column names to plot, or optionally 'log(colname)'. """
                                                   """If no column names provided, list allowed names and abort""")
    p.add_argument('--plot_together', help="""output filename for combined plot""")
    p.add_argument('--plot_together_filenames', action='store_true', help="""show filenames in combined plot""")
    p.add_argument('--plot_twinx_spacing', type=float, help="""spacing for extra twinx y axes""", default=0.15)
    p.add_argument('--quiet', '-q', action='store_true', help="""No progress output""")
    p.add_argument('infile', nargs='+', help="""input energies file, or old analysis files for replotting only (all actual analysis flags will be ignore)""")

    args = p.parse_args()

    if args.delta_P_GPa is not None:
        args.delta_P = args.delta_P_GPa * GPa

    if args.plot is not None:
        if len(args.plot) == 0:
            # don't actually do analysis, just return possible column names for plot
            args.nT = 1
        else:
            args.plot = " ".join(args.plot).split()

    if args.accurate_sum:
        sum_f = math.fsum
    else:
        sum_f = np.sum

    if MPI is not None:
        comm_rank = MPI.COMM_WORLD.Get_rank()
        comm_size = MPI.COMM_WORLD.Get_size()
        if comm_rank == 0:
            logging.warning(f'Using mpi nprocs={comm_size}\n')
    else:
        comm_rank = 0
        comm_size = 1

    linestyles = ['-', '--', '-.']

    if args.plot_together:
        fig = Figure()
    ax = {}

    def colname(colname_str):
        m = re.match(r'log\(([^)]*)\)$', colname_str)
        if m:
            return m.group(1)
        else:
            return colname_str


    # warn about P = 0
    if comm_rank == 0:
        if args.delta_P is None or args.delta_P == 0.0 or args.delta_P_GPa is None or args.delta_P_GPa == 0.0:
            logging.warning("Analysis at P=0 with variable cell is ill defined, and we got --delta_P None or 0.0, "
                          "so be careful if run had cell moves and _sampling_ P was 0.0")

    for infile_i, infile in enumerate(args.infile):
        iters = []
        Es = []
        vals = []
        analysis_header = None
        with open(infile) as fin:
            # find header, either real data or previous analysis
            for line in fin:
                if line.startswith('#'):
                    try:
                        header = json.loads(line[1:])
                        if isinstance(header, list):
                            analysis_header = header
                            header = None
                            n_walkers = "UNKNOWN"
                            n_cull = "UNKNOWN"
                            break
                        elif isinstance(header, dict):
                            break
                        # ignore other lines that happens to be json-parsable
                    except json.decoder.JSONDecodeError:
                        pass
                else:
                    # past # header lines
                    break

            if header is None and analysis_header is None:
                raise ValueError(f"File {infile} appears to have no json-parseable header")

            if analysis_header is None:
                # read real sampled data
                for l_i, line in enumerate(fin):
                    if args.line_end is not None and l_i >= args.line_end:
                        break
                    if l_i < args.line_skip or l_i % args.interval != 0:
                        continue
                    f = line.split()
                    ## should we ignore certain kinds of malformed lines?
                    ## try:
                    it = int(f[0])
                    vals_line = [float(v) for v in f[1:]]

                    if l_i == 0 and len(vals_line) != 1 + len(header["extras"]):
                        raise ValueError(f"Expecting 1 + {len(header['extras'])} extra fields, but got {len(vals_line)} on first line, refusing to continue")

                    iters.append(it)
                    Es.append(vals_line[0])
                    vals.append(vals_line[1:])
                    ## except:
                        ## pass

        if analysis_header is None:

            iters = np.asarray(iters)
            Es = np.asarray(Es)
            vals = np.asarray(vals)

            # pointer to natoms
            try:
                natoms_ind = header['extras'].index('natoms')
                natoms = vals[:, natoms_ind]
            except (KeyError, ValueError):
                natoms = None
            # pull out Vs
            try:
                vol_ind = header['extras'].index('volume')
                Vs = vals[:, vol_ind]
                inds = list(range(vals.shape[1]))
                del inds[vol_ind]
                vals = vals[:, inds]
                header['extras'].remove('volume')
            except (KeyError, ValueError):
                Vs = None

            # make into list of ndarrays, each of shape (Nsamples,)
            vals = list(vals.T)

            if args.delta_P is not None and args.delta_P != 0.0:
                if Vs is None:
                    raise RuntimeError('--delta_P != 0 requires volumes')
                Es += args.delta_P*Vs

            E_min = Es[-1]
            Es -= E_min

            # main
            n_walkers = header['n_walkers']
            n_cull = header.get('n_cull', 1)
            log_a = utils.calc_log_a(iters, n_walkers, n_cull)

            flat_V_prior = False
            if Vs is not None:
                flat_V_prior = header.get('flat_V_prior', True)

            item_keys = None
            data = []
            if args.quiet or comm_rank != 0:
                iterator = range(comm_rank, args.nT, comm_size)
            else:
                iterator = tqdm(range(comm_rank, args.nT, comm_size))
            for i_T in iterator:
                T = args.Tmin + i_T * args.dT
                results_dict = utils.analyse_T(T, Es, E_min, Vs, vals, log_a, flat_V_prior, natoms,
                                               args.kB, header.get('n_extra_DOF_per_atom', 3),
                                               p_entropy_min=args.probability_entropy_minimum,
                                               sum_f=sum_f)
                if item_keys is None:
                    item_keys = list(results_dict.keys())
                    try:
                        extras_ind = item_keys.index('extra_vals')
                        item_keys = item_keys[:extras_ind] + header['extras'] + item_keys[extras_ind+1:]
                    except ValueError:
                        pass

                    if comm_rank == 0 and args.plot is not None:
                        if len(args.plot) == 0:
                            raise ValueError(f"--plot must include column names from {item_keys}")
                        if not all([colname(pfield) in item_keys for pfield in args.plot]):
                            raise ValueError(f'--plot contains unknown fields {set(args.plot) - set(item_keys)}, must be in {item_keys}')

                results_list = list(results_dict.values())
                if 'extra_vals' in results_dict:
                    results_list = results_list[:extras_ind] + results_list[extras_ind] + results_list[extras_ind+1:]
                data.append([T] + results_list)

            if not args.quiet:
                sys.stderr.write('\n')

            try:
                data = MPI.COMM_WORLD.gather(data, root = 0)
                data = [item for sublist in data for item in sublist]
            except Exception as exc:
                logging.warning(f"Exception in MPI gather '{exc}'")
                pass

        else:
            # rereading an old analysis
            item_keys = analysis_header
            T_max = args.Tmin + (args.nT - 1) * args.dT
            data = []
            with open(infile) as fin:
                for line in fin:
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split()
                    if len(fields) == 0:
                        continue
                    for f_i, f in enumerate(fields):
                        try:
                            fields[f_i] = float(f)
                        except ValueError:
                            fields[f_i] = f

                    if fields[0] >= args.Tmin and fields[0] <= T_max:
                        data.append(fields)

        ##### OUTPUT ####
        def header_col(k):
            v = formats.get(k, default_format)[0]
            if v is None:
                return k
            else:
                return v

        def str_format(fmt):
            return re.sub(r'(\.[0-9]+)?[a-z]}$', r's}', fmt).replace(':', ':>')

        # formatting
        n_T_digits = 7
        T_format = ('T', f'{{:{n_T_digits}g}}')
        T_format_s = f'{{:>{n_T_digits-2}s}}'
        default_format = (None, '{:8g}')
        formats = {'log_Z' : ('log(Z)', '{:7g}'),
                   'FG' : ('F or G', '{:11g}'),
                   'U' : ('U', '{:11g}'),
                   'Cvp' : ('Cv or Cp', '{:11g}'),
                   'S' : ('S', '{:11g}'),
                   'low_percentile_config' : ('low % i', '{:10.0f}'),
                   'mode_config' : ('mode i', '{:10.0f}'),
                   'high_percentile_config' : ('high % i', '{:10.0f}'),
                   'p_entropy' : ('ent(p)', '{:6.3f}'),
                   'V' : ('V', '{:8g}'),
                   'thermal_exp' : ('alpha', '{:9.3g}'),
                   'problem' : ('problem', '{:5s}')}

        if header is not None:
            extensive_N = header.get('extensive', None)
        else:
            extensive_N = None

        extensive_fields = ['log_Z', 'FG', 'U', 'Cvp', 'S', 'V', 'thermal_exp']
        if comm_rank == 0:
            print("# ", infile, "n_walkers", n_walkers, "n_cull", n_cull)

            header_format = '# ' + T_format_s  + ' ' + ' '.join([str_format(formats.get(k, default_format)[1]) for k in item_keys])
            line_format =          T_format[1] + ' ' + ' '.join([formats.get(k, default_format)[1] for k in item_keys])

            if args.plot:
                plot_data = {'T': [], 'valid': []}
                for k in args.plot:
                    plot_data[colname(k)] = []

            print('#', json.dumps(item_keys))
            print(header_format.format(*(['T'] + [header_col(k) for k in item_keys])))
            data = sorted(data, key = lambda x: x[0])
            for row in data:
                if extensive_N is not None:
                    # rescale extensive quantities
                    for field_i in range(len(row)):
                        if item_keys[field_i-1] in extensive_fields:
                            row[field_i] /= extensive_N
                print(line_format.format(*row))
                if args.plot:
                    plot_data['T'].append(row[0])
                    col_i = item_keys.index('problem') + 1
                    plot_data['valid'].append(row[col_i] == 'false')
                    for pfield in args.plot:
                        try:
                            col_i = item_keys.index(colname(pfield)) + 1
                        except ValueError:
                            sys.stderr.write(f'ploting field {colname(pfield)} not found in {item_keys}\n')
                            sys.exit(1)
                        plot_data[colname(pfield)].append(row[col_i])

            if args.plot:
                for k in plot_data:
                    plot_data[k] = np.asarray(plot_data[k])

            print('')
            print('')

            if args.plot:
                if not args.plot_together:
                    fig = Figure()
                    ax = {}
                for field_i, pfield in enumerate(args.plot):
                    # should this be done here?  should it be more general, e.g. eval()?
                    col_log = pfield.startswith('log')
                    pfield = colname(pfield)
                    if len(ax) == 0:
                        ax[pfield] = fig.add_subplot()
                        ax[pfield].set_xlabel('T')
                    else:
                        if pfield not in ax:
                            ax[pfield] = ax[list(ax.keys())[0]].twinx()
                            if len(ax) > 2:
                                # offset spine
                                factor = 1.0 + args.plot_twinx_spacing * (len(ax) - 2)
                                ax[pfield].spines.right.set_position(("axes", factor))

                    valid_Ts_bool = plot_data['valid']

                    def do_plot_sections(pfield, linestyle, color, label):
                        if col_log:
                            pp = ax[pfield].semilogy
                        else:
                            pp = ax[pfield].plot

                        section_start = 0
                        got_label = False
                        for T_i in range(1, len(plot_data['T'])):
                            if valid_Ts_bool[T_i] != valid_Ts_bool[section_start]:
                                # section ended on previous
                                if valid_Ts_bool[section_start]:
                                    # valid section
                                    plot_section_start = section_start
                                    plot_section_end = min(T_i - 1 + 1, len(plot_data['T']))
                                else:
                                    plot_section_start = max(section_start - 1, 0)
                                    plot_section_end = min(T_i + 1, len(plot_data['T']))
                                if not got_label and valid_Ts_bool[section_start]:
                                    use_label = label
                                    got_label = True
                                else:
                                    use_label = None
                                pp(plot_data['T'][plot_section_start:plot_section_end], plot_data[pfield][plot_section_start:plot_section_end],
                                   linestyle if valid_Ts_bool[section_start] else ':',
                                   color=color, label=use_label)
                                section_start = T_i
                        plot_section_start = max(section_start - 1, 0)
                        plot_section_end = len(plot_data['T'])
                        if not got_label:
                            use_label = label
                        pp(plot_data['T'][plot_section_start:plot_section_end], plot_data[pfield][plot_section_start:plot_section_end],
                           linestyle if valid_Ts_bool[section_start] else ':',
                           color=color, label=use_label)

                        ax[pfield].set_ylabel(header_col(pfield))

                    if args.plot_together:
                        label = header_col(pfield)

                        if args.plot_together_filenames and pfield == list(ax.keys())[0]:
                            # first field, append filenames
                            label += ' ' + infile
                        else:
                            if infile_i != 0:
                                label = None

                        do_plot_sections(pfield, linestyles[field_i % len(linestyles)], f'C{infile_i}', label)
                    else:
                        do_plot_sections(pfield, '-', f'C{field_i}', header_col(pfield))

                if not args.plot_together:
                    fig.legend()
                    fig.savefig(infile+'.analysis.pdf', bbox_inches='tight')

    if args.plot_together:
        if args.plot_together_filenames:
            fig.legend(bbox_to_anchor=(1.2, 1.15))
        else:
            fig.legend()
        fig.savefig(args.plot_together, bbox_inches='tight')


if __name__ == "__main__":
    main()
