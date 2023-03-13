# Analysis

`pymatnext` comes with two analysis programs, `ns_analyse` and `ns_analyse_traj`. The first processes information in the `<prefix>.NS_samples` file,
and the second in the `<prefix>.traj.extxyz` file.

Both can detect that they are running under MPI and parallelize the analysis over the list of temperatures.

## Analyzing the `<prefix>.NS_samples` file

### Example command

```
ns_analyse -M 100 -D 10 -n 1500 --delta_P_GPa 0.3 --plot 'log(Cvp)' V -- LJ_LAMMPS.NS_samples > analysis.txt
```

Run an analysis, computing result starting at $`100~\mathrm{K}`$, every $`10~\mathrm{K}`$, up to $`100 + 10 * 1499 = 15090~\mathrm{K}`$.
Reweight the samples for a $0.3~\mathrm{GPa}$ higher pressure than the run, and plot $`\log(C_p)`$ and $`V`$ columns.
The plot will be saved to `LJ_LAMMPS.NS_samples.analysis.pdf`.

### text output

In addition to a header prefixed by `#`, the redirected standard output will contain information on
thermodynamic quantities:
  - $`T`$: temperature
  - $`\log(Z)`$: partition function
  - $`F`$ or $`G`$: free energy or enthalpy (if variable V) or grand potential (if variable composition)
  - $`U`$: internal energy
  - $`S`$: entropy [NOTE: relative to?]
  - $`C_v`$ or $`C_p`$: specific heat
  - $`V`$: cell volume
  - $`\alpha`$: thermal expansion coefficient
  - `K-S(V)` Komogorov-Smirnoff non-Gaussianity of volume distribution
  - low tail, mode, and high tail configuration numbers that contribute to this temperature
  - `frac(Z)`: fraction of partition function that is accounted for by this range of configs
  - `problem`: whether there is indication that the sampling was not long enough to converge this temperature

Additional rows that may be printed include
  - `natoms`: number of atoms in cell
  - $`x_{Z}`$: fraction of cell with atomic number $Z$.

### plots

Any column (or its natural log) can be plotted by specifying its internal name (second line of the
text output) in the `--plot` option.  Multiple samples files can be plotted together
by adding `--plot_together <output>.pdf`.  By default the different files are
plotted with no labels, one _line type_ per quantity, which is reasonable for multiple
instances of the same parameters.  If you are plotting different types of runs, you can
add `--plot_together_filenames` which will add a label with the filenames corresponding
to each color.

## Analyzing the `<prefix>.traj.extxyz` file

*NOTE: trajectory analysis is less well tested*

Command line arguments for `ns_analyse_traj` are similar to those of `ns_analyse`, with a few differences

 - `--temperature T1 [ T2 T2 ... ]` giving a list of temperatures can be specified, instead of a minimum, interval, and number of values.
 - Either a samples file must be specified by `--samples_file`, or the needed quantities from the header must be 
   passed in `--walkers`, `--cull`, and `--flat_V_prior`
 - The type of analysis must be passed in `--analysis`

### Specifying the analysis

The analysis is specified by the name of a python module that defines a `analysis()` function. This can
be the name of a predefined routine in a submodule below `pymatnext.analysis.tools`, or any other module
that can be found through the python module search path.

The `analysis()` function must take an `ase.atoms.Atoms` object as its first argument, other arbitrary
arguments, and return an `np.ndarray`.  If the `--plot` flag is passed, the array must have two rows,
one of $`x`$ values, one of $`y`$ values, and accept a `header=bool` optional argument that returns
the axis labels instead of the results.

Arbitrary arguments to the `analysis()` function can be passed as JSON-encoded string in the `--analysis "<ANALYSIS> <ARGS_JSON>"`
option.  `<ARGS_JSON>` must decode to a 2-element list, containing a list for `*args` and a dict of `**kargs`.
If `<ARGS_JSON>` is present both must be specified, but either may be empty.

### Caching results

By default `ns_analyse_traj` attempts to cache the results of the analyses, and read them back if available
instead of recalculating.
