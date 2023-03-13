# Overview

`pymatnext` is a Python package for doing nested sampling (NS) which
was developed as a complete rewrite of [pymatnest](https://github.com/libAtoms/pymatnest).
While it is coded with some generalizability in mind, it is 
designed for atomistic materials science applications, and currently
supports single atomic configurations with the
[ASE atoms object](https://wiki.fysik.dtu.dk/ase/ase/atoms.html).

# Features

## Nested Sampling

 - Sample on potential energy $`E`$, with or without enthalpy
   term $`+ P V`$ for fixed pressure and chemical potential term $`- \sum_i \mu_i N_i`$
   for variable composition (but constant total number of atoms, i.e. semi-grand-canonical ensemble).

 - Output arbitrary sampled quantities, including NS quantity ($`E + \delta_V P V  - \delta_N \sum_i \mu_i N_i`$),
   as well as configuration-type-specific quantities such as cell volume and composition.

 - Output sampled configurations, for postprocessed computation at any temperature
   of any quantity that can be computed from the configurations.

 - Save snapshots of population and random number generator states for restarts

 - One sample culled per NS iteration

### Parallelization

 - Optionally parallelized with MPI by distributing configurations among $`N_p`$ MPI tasks
   and walking one configuration per task for a shorter trajectory length $`L' = L / N_p`$.

## Configurations

 - Periodic or nonperiodic cells containing atoms of any chemical
   species. Use an extended version of the  `ase.atoms.Atoms` class, which
   stores all important data in two `np.ndarray` objects, so that MPI
   parallelized runs can communicate them without having to pack/unpack
   the data each time.

### Energy calculators

 - ASE `ase.calculators.Calculator`, with random walks carried out by Python code within `pymatnext`.
   
 - [LAMMPS](https://www.lammps.org/) potentials with random walks carried out inside LAMMPS
   with a custom set of `fix`es.

### Random walks

Walks consist of blocks of a several steps of a single type (position or cell or species).

#### Positions

 - Positions sampled with Galilean Monte Carlo, i.e. walking in a straight line in $`3 N`$
   dimensional positions space until energy exceeds NS maximum, then reflecting specularly,
   and accepting or rejecting entire trajectory depending on on final energy.

#### Cell

 - Cell shape and size sampled with Monte Carlo moves. Volume may be sampled from a $`V^{N}`$
   prior, which is the correct one for ensemble averages, or with a flat prior, which requires
   reweighting of the sampled configurations during analysis.  

 - Non-zero pressure for total energy can not be applied for LAMMPS, only ASE Calculators (but 
   calculator may use `LAMMPSlib`, although this may be lower efficiency than the LAMMPS `fix`es).

#### Species

 - Atom species sampled with Monte Carlso swap or semi-grand-canonical (total atom
   number conserving at an applied chemical potential) moves.

# Usage

see [here](README_usage.md)

# Analysis

see [here](README_analysis.md)

# Using LAMMPS

## LAMMPS compiled with `cmake`

To use lammps, set the env var `LAMMPS_DIR` to the top of the LAMMPS source directory, above `src/` and `cmake/`, and do
```
cd $LAMMPS_DIR
patch_file=$( python3 -c 'import pymatnext; from pathlib import Path; print(str(Path(pymatnext.__file__).parent) + "/extras/lammps.patch")' )
patch -p1 < $patch_file
```

The `NS` package must then be added to enable the `fix ns/*` commands, and LAMMPS
must be recompiled *without MPI support*.  The LAMMPS python interface must then be installed.

# TODO

(in order of priority?)

 - refactor `<prefix>.NS_samples` and `<prefix>.traj.<suffix>` truncation code (minor)
 - create `ABC` for `NSConfig` (med)
 - apply pressure in LAMMPS `fix ns/cellmc` (med)
 - sample positions with TE-HMC (major)

## Done

 - restart from snapshots
 - refactor `NSConfig.n_quantities` to be a class rather than instance attribute (minor)
