/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// #define SWAP_POS

#ifdef DEBUG
#include <iostream>
#endif

#include <stdio.h>
#include <string.h>
#include "fix_ns_type.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "neighbor.h"
#include "compute.h"
#include "update.h"
#include "random_mars.h"
#include "domain.h"
#include "memory.h"
#include "respa.h"
#include "error.h"
#include "math_extra.h"
#include <math.h>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNSType::FixNSType(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // args [fix ID group] seed(i) Emax(d) semi_GC_flag(b) mu_1 ...
  if (strcmp(style,"ns/type") != 0 && narg != 6 && narg != 6 + atom->ntypes)
    error->all(FLERR,"Illegal number of args in fix ns/type command");

  // copied from fix_gmc.cpp
  dynamic_group_allow = 1;
  time_integrate = 1;

  // copied from MC/fix_widom.cpp
  // vector of attempt/success counts
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  // parse args
  int iarg=3;
  seed = utils::inumeric(FLERR,arg[iarg++],true,lmp);
  Emax = utils::numeric(FLERR,arg[iarg++],true,lmp);

  semi_GC_flag = 1;
  if (strcmp(arg[iarg],"no") == 0) semi_GC_flag = 0;
  else if (strcmp(arg[iarg],"yes") != 0)
    error->all(FLERR,"Illegal fix ns/type semi_GC_flag value");
  iarg++;

  if (semi_GC_flag) {
    if (narg != 6 + atom->ntypes)
      error->all(FLERR,"semi_GC_flag requires ntypes mu values");
    mu = new double[atom->ntypes];
    for (int i=0; i < atom->ntypes; i++) {
      mu[i] = utils::numeric(FLERR,arg[iarg++],true,lmp);
    }
  } else {
    mu = 0;
  }

  // random_g will be used to pick one process's atoms to be perturbed, so
  // every process has the same seed, so that all processes agree on which
  // process was picked.
  // random_l will be used only on process that contains that atom, to
  // pick details of perturbation, and it may be called an unpredictable number of
  // times, which will make it diverge between the different processes.
  random_g = new RanMars(lmp,seed);
  random_l = new RanMars(lmp,seed + 1 + comm->me);
  memory->create(prevtype,atom->nmax,"ns/type:prevtype");
#ifdef SWAP_POS
  memory->create(prevx,atom->nmax,3,"ns/type:prevx");
#endif

  n_attempt = n_success = 0;

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
}

/* ---------------------------------------------------------------------- */

int FixNSType::setmask()
{
  // from gmc
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= PRE_EXCHANGE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNSType::init()
{
  // from gmc
  int id = modify->find_compute("thermo_pe");

  modify->compute[id]->invoked_scalar = -1;
  pe_compute = modify->compute[id];
  pe_compute->addstep(update->ntimestep+1);

  if (strstr(update->integrate_style,"respa"))
    error->all(FLERR,"fix ns/type not compatible with RESPA");

  // based on fix_atom_swap.cpp
  int ntypes = atom->ntypes;
  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int itype = 1; itype <= ntypes; itype++)
    for (int jtype = 1; jtype <= ntypes; jtype++)
      for (int ktype = 1; ktype <= ntypes; ktype++)
        if (cutsq[itype][ktype] != cutsq[jtype][ktype])
          unequal_cutoffs = true;

  // need to accumulate dmuN relative to run-initial composition, since
  // Emax is shifted based on that initial, fixed composition
  cumulative_dmuN = 0.0;

}

void FixNSType::pre_exchange()
{
  // NOTE: need to check if this is really needed, since unlike atom_swap.cpp
  // remainder of actual normal sequence will run before energy is calculated
  // Also, presumable need to do this again if step is rejected, but not sure where
  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  }

  next_reneighbor = update->ntimestep + 1;
}

void FixNSType::initial_integrate(int vflag)
{
  // do move (unless rejected)

  n_attempt++;

  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;

  int *type = atom->type;
  for (int iat=0; iat < nlocal; iat++)
      prevtype[iat] = type[iat];

#ifdef SWAP_POS
  double **x = atom->x;
  for (int iat=0; iat < nlocal; iat++)
      for (int jat=0; jat < 3; jat++)
          prevx[iat][jat] = x[iat][jat];
#endif

  double rv = random_g->uniform();
  int rnd_proc = static_cast<int>(comm->nprocs * rv) % comm->nprocs;
  if (comm->me != rnd_proc)
    return;

  rv = random_l->uniform();
  int atom_0 = static_cast<int>(nlocal * rv) % nlocal;

  if (semi_GC_flag) {
    // perturb type randomly, store change to mu*N term
    rv = random_l->uniform();
    int dtype = 1 + static_cast<int>(rv * (ntypes-1)) % (ntypes-1);
    int new_type = 1 + (((type[atom_0]-1) + dtype) % ntypes);
    dmuN = mu[new_type-1] - mu[type[atom_0]-1];
    type[atom_0] = new_type;
#ifdef DEBUG
    std::cout << "SEMI-GC " << atom_0 << " t " << prevtype[atom_0] << " mu " << mu[prevtype[atom_0]-1] <<  " -> " <<
                                                  type[atom_0] << " mu " << mu[type[atom_0]-1] << " ";
#endif
  } else {
    bool all_same = true;
    for (int i=1; i < nlocal; i++)
      if (type[0] != type[i]) {
        all_same = false;
        break;
      }
    if (all_same)
      return;

    int atom_1 = atom_0;
    while (type[atom_0] == type[atom_1]) {
        rv = random_l->uniform();
        atom_1 = static_cast<int>(nlocal * rv) % nlocal;
    }

#ifdef DEBUG
#ifdef SWAP_POS
    std::cout << "SWAP i " << atom_0 << " type " << type[atom_0] << " x " << x[atom_0][0] << " " << x[atom_0][1] << " " << x[atom_0][2] << " <-> " <<
                     " i " << atom_1 << " type " << type[atom_1] << " x " << x[atom_1][0] << " " << x[atom_1][1] << " " << x[atom_1][2] << " ";
#else
    std::cout << "SWAP i " << atom_0 << " type " << type[atom_0] << " <-> i " << atom_1 << " type " << type[atom_1] << " ";
#endif
#endif

#ifdef SWAP_POS
    double tx[3];
    tx[0] = x[atom_0][0];
    tx[1] = x[atom_0][1];
    tx[2] = x[atom_0][2];
    x[atom_0][0] = x[atom_1][0];
    x[atom_0][1] = x[atom_1][1];
    x[atom_0][2] = x[atom_1][2];
    x[atom_1][0] = tx[0];
    x[atom_1][1] = tx[1];
    x[atom_1][2] = tx[2];
#else
    int t_type = type[atom_0];
    type[atom_0] = type[atom_1];
    type[atom_1] = t_type;
#endif

    dmuN = 0.0;
  }

  // copied from fix_gmc.cpp
  // not sure if it's needed on rejections
  int id = modify->find_compute("thermo_pe");

  modify->compute[id]->invoked_scalar = -1;
  pe_compute = modify->compute[id];
  pe_compute->addstep(update->ntimestep+1);
}

/* ---------------------------------------------------------------------- */

void FixNSType::final_integrate()
{
  double ecurrent = modify->compute[modify->find_compute("thermo_pe")]->compute_scalar();

  // if potential energy + d(mu N) is above Emax then reject move
  // need to include previous steps' cumulative dmuN contributions, as well as current ones
  if (ecurrent - cumulative_dmuN - dmuN >= Emax) {
#ifdef DEBUG
    std::cout << "REJECT E == " << ecurrent << " - " << cumulative_dmuN + dmuN << " >= Emax == " << Emax << " " << std::endl;
#endif
    // reject move, so don't touch cumulative_dmuN, since type change that led to current dmuN was reverted

    int nlocal = atom->nlocal;
    int *type = atom->type;
// With SWAP_POS, type is only modified for semi-GC, so only has to be undone in that case, and x
// has to be reverted to undo swap move.
// By default (no SWAP_POS), type is modified for both swap and semi-GC, so undoing either step type
// uses prevtype
#ifdef SWAP_POS
    if (semi_GC_flag) {
#endif
        for (int iat=0; iat < nlocal; iat++)
            type[iat] = prevtype[iat];
#ifdef SWAP_POS
    } else {
        double **x = atom->x;
        for (int iat=0; iat < nlocal; iat++)
            for (int jat=0; jat < 3; jat++)
                x[iat][jat] = prevx[iat][jat];
    }
#endif

    // NOTE: no idea if this is necessary, or the right place to do this after rejecting a move
    // I suspect it's not necessary at all, since next step all these things will be done anyway
    if (unequal_cutoffs) {
      if (domain->triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      comm->exchange();
      comm->borders();
      if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      if (modify->n_pre_neighbor) modify->pre_neighbor();
      neighbor->build(1);
    }
  } else {
    // accept move, so accumulate dmuN contribution from this step
    cumulative_dmuN += dmuN;
    n_success++;
#ifdef DEBUG
    std::cout << "ACCEPT E == " << ecurrent << " + " << cumulative_dmuN << " < Emax == " << Emax << std::endl;
#endif
  }
}

FixNSType::~FixNSType()
{

  // delete temperature and pressure if fix created them
  delete random_g;
  delete random_l;
  if (mu) {
    delete[] mu;
  }

  memory->destroy(prevtype);
#ifdef SWAP_POS
  memory->destroy(prevx);
#endif

}

/* ----------------------------------------------------------------------
  return acceptance numbers
------------------------------------------------------------------------- */

double FixNSType::compute_vector(int n)
{
  switch (n) {
      case 0: return n_attempt;
      case 1: return n_success;
  }
  return -1.0;
}
