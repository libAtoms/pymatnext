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

#ifdef DEBUG
#include <iostream>
#endif

#include <stdio.h>
#include <string.h>
#include "fix_ns_cellmc.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
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

#define MOVE_UNDEF -1
#define MOVE_VOL 0
#define MOVE_STRETCH 1
#define MOVE_SHEAR 2

/* ---------------------------------------------------------------------- */

FixNSCellMC::FixNSCellMC(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // args [fix ID group] 3..12 (4 walk, 6 step, 1? flat_V_prior)
  // seed(i) Emax(d) min_aspect_ratio(d) pressure(d) pVol(d) dVol(d) pStretch(d) dStretch(d) pShear(d) dShear(d) [flat_V_prior(y/n)]
  if (strcmp(style,"ns/cellmc") != 0 && narg != 3 + 4 + 6 && narg != 3 + 4 + 6 + 1)
    error->all(FLERR,"Illegal number of args in fix ns/cellmc command");

  if (!domain->triclinic)
    error->all(FLERR,"fix ns/cellmc requires triclinic box (for now)");

  // copied from fix_gmc.cpp
  dynamic_group_allow = 1;
  time_integrate = 1;

  // copied from MC/fix_widom.cpp
  // vector of attempt/success counts
  vector_flag = 1;
  size_vector = 6;
  global_freq = 1;
  extvector = 0;

  // parse args
  int iarg=3;
  seed = utils::inumeric(FLERR,arg[iarg++],true,lmp);
  Emax = utils::numeric(FLERR,arg[iarg++],true,lmp);
  min_aspect_ratio = utils::numeric(FLERR,arg[iarg++],true,lmp);
  pressure = utils::numeric(FLERR,arg[iarg++],true,lmp);

  pVol = utils::numeric(FLERR,arg[iarg++],true, lmp);
  dVol = utils::numeric(FLERR,arg[iarg++],true, lmp);
  pStretch = utils::numeric(FLERR,arg[iarg++],true, lmp);
  dStretch = utils::numeric(FLERR,arg[iarg++],true, lmp);
  pShear = utils::numeric(FLERR,arg[iarg++],true, lmp);
  dShear = utils::numeric(FLERR,arg[iarg++],true, lmp);
  flat_V_prior = 1;
  if (narg == 3+4+6+1) {
    if (strcmp(arg[iarg],"no") == 0) flat_V_prior = 0;
    else if (strcmp(arg[iarg],"yes") != 0)
      error->all(FLERR,"Illegal fix ns/cellmc flat_V_prior value");
    iarg++;
  }

  // enforce min prob. is 0.0
  pVol = (pVol < 0.0) ? 0.0 : pVol;
  pStretch = (pStretch < 0.0) ? 0.0 : pStretch;
  pShear = (pShear < 0.0) ? 0.0 : pShear;

  // normalize (i.e. relative probbilities), 
  double pSum = pVol + pStretch + pShear;
  pVol /= pSum;
  pStretch /= pSum;
  pShear /= pSum;
  // cumulative probabilities: Vol then Stretch then Shear
  pStretch += pVol;
  pShear = 1.0; // should be "pShear += pStretch;", but ensure there's no roundoff

  // every node should have same seed, so random steps will be the same
  // copied from fix_gmc.cpp, without the "+ comm->me"
  random = new RanMars(lmp,seed);
  memory->create(prevx,atom->nmax,3,"ns/cellmc:prevx");

  last_move_type = MOVE_UNDEF;
  n_attempt_vol = n_attempt_stretch = n_attempt_shear = 0;
  n_success_vol = n_success_stretch = n_success_shear = 0;

  // from fix_nh.cpp
  // who knows what other things may be needed?
  box_change |= (BOX_CHANGE_X | BOX_CHANGE_Y | BOX_CHANGE_Z |
                 BOX_CHANGE_YZ | BOX_CHANGE_XZ | BOX_CHANGE_XY);
  no_change_box = 1;
}

/* ---------------------------------------------------------------------- */

int FixNSCellMC::setmask()
{
  // from gmc
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNSCellMC::init()
{

  // from gmc, but without having to create a random direction vector
  int id = modify->find_compute("thermo_pe");

  modify->compute[id]->invoked_scalar = -1;
  pe_compute = modify->compute[id];
  pe_compute->addstep(update->ntimestep+1);

  if (strstr(update->integrate_style,"respa"))
    error->all(FLERR,"fix ns/cellmc not compatible with RESPA");

  // need to accumulate dPV relative to run-initial volume, since
  // Emax is shifted based on that initial, fixed volume
  cumulative_dPV = 0.0;

}

void FixNSCellMC::initial_integrate(int vflag)
{
  // do move (unles rejected)

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  double **x = atom->x;

  // save previous cell and positions
  for (int i=0; i < 3; i++) {
    prev_boxlo[i] = domain->boxlo[i];
    prev_boxhi[i] = domain->boxhi[i];
  }
  prev_xy = domain->xy;
  prev_yz = domain->yz;
  prev_xz = domain->xz;

  for (int iat=0; iat < nlocal; iat++)
    for (int j=0; j < 3; j++)
        prevx[iat][j] = x[iat][j];

  double boxext[3];
  for (int i=0; i < 3; i++)
    boxext[i] = domain->boxhi[i] - domain->boxlo[i];

  last_move_type = MOVE_UNDEF;
  move_rejected_early = false;

  // keep track of change PV in this step
  // default to 0, set otherwise when volume move is selected
  dPV = 0.0;

  // default to accept, will be changed below to reject (-1.0) or probablistic (if flat_V_prior != 1)
  double rv = random->uniform();
  double new_cell[3][3];
  // pick a step type
  if (rv < pVol) {
    n_attempt_vol++;
    last_move_type = MOVE_VOL;
    // volume step
    double orig_V;
    if (domain->dimension == 3) orig_V = domain->xprd * domain->yprd * domain->zprd;
    else orig_V = domain->xprd * domain->yprd;

    double dV = random->gaussian(0.0, dVol);
    double new_V = orig_V + dV;
#ifdef DEBUG
    std::cout << "VOLUME volumetric strain " << dV << " / " << orig_V << " = " << dV/orig_V << " ";
#endif
    if (new_V/orig_V < 0.5) {
      move_rejected_early = true;
#ifdef DEBUG
      std::cout << "REJECT new_V/orig_V < 0.5" << std::endl;
#endif
    } else {
      if (flat_V_prior == 0 && new_V < orig_V && random->uniform() > pow(new_V / orig_V, natoms)) {
        move_rejected_early = true;
#ifdef DEBUG
        std::cout << "REJECT V probability " << pow(new_V / orig_V, natoms) << std::endl;
#endif
      } else {
        double transform_diag = pow(new_V / orig_V, 1.0/3.0);
        new_cell[0][0] = boxext[0] * transform_diag;
        new_cell[0][1] = 0.0;
        new_cell[0][2] = 0.0;
        new_cell[1][0] = domain->xy * transform_diag;
        new_cell[1][1] = boxext[1] * transform_diag;
        new_cell[1][2] = 0.0;
        new_cell[2][0] = domain->xz * transform_diag;
        new_cell[2][1] = domain->yz * transform_diag;
        new_cell[2][2] = boxext[2] * transform_diag;
        dPV = pressure * dV
      }
    }
  } else if (rv < pStretch) {
    n_attempt_stretch++;
    last_move_type = MOVE_STRETCH;
    // stretch step
    // pick directions to use v_ind and (v_ind+1)%3
#ifdef DEBUG
    std::cout << "STRETCH ";
#endif
    int v_ind = int(3 * random->uniform()) % 3;
    rv = random->gaussian(0.0, dStretch);
    double transform_diag[3];
    transform_diag[v_ind] = exp(rv);
    transform_diag[(v_ind+1)%3] = exp(-rv);
    transform_diag[(v_ind+2)%3] = 1.0;
#ifdef DEBUG
    std::cout << transform_diag[0] << " " << transform_diag[1] << " " << transform_diag[2] << " ";
#endif
    // create new cell for aspect ratio test and new domain
    new_cell[0][0] = boxext[0] * transform_diag[0];
    new_cell[0][1] = 0.0;
    new_cell[0][2] = 0.0;
    new_cell[1][0] = domain->xy * transform_diag[0];
    new_cell[1][1] = boxext[1] * transform_diag[1];
    new_cell[1][2] = 0.0;
    new_cell[2][0] = domain->xz * transform_diag[0];
    new_cell[2][1] = domain->yz * transform_diag[1];
    new_cell[2][2] = boxext[2] * transform_diag[2];
    if (min_aspect_ratio_val(new_cell) < min_aspect_ratio) {
      move_rejected_early = true;
#ifdef DEBUG
      std::cout << "REJECT min_aspect_ratio " << min_aspect_ratio_val(new_cell) << std::endl;
#endif
    }
  } else {
    n_attempt_shear++;
    last_move_type = MOVE_SHEAR;
    // shear step
    // save original cell and a temporary cell
#ifdef DEBUG
    std::cout << "SHEAR ";
#endif
    double orig_cell[3][3], t_cell[3][3];
    t_cell[0][0] = orig_cell[0][0] = boxext[0];
    t_cell[0][1] = orig_cell[0][1] = 0.0;
    t_cell[0][2] = orig_cell[0][2] = 0.0;
    t_cell[1][0] = orig_cell[1][0] = domain->xy;
    t_cell[1][1] = orig_cell[1][1] = boxext[1];
    t_cell[1][2] = orig_cell[1][2] = 0.0;
    t_cell[2][0] = orig_cell[2][0] = domain->xz;
    t_cell[2][1] = orig_cell[2][1] = domain->yz;
    t_cell[2][2] = orig_cell[2][2] = boxext[2];

    // pick vector to perturb
    int vec_ind = int(3 * random->uniform()) % 3;

    // perturb t_cell[vec_ind] along other two vectors
    double vhat[3];
    for (int di=1; di < 3; di++) {
      for (int i=0; i < 3; i++)
        vhat[i] = orig_cell[(vec_ind + di) % 3][i];
      MathExtra::norm3(vhat);
      rv = random->gaussian(0.0, dShear);
#ifdef DEBUG
    std::cout << (vec_ind + di) % 3 << " " << rv << " ";
#endif
      for (int i=0; i < 3; i++)
        t_cell[vec_ind][i] += rv * vhat[i];
    }

    if (min_aspect_ratio_val(t_cell) < min_aspect_ratio) {
      move_rejected_early = true;
#ifdef DEBUG
      std::cout << "REJECT min_aspect_ratio " << min_aspect_ratio_val(t_cell) << std::endl;
#endif
    } else {
      // rotate new_cell back to LAMMPS orientation

      double a0_norm = sqrt(MathExtra::dot3(t_cell[0], t_cell[0]));
      new_cell[0][0] = a0_norm;
      new_cell[0][1] = 0.0;
      new_cell[0][2] = 0.0;

      double a0_hat[3] = {t_cell[0][0], t_cell[0][1], t_cell[0][2]}; MathExtra::norm3(a0_hat);
      new_cell[1][0] = MathExtra::dot3(t_cell[1], a0_hat);
      double a0_hat_cross_a1[3]; MathExtra::cross3(a0_hat, t_cell[1], a0_hat_cross_a1);
      new_cell[1][1] = sqrt(MathExtra::dot3(a0_hat_cross_a1, a0_hat_cross_a1));
      new_cell[1][2] = 0.0;

      double a0_cross_a1_hat[3] = {a0_hat_cross_a1[0], a0_hat_cross_a1[1], a0_hat_cross_a1[2]};
      MathExtra::norm3(a0_cross_a1_hat);

      double a0_cross_a1_hat_cross_a0_hat[3];
      MathExtra::cross3(a0_cross_a1_hat, a0_hat, a0_cross_a1_hat_cross_a0_hat);
      new_cell[2][0] = MathExtra::dot3(t_cell[2], a0_hat);
      new_cell[2][1] = MathExtra::dot3(t_cell[2], a0_cross_a1_hat_cross_a0_hat);
      new_cell[2][2] = MathExtra::dot3(t_cell[2], a0_cross_a1_hat);
    }
  }

  if (!move_rejected_early) {
    // apply move to box
    double boxctr[3];
    for (int i=0; i < 3; i++)
      boxctr[i] = 0.5*(prev_boxhi[0] + prev_boxlo[0]);

    domain->x2lamda(nlocal);

    boxext[0] =  new_cell[0][0];
    domain->xy = new_cell[1][0];
    boxext[1] =  new_cell[1][1];
    domain->xz = new_cell[2][0];
    domain->yz = new_cell[2][1];
    boxext[2] =  new_cell[2][2];

    for (int i=0; i < 3; i++) {
      domain->boxlo[i] = boxctr[i] - boxext[i]/2.0;
      domain->boxhi[i] = boxctr[i] + boxext[i]/2.0;
    }

    domain->set_global_box();
    domain->set_local_box();

    domain->lamda2x(nlocal);
  }

  // copied from fix_gmc.cpp
  // not sure if it's needed on rejections
  int id = modify->find_compute("thermo_pe");

  modify->compute[id]->invoked_scalar = -1;
  pe_compute = modify->compute[id];
  pe_compute->addstep(update->ntimestep+1);
}

/* ---------------------------------------------------------------------- */

void FixNSCellMC::final_integrate()
{
  // if potential energy is above Emax then reject move
  double ecurrent = modify->compute[modify->find_compute("thermo_pe")]->compute_scalar();

  if (move_rejected_early) {
    return;
  }

  // if potential energy - d(P V) is above Emax then reject move
  // need to include previous steps' cumulative dPV contributions, as well as current ones
  if (ecurrent - cumulative_dPV - dPV >= Emax) {
#ifdef DEBUG
    std::cout << "REJECT E == " << ecurrent << " - " << cumulative_dPV + dPV << " >= Emax == " << Emax << std::endl;
#endif
    // reject move, so don't touch cumulative_dPV, since type change that led to current dPV was reverted

    for (int i=0; i < 3; i++) {
      domain->boxlo[i] = prev_boxlo[i];
      domain->boxhi[i] = prev_boxhi[i];
    }
    domain->xy = prev_xy;
    domain->yz = prev_yz;
    domain->xz = prev_xz;

    domain->set_global_box();
    domain->set_local_box();

    double **x = atom->x;
    int nlocal = atom->nlocal;
    for (int iat=0; iat < nlocal; iat++)
      for (int j=0; j < 3; j++)
        x[iat][j] = prevx[iat][j];

  } else {
    // accept move, so accumulate dPV contribution from this step
    cumulative_dPV += dPV;
    switch (last_move_type) {
        case MOVE_VOL:
            n_success_vol++;
            break;
        case MOVE_STRETCH:
            n_success_stretch++;
            break;
        case MOVE_SHEAR:
            n_success_shear++;
            break;
        default:
            error->all(FLERR,"Illegal value of last_move_type increment n_success_*");
    }
#ifdef DEBUG
double new_cell[3][3];
new_cell[0][0] = domain->boxhi[0] - domain->boxlo[0];
new_cell[0][1] = 0.0;
new_cell[0][2] = 0.0;
new_cell[1][0] = domain->xy;
new_cell[1][1] = domain->boxhi[1] - domain->boxlo[1];
new_cell[1][2] = 0.0;
new_cell[2][0] = domain->xz;
new_cell[2][1] = domain->yz;
new_cell[2][2] = domain->boxhi[2] - domain->boxlo[2];
    std::cout << "ACCEPT E == " << ecurrent " - " << cumulative_dPV << " < Emax == " << Emax << " min_aspect " << min_aspect_ratio_val(new_cell) << std::endl;
    // std::cout << "final cell " << new_cell[0][0] << " " << new_cell[0][1] << " " << new_cell[0][2] << std::endl;
    // std::cout << "           " << new_cell[1][0] << " " << new_cell[1][1] << " " << new_cell[1][2] << std::endl;
    // std::cout << "           " << new_cell[2][0] << " " << new_cell[2][1] << " " << new_cell[2][2] << std::endl;
#endif
  }
}

FixNSCellMC::~FixNSCellMC()
{

  // delete temperature and pressure if fix created them
  delete random;

  memory->destroy(prevx);

}

double FixNSCellMC::min_aspect_ratio_val(double cell[3][3]) {
    double min_val = std::numeric_limits<double>::max();
    for (int i=0; i < 3; i++) {
        double vnorm_hat[3];
        MathExtra::cross3(cell[(i+1)%3], cell[(i+2)%3], vnorm_hat);
        MathExtra::norm3(vnorm_hat);
        double val = fabs(MathExtra::dot3(vnorm_hat, cell[i]));
        min_val = (val < min_val) ? val : min_val;
    }

    double V;
    if (domain->dimension == 3) V = domain->xprd * domain->yprd * domain->zprd;
    else V = domain->xprd * domain->yprd;

    return min_val / pow(V, 1.0/3.0);
}

/* ----------------------------------------------------------------------
  return acceptance numbers
------------------------------------------------------------------------- */

double FixNSCellMC::compute_vector(int n)
{
  switch (n) {
      case 0: return n_attempt_vol;
      case 1: return n_success_vol;
      case 2: return n_attempt_stretch;
      case 3: return n_success_stretch;
      case 4: return n_attempt_shear;
      case 5: return n_success_shear;
  }
  return -1.0;
}
