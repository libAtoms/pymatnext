/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(ns,FixNS)

#else

#ifndef LMP_FIX_NS_H
#define LMP_FIX_NS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNS : public Fix {
 public:
  FixNS(class LAMMPS *, int, char **);
  ~FixNS() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;

 protected:
  class Compute *pe_compute;
  double **dx, **prevx;
  int *prevtype;
  double prev_boxlo[3], prev_boxhi[3], prev_xy, prev_yz, prev_xz;
  // global params
  int seed;
  double Emax;
  double pPos, pCell, pType;
  int state_cur_move;
  int state_traj_steps_remaining;
  // pos GMC params
  int pos_n_steps;
  double gmc_step_size;
  // cell MC params
  int cell_n_steps;
  double min_aspect_ratio, pressure;
  int flat_V_prior;
  double pVol, dVol, pStretch, dStretch, pShear, dShear;
  // type MC params
  int type_n_steps;
  int semi_GC_flag;
  double *mu;
  // other data
  class RanMars *random_g, *random_l;
  int peflag;
  char *id_pe;
  int max_n_steps;

  // pos GMC internal data
  int n_attempt_pos, n_success_pos;
  // cell MC internal data
  double dPV, cumulative_dPV;
  bool cell_move_rejected_early;
  int cell_cur_move;
  int n_attempt_vol, n_attempt_stretch, n_attempt_shear, n_success_vol, n_success_stretch, n_success_shear;
  // type MC internal data
  double dmuN, cumulative_dmuN;
  bool unequal_cutoffs;
  int n_attempt_type, n_success_type;

 private:
  double cur_aspect_ratio(double cell[3][3]);
  void pos_gmc_traj_prep();

  void pos_gmc_initial_integrate();
  void cell_initial_integrate();
  void type_initial_integrate();
  void pos_gmc_final_integrate();
  void cell_final_integrate();
  void type_final_integrate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
