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

FixStyle(ns/cellmc,FixNSCellMC)

#else

#ifndef LMP_FIX_NS_CELLMC_H
#define LMP_FIX_NS_CELLMC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNSCellMC : public Fix {
 public:
  FixNSCellMC(class LAMMPS *, int, char **);
  ~FixNSCellMC();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  double compute_vector(int);

 protected:
  class Compute *pe_compute;
  double **prevx;
  double prev_boxlo[3], prev_boxhi[3], prev_xy, prev_yz, prev_xz;
  int seed;
  double Emax;
  double min_aspect_ratio, pressure;
  int flat_V_prior;
  class RanMars *random;
  double pVol, dVol, pStretch, dStretch, pShear, dShear;
  int peflag;
  char *id_pe;
  char str[64];

  double dPV, cumulative_dPV;

  bool move_rejected_early;
  int last_move_type;
  int n_attempt_vol, n_attempt_stretch, n_attempt_shear, n_success_vol, n_success_stretch, n_success_shear;

 private:
  double min_aspect_ratio_val(double cell[3][3]);
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
