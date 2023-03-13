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

FixStyle(ns/type,FixNSType)

#else

#ifndef LMP_FIX_NS_TYPE_H
#define LMP_FIX_NS_TYPE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNSType : public Fix {
 public:
  FixNSType(class LAMMPS *, int, char **);
  ~FixNSType();
  int setmask();
  virtual void init();
  virtual void pre_exchange();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  double compute_vector(int);

 protected:
  class Compute *pe_compute;
  int *prevtype;
  double **prevx;
  int seed;
  double Emax;
  class RanMars *random_g, *random_l;
  int semi_GC_flag;
  double *mu;
  int peflag;
  char *id_pe;
  char str[64];
  double dmuN, cumulative_dmuN;

  bool unequal_cutoffs;

  int n_attempt, n_success;
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
