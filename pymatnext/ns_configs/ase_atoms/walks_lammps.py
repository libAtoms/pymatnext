import warnings

import numpy as np

import lammps

# avoiding every lammps fix from having to do the full energy shift is useful
# we want
#   E + P V - mu N < Emax
# therefore, we modify Emax passed to lammps defining Emax^lammps, with V0 and N0 
#        defined as the initial volume and species numbers
#   E < Emax - P V0 + mu N0 = Emax^lammps
# this is sufficient for position moves, where V and N don't change
#
# for volume changes, we want
#   E + P (V0 + dV) - mu N0 < Emax
# where dV is the cumulative difference of the current steps' volume and the step where
#        the fix trajectory started
#   E < Emax - P (V0 + dV) + mu N0
#   E < Emax - P V0 + mu N0 - P dV
#   E < Emax^lammps - P dV
# which is equivalent to
#   E + P dV < Emax^lammps

def set_lammps_from_atoms(ns_atoms):
    """set lammps internal configuration data (cell, types, positions, velocities) from
    ns_atoms object

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        object containing configuration and lammps calculator to copoy from

    Returns
    -------
    types: np.ndarray(N_atoms, dtype=int)
        internal LAMMPS data with LAMMPS type of each atom
    pos: np.ndarray(N_atoms, d, dtype=float)
        internal LAMMPS data with atomic positions
    vel: np.ndarray(N_atoms, d, dtype=float)
        internal LAMMPS data with atomic velocities
    """

    atoms = ns_atoms.atoms

    cell_cmd = (f"change_box all     "
                f"x final 0 {atoms.cell[0, 0]} y final 0 {atoms.cell[1, 1]} z final 0 {atoms.cell[2, 2]}      "
                f"xy final {atoms.cell[1, 0]} xz final {atoms.cell[2, 0]} yz final {atoms.cell[2, 1]} units box")
    ns_atoms.calc.command(cell_cmd)

    # get per-atom pointers from LAMMPS
    types = ns_atoms.calc.numpy.extract_atom("type")
    pos = ns_atoms.calc.numpy.extract_atom("x")
    vel = ns_atoms.calc.numpy.extract_atom("v")
    # set them to current value
    types[:] = ns_atoms.type_of_Z[atoms.numbers]
    pos[:] = atoms.positions
    vel[:] = atoms.arrays["NS_velocities"]

    return types, pos, vel

def set_atoms_from_lammps(ns_atoms, types, pos, vel, E, F, update_energy_shift):
    """set ns_atoms.atoms object from lammps internal configuration data

    Stores new cell, atomic numbers, positions, velocities, as well as NS_energy, NS_forces,
    and optionally NS_energy_shift

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        object with configuration and calculator to store values in
    types: np.ndarray(N_atoms, dtype=int)
        internal LAMMPS data with LAMMPS type of each atom
    pos: np.ndarray(N_atoms, d, dtype=float)
        internal LAMMPS data with atomic positions
    vel: np.ndarray(N_atoms, d, dtype=float)
        internal LAMMPS data with atomic velocities
    E: float
        internal energy
    F: np.ndarray(N_atoms, 3, dtype=float)
        forces
    upadte_energy_shift: bool
        recalculate and store NS_energy_shift
    """
    atoms = ns_atoms.atoms

    atoms.numbers[:] = ns_atoms.Z_of_type[types]
    atoms.positions[...] = pos.reshape((-1, 3))
    atoms.arrays["NS_velocities"][...] = vel.reshape((-1, 3))
    atoms.info["NS_energy"][...] = E
    atoms.arrays["NS_forces"][...] = F

    box = ns_atoms.calc.extract_box()
    atoms.cell[0, 0] = box[1][0] - box[0][0]
    atoms.cell[1, 1] = box[1][1] - box[0][1]
    atoms.cell[2, 2] = box[1][2] - box[0][2]
    atoms.cell[1, 0] = box[2]
    atoms.cell[2, 1] = box[3]
    atoms.cell[2, 0] = box[4]

    if update_energy_shift:
        ns_atoms.atoms.info["NS_energy_shift"][...] = ns_atoms.calc_NS_energy_shift()

def extract_E_F(lmp, recalc):
    """extract energy and force calculation values from lammps

    Parameters
    ----------
    recalc: bool
        recalculate values before extracting

    Returns
    -------
    E: float
        internal energy
    F: np.ndarray(N_atoms, 3, dtype=float)
        atomic forces
    """
    if recalc:
        lmp.command("run 0 post no")

    return (lmp.extract_compute("pe", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR),
            lmp.numpy.extract_atom("f"))

def walk_pos_gmc(ns_atoms, Emax, rng):
    """walk atomic positions with GMC

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        atomic configuration
    Emax: float
        maximm shifted energy
    rng: np.Generator
        random number generator

    Returns
    -------
    [("pos_gmc_each_atom", 1, acc)] with acc 0 for rejected trajectory and 1 for accepted
    """
    ns_atoms.calc.command(f"unfix NS")
    atoms = ns_atoms.atoms
    # for LAMMPS RanMars RNG
    lammps_seed = rng.integers(1, 900000000)

    types, pos, vel = set_lammps_from_atoms(ns_atoms)

    Emax -= atoms.info["NS_energy_shift"]
    ns_atoms.calc.command(f"fix NS all ns/gmc {lammps_seed} {Emax}")
    ns_atoms.calc.command(f"timestep {ns_atoms.step_size['pos_gmc_each_atom']}")

    try:
        ns_atoms.calc.command(f"run {ns_atoms.walk_traj_len['gmc']} post no")
        # LAMMPS ns/gmc never rejects moves, just reflects, so already computed energy is correct
        E, F = extract_E_F(ns_atoms.calc, False)
        reject = (E >= Emax)
    except Exception as exc:
        warnings.warn(f"LAMMPS ns/gmc run raised exception {exc}")
        reject = True

    if not reject:
        # set atoms from current lammps internal state
        set_atoms_from_lammps(ns_atoms, types, pos, vel, E, F, False)
        # wrap to avoid atoms moving far enough in periodic images for lammps to lose them
        atoms.wrap()

    return [("pos_gmc_each_atom", 1, 0 if reject else 1)]

def walk_cell(ns_atoms, Emax, rng):
    """walk atomic cell with MC

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        atomic configuration
    Emax: float
        maximm shifted energy
    rng: np.Generator
        random number generator

    Returns
    -------
    [("cell_volume_per_atom", n_att, n_acc),
     ("cell_shear", n_att, n_acc),
     ("cell_stretch", n_att, n_acc)] with n_att and n_acc number of attempted and accepted
     move for each submove type
    """
    ns_atoms.calc.command(f"unfix NS")
    atoms = ns_atoms.atoms
    # for LAMMPS RanMars RNG
    lammps_seed = rng.integers(1, 900000000)

    step_size_volume = len(ns_atoms.atoms) * ns_atoms.step_size["cell_volume_per_atom"]
    step_size_shear = ns_atoms.step_size["cell_shear"]
    step_size_stretch = ns_atoms.step_size["cell_stretch"]

    types, pos, vel = set_lammps_from_atoms(ns_atoms)

    Emax -= atoms.info["NS_energy_shift"]
    move_params_cell = ns_atoms.move_params["cell"]
    submove_probs = move_params_cell["submove_probabilities"]
    # NOTE: would it be faster to construct more of this command ahead of time?
    # would either still have to do step_sizes here, or recreate command
    # every time step sizes are changed
    fix_cmd = (f"fix NS all ns/cellmc {lammps_seed} {Emax} {move_params_cell['min_aspect_ratio']} {ns_atoms.pressure} "
               f"{submove_probs['volume']} {step_size_volume} "
               f"{submove_probs['stretch']} {step_size_stretch} "
               f"{submove_probs['shear']} {step_size_shear} "
               f"{'yes' if ns_atoms.move_params['cell']['flat_V_prior'] else 'no'}")
    ns_atoms.calc.command(fix_cmd)

    try:
        ns_atoms.calc.command(f"run {ns_atoms.walk_traj_len['cell']} post no")
        failed = False
    except Exception as exc:
        warnings.warn(f"LAMMPS ns/cellmc run raised exception {exc}")
        failed = True

    if not failed:
        # LAMMPS ns/cellmc can reject moves, and if final move is rejected,
        # it won't update energy/force (but lammps _geometry_ should be correct).
        # Must recompute in case last move was rejected 
        E, F = extract_E_F(ns_atoms.calc, True)
        # set atoms from current lammps internal state
        set_atoms_from_lammps(ns_atoms, types, pos, vel, E, F, True)

    # gather number of attempted and accepted moves from fix
    n_att = {}
    n_acc = {}
    for submove_i, submove_type in enumerate(["volume", "stretch", "shear"]):
        n_att[submove_type] = int(ns_atoms.calc.extract_fix("NS", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR, 2 * submove_i + 0, 0))
        n_acc[submove_type] = int(ns_atoms.calc.extract_fix("NS", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR, 2 * submove_i + 1, 0))

    return [("cell_volume_per_atom", n_att["volume"], n_acc["volume"]),
            ("cell_shear", n_att["shear"], n_acc["shear"]),
            ("cell_stretch", n_att["stretch"], n_acc["stretch"])]

def walk_type(ns_atoms, Emax, rng):
    """walk atomic types with swap or semi-grand-canonical MC

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        atomic configuration
    Emax: float
        maximm shifted energy
    rng: np.Generator
        random number generator

    Returns
    -------
    []: empty list
    """
    ns_atoms.calc.command(f"unfix NS")
    atoms = ns_atoms.atoms
    # for LAMMPS RanMars RNG
    lammps_seed = rng.integers(1, 900000000)

    types, pos, vel = set_lammps_from_atoms(ns_atoms)

    Emax -= atoms.info["NS_energy_shift"]
    move_params_type = ns_atoms.move_params["type"]

    fix_cmd = (f"fix NS all ns/type {lammps_seed} {Emax} ")
    if move_params_type["sGC"]:
        # NOTE: if we don't mind being unable to change mu, we can construct this
        # string once when we set up the calculator
        fix_cmd += "yes " + " ".join([f"{mu:.10f}" for mu in ns_atoms.mu[ns_atoms.Z_of_type[1:]]])
    else:
        fix_cmd += "no"

    ns_atoms.calc.command(fix_cmd)
    try:
        ns_atoms.calc.command(f"run {ns_atoms.walk_traj_len['type']} post no")
        failed = False
    except Exception as exc:
        warnings.warn(f"LAMMPS ns/type run raised exception {exc}")
        failed = True

    if not failed:
        # LAMMPS ns/type can reject moves, and if final move is rejected,
        # it won't update energy/force (but lammps _geometry_ should be correct).
        # Must recompute in case last move was rejected 
        E, F = extract_E_F(ns_atoms.calc, True)
        # set atoms from current lammps internal state
        set_atoms_from_lammps(ns_atoms, types, pos, vel, E, F, True)

    # not actually always satisfied, e.g. if there's degeneracy (so some config 
    # starts with E == Emax) and a walk gets no accepted moves (so E is not lowered below Emax)
    #
    # assert E + ns_atoms.atoms.info["NS_energy_shift"] < Emax_orig:

    # NOT USED
    # gather number of attempted and accepted moves from fix
    # n_att = int(ns_atoms.calc.extract_fix("NS", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR, 0, 0))
    # n_acc = int(ns_atoms.calc.extract_fix("NS", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR, 1, 0))
    # print("BOB type step", n_acc, "/", n_att)

    # return an empty list, otherwise step-size setting code will try to use label to adjust a corresponding
    # step size
    return []
