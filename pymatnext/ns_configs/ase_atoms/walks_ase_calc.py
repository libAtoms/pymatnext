import warnings

import numpy as np

one_third = 1.0/3.0


def walk_pos_gmc(ns_atoms, Emax, rng):
    """Walk atomic positions using Galilean Monte-Carlo

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    """
    atoms = ns_atoms.atoms
    # new random velocities
    atoms.arrays["NS_velocities"][...] = rng.normal(scale=ns_atoms.step_size["pos_gmc_each_atom"], size=atoms.positions.shape)

    # below here operate only on _internal_ energy, without any "+ P V - mu N" shifts
    Emax -= atoms.info["NS_energy_shift"]
    atoms.calc.calculate(atoms, properties=["free_energy", "forces"])

    # store orig position in case move is rejected
    atoms.prev_positions[...] = atoms.positions
    n_failed_in_a_row = 0
    for i_step in range(ns_atoms.walk_traj_len["gmc"]):
        # step and evaluate new energy, forces
        atoms.positions += atoms.arrays["NS_velocities"]
        atoms.calc.calculate(atoms, properties=["free_energy", "forces"])
        E = atoms.calc.results["free_energy"]
        F = atoms.calc.results["forces"]

        if E >= Emax: # reflect or fail
            n_failed_in_a_row += 1
            if n_failed_in_a_row >= 2:
                break
            if np.sum(F*F) == 0:
                warnings.warn("Got F=0 while reflecting, giving up")
                break
            F_hat = F / np.sqrt(np.sum(F * F))
            atoms.arrays["NS_velocities"] -= F_hat * 2.0 * np.sum(atoms.arrays["NS_velocities"] * F_hat)
        else:
            n_failed_in_a_row = 0

    if n_failed_in_a_row > 0:
        # revert
        atoms.positions[...] = atoms.prev_positions

        return [("pos_gmc_each_atom", 1, 0)]
    else:
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = F

        return [("pos_gmc_each_atom", 1, 1)]


def _min_aspect_ratio(cell):
    """Calculate minimum aspect ratio of cell

    Parameters
    ----------
    cell: float (3,3) np.ndarray
        array of cell row vectors

    Returns
    -------
    minimum_aspect_ratio: float
    """
    crosses = np.asarray([np.cross(cell[i], cell[(i + 1) %3]) for i in range(3)])
    cross_norms = np.linalg.norm(crosses, axis=1)
    vol = np.abs(np.sum(cell[0] * np.cross(cell[1], cell[2])))
    return np.min(vol / cross_norms) / (vol ** one_third)


def _eval_and_accept_or_signal_revert(atoms, Emax, delta_PV=0.0, delta_muN=0.0):
    """Evaluate energy of configuration and accept it, updating NS_energy, NS_forces,
    and NS_energy_sfhit info/arrays entries, or signal that calling routine must revert move

    parameters
    ----------
    atoms: ase.atoms.Atoms
        atomic configuration
    Emax: float
        maximum shifted energy allowed

    Returns
    -------
    revert: bool, True if move is rejected and must be reverted
    """
    atoms.calc.calculate(atoms, properties=["free_energy", "forces"])
    E = atoms.calc.results["free_energy"]
    E_shift = atoms.info["NS_energy_shift"] + delta_PV - delta_muN
    if E + E_shift < Emax:
        # accept can happen here, because it's always the same action
        atoms.info["NS_energy_shift"][...] = E_shift
        atoms.info["NS_energy"][...] = E
        atoms.arrays["NS_forces"][...] = atoms.calc.results["forces"]
        return False
    else:
        # revert can be different, can only signal since it must be in the calling routine
        return True


def walk_cell(ns_atoms, Emax, rng):
    """Walk atom cell using Monte Carlo moves

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    """
    atoms = ns_atoms.atoms
    N_atoms = len(atoms)
    step_size_volume = N_atoms * ns_atoms.step_size["cell_volume_per_atom"]
    step_size_shear = ns_atoms.step_size["cell_shear"]
    step_size_stretch = ns_atoms.step_size["cell_stretch"]
    min_aspect_ratio = ns_atoms.move_params["cell"]["min_aspect_ratio"]
    flat_V_prior = ns_atoms.move_params["cell"]["flat_V_prior"]

    n_att = {"volume": 0, "shear": 0, "stretch": 0}
    n_acc = {"volume": 0, "shear": 0, "stretch": 0}

    submoves = list(ns_atoms.move_params["cell"]["submove_probabilities"].keys())
    probs = list(ns_atoms.move_params["cell"]["submove_probabilities"].values())
    for i_step in range(ns_atoms.walk_traj_len["cell"]):
        atoms.prev_cell[...] = atoms.cell.array
        move = rng.choice(submoves, p=probs)
        delta_V = 0.0
        reject = False
        n_att[move] += 1
        if move == "volume":
            orig_V = atoms.get_volume()
            new_V = orig_V + rng.normal(scale=step_size_volume)
            delta_V = new_V - orig_V
            V_scale = new_V / orig_V
            if V_scale < 0.5:
                # too small, reject
                continue
            elif not flat_V_prior and new_V < orig_V and rng.uniform() > np.pow(V_scale, N_atoms):
                # V^N prior
                continue
            new_cell = atoms.cell * (V_scale**one_third)
        elif move == "stretch":
            v_ind = rng.integers(3)
            rv = rng.normal(scale=step_size_stretch)
            F = np.eye(3)
            F[v_ind, v_ind] = np.exp(rv)
            v_next = (v_ind + 1 ) % 3
            F[v_next, v_next] = np.exp(-rv)
            new_cell = atoms.cell @ F
            if _min_aspect_ratio(new_cell) < min_aspect_ratio:
                # min aspect ratio
                continue
        elif move == "shear":
            new_cell = atoms.cell.array.copy()
            v_ind = rng.integers(3)

            v1 = atoms.prev_cell[(v_ind + 1) % 3]
            v1_norm = np.sqrt(np.sum(v1**2))
            dv = rng.normal(scale=step_size_shear) * v1 / v1_norm
            new_cell[v_ind] += dv

            v2 = atoms.prev_cell[(v_ind + 2) % 3]
            v2_norm = np.sqrt(np.sum(v2**2))
            dv = rng.normal(scale=step_size_shear) * v2 / v2_norm
            new_cell[v_ind] += dv
            if _min_aspect_ratio(new_cell) < min_aspect_ratio:
                # min aspect ratio
                continue

        # save positions if needed, then deform
        atoms.prev_positions[...] = atoms.positions
        atoms.set_cell(new_cell, True)

        if _eval_and_accept_or_signal_revert(atoms, Emax, delta_PV = ns_atoms.pressure * delta_V):
            atoms.positions[...] = atoms.prev_positions
            atoms.cell.array[...] = atoms.prev_cell
        else:
            n_acc[move] += 1

    return [("cell_volume_per_atom", n_att["volume"], n_acc["volume"]),
            ("cell_shear", n_att["shear"], n_acc["shear"]),
            ("cell_stretch", n_att["stretch"], n_acc["stretch"])]


def walk_type(ns_atoms, Emax, rng):
    """Walk atom type using swap or semi-Grand-Canonical Monte Carlo moves

    Parameters
    ----------
    ns_atoms: NSConfig_ASE_Atoms
        initial atomic configuration
    Emax: float
        maximum shifted energy
    rng: numpy.Generator
        random number generator
    """
    atoms = ns_atoms.atoms
    N_atoms = len(atoms)
    sGC = ns_atoms.move_params["type"]["sGC"]
    if sGC:
        Zs = ns_atoms.Zs
        n_Zs = len(Zs)
    else:
        # check that swaps are possible, otherwise give up early
        if sum(atoms.numbers == atoms.numbers[0]) == N_atoms:
            return ns_atoms.walk_traj_len["type"], 0

    n_succeeded = 0
    for i_step in range(ns_atoms.walk_traj_len["type"]):
        i0 = rng.integers(N_atoms)
        if sGC:
            Z0 = atoms.numbers[i0]
            type0 = Zs.index(Z0)
            type0_new = (type0 + 1 + rng.integers(n_Zs - 1)) % n_Zs
            atoms.numbers[i0] = Zs[type0_new]

            delta_muN = ns_atoms.mu[Zs[type0_new]] - ns_atoms.mu[Z0]

            if _eval_and_accept_or_signal_revert(atoms, Emax, delta_muN=delta_muN):
                # revert
                atoms.numbers[i0] = Zs[type0]
        else: # swap
            # pick an atom of a different type
            Z0 = atoms.numbers[i0]
            i1 = rng.choice(np.where(atoms.numbers != Z0)[0])
            atoms.numbers[i0] = atoms.numbers[i1]
            atoms.numbers[i1] = Z0

            if _eval_and_accept_or_signal_revert(atoms, Emax):
                # revert
                Z0 = atoms.numbers[i0]
                atoms.numbers[i0] = atoms.numbers[i1]
                atoms.numbers[i1] = Z0
            else:
                n_succeeded += 1

    return []
