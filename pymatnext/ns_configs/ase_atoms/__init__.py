import sys
import warnings
import re
import importlib
from copy import deepcopy
import collections
import io
import json

import numpy as np

import ase.data
import ase.io
import ase.units
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import all_changes

from .atoms_contig_store import AtomsContiguousStorage

from pymatnext.params import check_fill_defaults
from .ase_atoms_params import param_defaults_ase_atoms, param_defaults_walk

try:
    import lammps
except:
    lammps = None

class NSConfig_ASE_Atoms():
    """Nested sampling configuration class containing an Atoms object

    Parameters
    ----------
    params: dict
        parameters from [configs] toml section for creating Atoms (only required if source
        is "random") and [configs.calculator] toml section for calculator (always required)
    compression: float, default np.inf
        compression (n_walkers + 1 - n_cull) / (n_walkers + 1) for temperature calculation
    source: "random" / Atoms, default "random"
        source of atomic configuration
    rng: np.random.Generator, default None
        generator local to this process, required for initializing random configs
    kB: float, default ase.units.kB
        value of kB for temperature calculations
    allocate_only: bool, default False
        do not actually initialize content, just allocate [NOTE: maybe refactor to separate
        allocation and initialization?]
    """

    filename_suffix = ".extxyz"
    n_quantities = -1

    _max_E_hist = collections.deque(maxlen=1000)
    _walk_moves = ["gmc", "cell", "type"]
    _step_size_params = ["pos_gmc_each_atom", "cell_volume_per_atom", "cell_shear", "cell_stretch"]
    _Zs = []


    @staticmethod
    def _parse_composition(composition):
        """Parse a composition string or array into Zs and counts

        Parameters
        ----------
        composition: str / list(str) / list(Z)
            chemical formula or list of chemical symbols with optional integer number
            or list of atomic numbers

        Returns
        -------
        Zs: list(int) sorted list of atomic numbers
        counts: list(int) same order list of counts for each number
        """
        # parse composition and set up symbols and cls._Zs
        if isinstance(composition, str):
            composition_p = re.split(r"([A-Z][a-z]?[0-9]*)", re.sub(r"\s+", "", composition))
            if any([s != "" for s in composition_p[0::2]]):
                raise ValueError(f"Unknown characters in composition {composition}")
            composition_p = composition_p[1::2]
        else:
            composition_p = composition
        Zs_counts = []
        for species_n in composition_p:
            try:
                Z = int(species_n)
                count = 1
            except ValueError:
                m = re.match(r"([A-Z][a-z]?)([0-9]*)", species_n)
                if not m:
                    raise ValueError(f"Unable to parse composition element {species_n}")

                symbol = m.group(1)
                Z = ase.data.chemical_symbols.index(symbol)
                count = int(m.group(2) if len(m.group(2)) > 0 else 1)

            Zs_counts.append([Z, count])

        Zs_counts = sorted(Zs_counts, key=lambda Z_count: Z_count[0])
        Zs_counts = np.asarray(Zs_counts)

        return Zs_counts[:, 0], Zs_counts[:, 1]


    @classmethod
    def initialize(cls, params):
        """Initialize class attributes, in particular Zs and n_quantities

        Parameters
        ----------
        params: dict
            [configs] toml section
        """
        check_fill_defaults(params, param_defaults_ase_atoms, label="configs")
        full_composition = params["full_composition"]
        if len(full_composition) == 0:
            full_composition = params["composition"]
        cls._Zs, _ = cls._parse_composition(full_composition)

        # NS quantity = internal energy + P V - \sum_i \mu_i N_i
        # cell volume
        # N_atoms
        # composition of each species iff len(Zs) > 1
        cls.n_quantities = 3 + (len(cls._Zs) if len(cls._Zs) > 1 else 0)


    def __init__(self, params, compression=np.inf, source="random", rng=None, kB=ase.units.kB, allocate_only=False):
        check_fill_defaults(params, param_defaults_ase_atoms, label="configs")

        if len(self._Zs) == 0:
            NSConfig_ASE_Atoms.initialize(params)

        # Save params for later construction of the calculator
        # Also save calc_type, since it's needed for some other setup tasks,
        # but don't actually construct the calculator object until init_calculator() method
        # is called, since some calculator object types cannot be communicated by mpi4py
        self._params_calc = deepcopy(params["calculator"])
        self.calc_type = self._params_calc["type"]
        self.calc = None

        # for temperature
        NSConfig_ASE_Atoms.kB = kB
        NSConfig_ASE_Atoms.log_compression = np.log(compression)

        # needed (at least formally) for both random generation and from Atoms
        initial_rand_vol_per_atom = params["initial_rand_vol_per_atom"]

        if source == "random":
            if not allocate_only:
                assert rng is not None

            n_atoms = params["n_atoms"]
            n_dims = params["dims"]
            pbc = params["pbc"]
            initial_rand_min_dist = params["initial_rand_min_dist"]
            initial_rand_n_tries = params["initial_rand_n_tries"]

            # dimensions and PBC
            assert n_dims in [2, 3]

            assert len(pbc) == n_dims
            pbc += [False] * (3 - n_dims)

            # get composition
            Zs, Z_counts = self._parse_composition(params["composition"])

            numbers = []
            for Z, count in zip(Zs, Z_counts):
                numbers += [Z] * count

            if n_atoms % len(numbers) != 0:
                raise ValueError(f"composition {params['composition']} number of atoms {len(numbers)} not compatible with n_atoms {n_atoms}")

            # duplicate formula unit as much as needed
            numbers *= n_atoms // len(numbers)

            # create cell
            if allocate_only:
                cell = np.eye(3)
            else:
                cell = [(initial_rand_vol_per_atom * n_atoms) ** (1.0 / n_dims)] * n_dims + [0.0] * (3-n_dims)
                # perturb slightly
                F = np.eye(3) + np.diag(rng.normal(scale=0.01, size=3))
                # off diagonals apply twice, so scale by 1/sqrt(2) for equal magnitude relative to diagonal
                # NOTE: just hope that we're not violating min_aspect_ratio here
                F_off_diag = rng.normal(scale=0.01 / np.sqrt(2), size=(3,3))
                F_off_diag -= np.diag(np.diag(F_off_diag))
                F += F_off_diag
                cell = cell @ F

            # uniform positions
            if allocate_only:
                scaled_positions = np.zeros((n_atoms, 3))
            else:
                scaled_positions = rng.uniform(size=(n_atoms, 3))

            # create Atoms object
            self.atoms = AtomsContiguousStorage(numbers=numbers, cell=cell, scaled_positions=scaled_positions, pbc=pbc)

            if not allocate_only and initial_rand_min_dist is not None:
                d = neighbor_list('d', self.atoms, cutoff = initial_rand_min_dist, self_interaction = False)
                i_iter = 0
                while len(d) > 0 and i_iter < initial_rand_n_tries:
                    self.atoms.set_scaled_positions(rng.uniform(size=((n_atoms, 3))))
                    d = neighbor_list('d', self.atoms, cutoff = initial_rand_min_dist, self_interaction = False)
                    i_iter += 1
                if i_iter > 0:
                    if len(d) > 0:
                        raise RuntimeError(f"Failed to make initial configuration in {i_iter} tries")
                    warnings.warn(f"Required {i_iter} iterations to get a valid initial config")
        elif isinstance(source, Atoms):
            self.atoms = AtomsContiguousStorage(source)
        else:
            raise ValueError(f"NSConfig_ASE_Atoms from unknown source of type {type(source)}")

        # prepare for walks
        params_walk = params["walk"]
        check_fill_defaults(params_walk, param_defaults_walk, label="configs / walk")

        self._prep_walk(params_walk, vol_per_atom=initial_rand_vol_per_atom)
        self._rotate_to_lammps()

        # initialize space for NS quantities
        self.atoms.info["NS_quantities"] = np.zeros(self.n_quantities)
        # internal energy (without shift)
        self.atoms.info["NS_energy"] = np.asarray(0.0)
        # shifts like "+ P V - mu N" that need to be added to raw internal energy
        self.atoms.info["NS_energy_shift"] = np.asarray(0.0)
        self.atoms.new_array("NS_forces", np.zeros(self.atoms.positions.shape))

        self.atoms.make_contiguous()

        # space for quantities needed for NS but that shouldn't be communicated, and hence
        # not part of set that's made contiguous
        self.atoms.prev_positions = np.zeros(self.atoms.positions.shape)
        self.atoms.prev_cell = np.zeros(self.atoms.cell.array.shape)

        self.reset_walk_counters()


    def reset_walk_counters(self):
        """Reset attempted and successful step counters
        """

        self.n_att_acc = np.zeros((len(NSConfig_ASE_Atoms._step_size_params), 2), dtype=np.int64)


    def end_calculator(self):
        """Close any existing initialized calculator
        """

        if self.calc_type == "ASE":
            pass
        elif self.calc_type == "LAMMPS":
            if self.calc is not None:
                self.calc.close()
        else:
            raise NotImplementedError(f"Unknown calculator type {self.calc_type}")

        self.calc = None


    def init_calculator(self, skip_initial_store=False):
        """Initialize calculator.  Not part of constructor since some calculators (e.g. LAMMPS)
        cannot be pickled for mpi4py communication.

        Parameters
        ----------
        skip_initial_store: bool, default False
            skip storing the results of the initial calculation
        """

        params_calc = self._params_calc

        if self.calc_type == "ASE":
            # ASE Calculator
            calc_module = importlib.import_module(params_calc["args"]["module"])
            self.calc = calc_module.calc
        elif self.calc_type == "LAMMPS":
            lammps_header = params_calc["args"].get("header", ["units metal", "atom_style atomic", "atom_modify map array sort 0 0"]).copy()

            # base command args
            lammps_cmd_args = params_calc["args"].get("cmd_args", ["-echo", "log", "-screen", "none", "-nocite"])
            # special setting for log file, default none
            lammps_cmd_args.extend(["-log", params_calc["args"].get("log_file", "none")])

            lammps_name = params_calc["args"].get("name", "")
            self.calc = lammps.lammps(lammps_name, lammps_cmd_args)
            for cmd in lammps_header:
                self.calc.command(cmd)
            self.calc.command("boundary " + " ".join([params_calc["args"].get("boundary", "p")] * 3))
            self.calc.command("box tilt large")
            self.calc.command("region cell prism   0 1    0 1    0 1   0 0 0 units box")
            species_types = params_calc["args"]["types"]
            self.calc.command(f"create_box {len(species_types)} cell")
            self.calc.command("mass * 1.0")
            self.calc.command("neigh_modify delay 0 every 1 check yes")
            self.calc.command("compute pe all pe")

            lammps_cmds = params_calc["args"]["cmds"].copy()
            for cmd in lammps_cmds:
                try:
                    self.calc.command(cmd)
                except Exception as exc:
                    sys.stderr.write(f"LAMMPS exception while running command '{cmd}'\n")
                    raise

            # type for each Z
            # convert dict from chemical symbols to int Z
            Z_types = {}
            for species, species_type in species_types.items():
                try:
                    Z = int(species)
                except ValueError:
                    Z = ase.data.chemical_symbols.index(species)
                if species_type <= 0:
                    raise ValueError(f"type for species {Z} is {species_type}, must be > 0")
                Z_types[Z] = species_type

            # make numpy arrays for (presumably?) faster lookup
            self.type_of_Z = np.zeros(max(Z_types.keys()) + 1, dtype=int)
            self.Z_of_type = np.zeros(max(Z_types.values()) + 1, dtype=int)
            for Z, Z_type in Z_types.items():
                self.type_of_Z[Z] = Z_type
                self.Z_of_type[Z_type] = Z
            if len(species_types) > 0:
                for Z in self._Zs:
                    if self.type_of_Z[Z] == 0:
                        raise ValueError(f"Got composition Z={Z} and mapping of types {species_types} that does not include this species")

                    # also check backward mapping
                    assert self.Z_of_type[self.type_of_Z[Z]] != 0

        else:
            raise NotImplementedError(f"Unknown calculator type {self.calc_type}")

        # calculate energy/forces and save values
        self.initial_calc_and_store(skip_initial_store)


    def _rotate_to_lammps(self):
        """rotate to align with LAMMPS required orientation
        """
        cell = self.atoms.cell
        orig_cart = np.zeros((3,3))
        # original x is aligned with cell[0]
        orig_cart[0] = cell[0] / np.linalg.norm(cell[0])
        # original z is aligned with cell[0] x cell[1]
        orig_z = np.cross(cell[0], cell[1])
        orig_z /= np.linalg.norm(orig_z)
        orig_cart[2] = orig_z
        orig_cart[1] = np.cross(orig_cart[2], orig_cart[0])

        # R rotates orig cart to identity
        #     orig_cart @ R = eye
        #     R = orig_cart^{-1}
        R = np.linalg.inv(orig_cart)

        # rotate
        self.atoms.set_cell(cell @ R, True)


    def _prep_walk(self, params, vol_per_atom=None):
        """Set up data structures for walks based on params["configs"]["walk"]

        Parameters
        ----------
        params: dict
            information from [config.walk] toml section for step types and proportions in walk
        vol_per_atom: float, default None
            volume scale for setting default cell vol max step size. Required
            for default max step sizes for pos_gmc_each_atom or cell_volume_per_atom
            if they were not specified.
        """
        # trajectory lengths and (unnormalized) probabilities to achieve desired proportions
        self.walk_traj_len = {move: params[f"{move}_traj_len"] for move in NSConfig_ASE_Atoms._walk_moves}
        # list, so it can be used in Generator.choice, in same order as move type list
        self.walk_prob = np.asarray([params[f"{move}_proportion"] / self.walk_traj_len[move] for move in NSConfig_ASE_Atoms._walk_moves])

        if all([params[f"{move}_proportion"] == 0.0 for move in NSConfig_ASE_Atoms._walk_moves]):
            raise ValueError("At least some move must have proportion > 0")

        # probabilty check and normalization
        for move_i, move in enumerate(NSConfig_ASE_Atoms._walk_moves):
            assert self.walk_prob[move_i] >= 0.0
            if self.walk_prob[move_i] > 0.0:
                assert self.walk_traj_len[move] > 0

        self.walk_prob /= np.sum(self.walk_prob)

        self.move_params = {}

        # parameters for cell moves
        self.move_params["cell"] = deepcopy(params["cell"])
        normalization = sum(self.move_params["cell"]["submove_probabilities"].values())
        self.move_params["cell"]["submove_probabilities"] = {k: v / normalization for k, v in self.move_params["cell"]["submove_probabilities"].items()}
        if "pressure_GPa" in self.move_params["cell"]:
            if self.move_params["cell"].get("pressure", None) is not None:
                raise ValueError("Got both cell.pressure and cell.pressure_GPa")
            self.pressure = self.move_params["cell"]["pressure_GPa"] * ase.units.GPa
        elif "pressure" in self.move_params["cell"]:
            self.pressure = self.move_params["cell"]["pressure"]
        else:
            self.pressure = 0.0

        # mu always needs to be defined, since it'll be used for energy shift
        self.mu = np.zeros(len(ase.data.chemical_symbols))
        # parameters for type moves
        self.move_params["type"] = deepcopy(params["type"])
        if self.move_params["type"]["sGC"]:
            if self.move_params["type"].get("mu", {}) == 0:
                raise ValueError("if 'sGC' is specified, 'mu' is also required")
            mus = self.move_params["type"].pop("mu")
            Zs = [int(k) for k in mus.keys()]
            self.mu[Zs] = list(mus.values())

            assert set(Zs) == set(self._Zs)

        # max step sizes
        self.max_step_size = params["max_step_size"].copy()
        # max step size for position GMC and cell volume defaults are scaled to volume per atom
        if self.max_step_size["pos_gmc_each_atom"] < 0.0:
            self.max_step_size["pos_gmc_each_atom"] = (vol_per_atom ** (1.0/3.0)) / 10.0
        if self.max_step_size["cell_volume_per_atom"] < 0.0:
            self.max_step_size["cell_volume_per_atom"] = vol_per_atom / 20.0

        # actual step sizes
        self.step_size = params["step_size"].copy()
        # default to half the max for each type
        self.step_size = {k: (v if v >= 0.0 else self.max_step_size[k] / 2.0) for k, v in self.step_size.items()}

        assert set(list(self.max_step_size.keys())) == set(self._step_size_params)
        assert set(list(self.step_size.keys())) == set(self._step_size_params)

        # store function pointers for moves
        self.walk_func = {}
        if self.calc_type == "ASE":
            from .walks_ase_calc import walk_pos_gmc, walk_cell, walk_type
        elif self.calc_type == "LAMMPS":
            from .walks_lammps import walk_pos_gmc, walk_cell, walk_type
        else:
            raise NotImplementedError(f"Unknown calculator type {self.calc_type}")

        self.walk_func["gmc"] = walk_pos_gmc
        self.walk_func["cell"] = walk_cell
        self.walk_func["type"] = walk_type

        assert set(self.walk_func.keys()) == set(NSConfig_ASE_Atoms._walk_moves)

        if self.walk_prob[NSConfig_ASE_Atoms._walk_moves.index("gmc")] > 0.0:
            self.atoms.new_array("NS_velocities", np.zeros(self.atoms.positions.shape))


    def prepare(self):
        """do whatever is necessary to get ready for simulation. Here,
        make atom storage contiguous
        """

        self.atoms.make_contiguous()


    def initial_calc_and_store(self, skip_initial_store=False):
        """Calculate and store the results of a calculator, as well as other NS
        quantities, in info/arrays without breaking contiguous storage.  Currently
        hard-wired to free_energy in NS_energy and forces in NS_forces, as well
        as shift of raw energy related to volume (if constant pressure) and counts of
        each species (if constant chemical potential)

        Parameters
        ----------
        skip_initial_store: bool, default False
            skip storing the results of the initial calculation
        """
        if self.calc_type == "ASE":
            self.calc.calculate(self.atoms, properties=["free_energy", "forces"], system_changes=all_changes)
            if not skip_initial_store:
                self.atoms.info["NS_energy"][...] = self.calc.results.get("free_energy", self.calc.results.get("energy"))
                self.atoms.arrays["NS_forces"][...] = self.calc.results["forces"]
        elif self.calc_type == "LAMMPS":
            # WARNING: this duplicates code in walks_lammps.set_lammps_from_atoms, and at least at one point
            # there appears to have been a wrong implementation here.  Would be good to refactor somehow
            self.calc.reset_box([0.0, 0.0, 0.0], np.diag(self.atoms.cell), self.atoms.cell[1, 0], self.atoms.cell[2, 1], self.atoms.cell[2, 0])
            self.calc.create_atoms(len(self.atoms), list(np.arange(1, 1 + len(self.atoms))), self.type_of_Z[self.atoms.numbers],
                                   self.atoms.positions.reshape((-1)), self.atoms.arrays["NS_velocities"].reshape((-1)))
            self.calc.command("run 0")
            if not skip_initial_store:
                self.atoms.info["NS_energy"][...] = self.calc.extract_compute("pe", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR)
                self.atoms.arrays["NS_forces"][...] = self.calc.numpy.extract_atom("f")

            # do an initial 'fix NS' so that every walk can start with 'unfix NS'
            self.calc.command("fix NS all ns/gmc 1 0.0")

        if not skip_initial_store:
            self.atoms.info["NS_energy_shift"][...] = self.calc_NS_energy_shift()
            self.update_NS_quantities()


    def update_NS_quantities(self):
        """Update atoms.info["NS_quantities"] with current values
        """
        self.atoms.info["NS_quantities"][0] = self.atoms.info["NS_energy"] + self.atoms.info["NS_energy_shift"]
        self.atoms.info["NS_quantities"][1] = self.atoms.get_volume()
        N_atoms = len(self.atoms)
        self.atoms.info["NS_quantities"][2] = N_atoms
        if len(self._Zs) > 1:
            self.atoms.info["NS_quantities"][3:] = [sum(self.atoms.numbers == Z) / N_atoms for Z in self._Zs]


    def calc_NS_energy_shift(self):
        """Calculate current NS_energy_shift based on current volume and species
        """
        return self.pressure * self.atoms.get_volume() - np.sum(self.mu[self.atoms.numbers])


    @staticmethod
    def skip(fileobj):
        """skip a config, and report what NS iteration it came from

        Parameters
        ----------
        fileobj: IOBase
            open file object

        Returns
        -------
        NS_iter int, or None if not found
        """

        l = fileobj.readline()
        if not l:
            raise EOFError
        n = int(l.strip())

        l = fileobj.readline()
        if not l:
            raise EOFError
        comment = l.strip()

        m = re.search(r'\bNS_iter\s*=\s*([0-9]+)\b', comment)
        for i in range(n):
            l = fileobj.readline()
            if not l:
                raise EOFError

        if m:
            return int(m.group(1))
        else:
            return -1


    @staticmethod
    def read(fileobj, params):
        """read a file containing one or more configurations and create a list of NSConfig_ASE_Atoms
        objects from them

        Parameters
        ----------
        fileobj: str / Path / IOBase
            file (or its name) to read from
        params: dict
            parameters for creating Atoms (only [configs.calculator] toml section is required)

        Returns
        -------
        generator(NSConfig_ASE_Atoms): read-in configs
        """
        try:
            filename = fileobj.name
        except AttributeError:
            filename = str(fileobj)

        file_format = ase.io.formats.filetype(filename, read=False)
        for atoms in ase.io.iread(fileobj, ":", format=file_format, parallel=False):
            at = NSConfig_ASE_Atoms(params, source=atoms)
            at.step_size.update(json.loads(at.atoms.info.pop("_NS_step_size", "{}")))

            yield at


    def write(self, fileobj, extra_info={}, full_state=False):
        """write a configuration to file object

        Parameters
        ----------
        fileobj: str / Path / IOBase
            file (or its name) to write to
        extra_info: dict
            extra information to add to Atoms.info dict
        """
        self.atoms.info.update(extra_info)
        if full_state:
            self.atoms.info["_NS_step_size"] = json.dumps(self.step_size)

        try:
            filename = fileobj.name
        except AttributeError:
            filename = str(fileobj)
        try:
            ase.io.write(fileobj, self.atoms, format=ase.io.formats.filetype(filename, read=False), write_results=False, parallel=False)
        except ase.io.formats.UnknownFileTypeError:
            ase.io.write(fileobj, self.atoms, format="extxyz", write_results=False, parallel=False)

        for k in extra_info:
            del self.atoms.info[k]
        if full_state:
            del self.atoms.info["_NS_step_size"]


    def header_dict(self):
        """Header line for NS_samples file

        Returns
        -------
        header: dict
        """
        header_dict = { "n_extra_DOF_per_atom": 3,
                        "pressure": self.pressure,
                        "flat_V_prior": self.move_params["cell"]["flat_V_prior"],
                        "extras": [ "volume", "natoms"] + ([f"x_{Z}" for Z in self._Zs] if len(self._Zs) > 1 else []) }

        return header_dict


    def ns_quantities(self):
        """Update and return values of nested sampling quantity and other important quantities
        (volume, counts of each atomic number)

        Returns
        -------
        vals: list(float)
        """
        self.update_NS_quantities()
        return self.atoms.info["NS_quantities"]


    def send(self, to_rank, comm, MPI):
        """send configuration data to another process
        """
        comm.Send([self.atoms.contig_storage_int, MPI.INT64_T], to_rank, tag=17)
        comm.Send([self.atoms.contig_storage_float, MPI.DOUBLE], to_rank, tag=18)


    def recv(self, from_rank, comm, MPI):
        """receive configuration data from another process
        """
        comm.Recv([self.atoms.contig_storage_int, MPI.INT64_T], source=from_rank, tag=17)
        comm.Recv([self.atoms.contig_storage_float, MPI.DOUBLE], source=from_rank, tag=18)


    def backup(self):
        """Store configuration data in backup arrays

        Returns
        -------
        (contig_storage_int, contig_storage_float): np.ndarray copies of int and float storage
        """
        return self.atoms.contig_storage_int.copy(), self.atoms.contig_storage_float.copy()


    def restore(self, source):
        """Restore configuration data from backup arrays

        Parameters
        ----------
        source: tuple(np.ndarray, np.ndarray)
            2-element tuple with integer and float contiguous storage arrays from which to restore configuration
        """
        self.atoms.contig_storage_int[:] = source[0]
        self.atoms.contig_storage_float[:] = source[1]


    def copy_contents(self, source):
        """copy configuration data from another NSConfig_ASE_Atoms
        """
        self.atoms.contig_storage_int[:] = source.atoms.contig_storage_int
        self.atoms.contig_storage_float[:] = source.atoms.contig_storage_float


    def walk(self, Emax, walk_len, rng):
        """Walk a configuration

        Parameters
        ----------
        Emax: flat
            max energy for NS walk
        walk_len: int
            number of steps
        rng: np.Generator
            random number generator

        Returns
        -------
        np.ndarray(n_move_types, 2): array containing number of attempted moves and
        number of successful moves (2nd index 0 and 1) for each move type, with value
        of 1st index from position of that step size type in
        NSConfig_ASE_Atoms._step_size_params
        """
        # if we fixed the number of steps for every move type, and only varied proportions,
        # we could do all the rng in a single call
        self.atoms.calc = self.calc
        walk_len_so_far = 0
        while walk_len_so_far < walk_len:
            move = rng.choice(NSConfig_ASE_Atoms._walk_moves, p=self.walk_prob)

            n_att_acc_walk = self.walk_func[move](self, Emax, rng)
            # can this be done with a single numpy call somehow?
            for param, n_att, n_acc in n_att_acc_walk:
                self.n_att_acc[NSConfig_ASE_Atoms._step_size_params.index(param)] += (n_att, n_acc)

            walk_len_so_far += self.walk_traj_len[move]

        # move walks keep track of NS_energy_shift by accumulating changes.  It would definitely
        # be more stable to recalculate shift from scratch, although there's a chance it might lead to
        # energies above Emax.  Perhaps here we should do
        #   # recalculate to avoid energy shift drifting while being accumulated during walk
        #   self.atoms.info["NS_energy_shift"][...] = self.calc_NS_energy_shift()
        #   # make sure shifted energy <= Emax (is this enough, or do we need strictly <, and if so, how do we achieve that ?)
        #   self.atoms.info["NS_quantities"][0] = min(Emax, self.atoms.info["NS_energy"] + self.atoms.info["NS_energy_shift"])

        return self.n_att_acc


    @classmethod
    def report_store(cls, loop_iter, Emax):
        """Store quantities that are needed for reports on progress of NS iterations

        Parameters
        ----------
        loop_iter: int
            current loop iteration
        Emax: float
            current NS quantity maximum
        """
        cls._max_E_hist.append((loop_iter, Emax))


    @classmethod
    def report(cls):
        """Report on progress of NS iteration

        Returns
        -------
        T_report: str
            current estimated temperature
        """
        return f"T {cls.temperature()}"


    @classmethod
    def temperature(cls):
        """Calculate estimated current temperature

        Returns
        -------
        T: float or None
            current temperature (units depend on self.kB)
        """
        if len(cls._max_E_hist) > 1:
            beta = (cls._max_E_hist[-1][0] - cls._max_E_hist[0][0]) * cls.log_compression / (cls._max_E_hist[-1][1] - cls._max_E_hist[0][1])
            return 1.0 / (cls.kB * beta)
        else:
            return None


    @classmethod
    def new_configs_generator(cls, n_configs, params_configs, rng, configs_file):
        """Return a generator the returns a sequence of initial configurations

        Parameters
        ----------
        n_configs: int
            number of configurations to generate
        params_configs: dict
            parameters for creating configurations
        rng: np.random.Generator
            random number generator
        config_file: str / Path
            file containing configurations, or None to generate from source specified in params_configs

        Returns
        -------
        generator returning NSConfig objects
        """
        if configs_file is None:
            # source specified in params
            configs_file = params_configs.pop("file", None)

        if configs_file is not None:
            def new_configs_generator_file():
                with open(configs_file) as fin:
                    for config_i, config in enumerate(cls.read(fin, params_configs)):
                        if config_i >= n_configs:
                            raise RuntimeError(f"Found too many configs {config_i + 1} than requested "
                                               f"{n_configs} in file {configs_file}")
                        yield config

            return new_configs_generator_file

        else:
            def new_configs_generator_random():
                for i in range(n_configs):
                    yield cls(params_configs, rng=rng,
                                        compression=n_configs / (n_configs + 1))

            return new_configs_generator_random
